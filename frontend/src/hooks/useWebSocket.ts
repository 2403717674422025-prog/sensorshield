import { useEffect, useRef, useCallback } from 'react'
import type { WsMessage } from '../types'

type MessageHandler = (msg: WsMessage) => void

/**
 * Manages a WebSocket connection to /ws/sensors.
 * Reconnects automatically after disconnect with exponential back-off.
 */
export function useWebSocket(onMessage: MessageHandler) {
  const wsRef        = useRef<WebSocket | null>(null)
  const handlerRef   = useRef(onMessage)
  const retryRef     = useRef(0)
  const unmountedRef = useRef(false)

  // Keep handler ref current without re-connecting on every render
  handlerRef.current = onMessage

  const connect = useCallback(() => {
    if (unmountedRef.current) return

    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url   = `${proto}://${window.location.host}/ws/sensors`
    const ws    = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      retryRef.current = 0
      // Send a ping every 30 s to keep the connection alive
      const ping = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping')
        else clearInterval(ping)
      }, 30_000)
    }

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as WsMessage
        handlerRef.current(msg)
      } catch { /* ignore malformed frames */ }
    }

    ws.onclose = () => {
      if (unmountedRef.current) return
      const delay = Math.min(1000 * 2 ** retryRef.current, 30_000)
      retryRef.current += 1
      setTimeout(connect, delay)
    }

    ws.onerror = () => ws.close()
  }, [])

  useEffect(() => {
    unmountedRef.current = false
    connect()
    return () => {
      unmountedRef.current = true
      wsRef.current?.close()
    }
  }, [connect])
}
