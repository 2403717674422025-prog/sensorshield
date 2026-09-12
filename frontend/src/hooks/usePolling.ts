import { useEffect, useRef } from 'react'

/**
 * Calls `fn` immediately and then every `intervalMs` milliseconds.
 * Cleans up on unmount. Replaces polling with WebSocket wherever possible,
 * but alerts/machines lists still use polling as a fallback.
 */
export function usePolling(fn: () => void, intervalMs: number) {
  const fnRef = useRef(fn)
  fnRef.current = fn

  useEffect(() => {
    fnRef.current()
    const id = setInterval(() => fnRef.current(), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
}
