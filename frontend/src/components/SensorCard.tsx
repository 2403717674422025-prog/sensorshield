import React, { useEffect, useRef, useState } from 'react'
import type { Sensor, SensorHealth } from '../types'
import { StatusBadge } from './StatusBadge'
import { HealthBar } from './HealthBar'
import { SensorIcon } from './SensorIcon'
import { Sparkline } from './Sparkline'

interface Props {
  sensor:       Sensor
  health?:      SensorHealth
  onClick:      () => void
  active:       boolean
  accentColor?: string
}

const STATUS_COLORS: Record<string, string> = {
  HEALTHY:      '#22d3a0',
  DRIFTING:     '#f0b429',
  NOISY:        '#f0b429',
  STUCK:        '#ff4d6a',
  INTERMITTENT: '#f0b429',
  MISSING:      '#ff4d6a',
  FAILED:       '#ff4d6a',
  UNKNOWN:      '#6b82a8',
}

export function SensorCard({ sensor, health, onClick, active, accentColor }: Props) {
  const score  = health?.health_score ?? null
  const status = health?.status ?? sensor.status
  const color  = accentColor ?? STATUS_COLORS[status] ?? '#4da6ff'
  const iconColor = STATUS_COLORS[status] ?? '#4da6ff'

  // Keep a small ring buffer of recent health scores for sparkline
  const sparkRef = useRef<number[]>([])
  const [sparkData, setSparkData] = useState<number[]>([])

  useEffect(() => {
    if (score !== null) {
      sparkRef.current = [...sparkRef.current.slice(-19), score]
      setSparkData([...sparkRef.current])
    }
  }, [score])

  const borderColor = active
    ? color
    : status === 'HEALTHY' ? `${color}20`
    : status === 'FAILED' || status === 'MISSING' ? '#ff4d6a30'
    : '#1e2d45'

  return (
    <div
      onClick={onClick}
      style={{
        background:   active
          ? `linear-gradient(135deg, ${color}12 0%, #121c2e 100%)`
          : 'linear-gradient(135deg, #0d1520 0%, #0a1220 100%)',
        border:       `1px solid ${borderColor}`,
        borderRadius: '14px',
        padding:      '14px',
        cursor:       'pointer',
        transition:   'all 0.2s ease',
        position:     'relative',
        overflow:     'hidden',
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLDivElement).style.borderColor = `${color}50`
        ;(e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)'
        ;(e.currentTarget as HTMLDivElement).style.boxShadow = `0 8px 24px ${color}15`
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLDivElement).style.borderColor = borderColor
        ;(e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)'
        ;(e.currentTarget as HTMLDivElement).style.boxShadow = 'none'
      }}
    >
      {/* Top accent line */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '2px',
        background: `linear-gradient(90deg, transparent, ${iconColor}80, transparent)`,
      }}/>

      {/* Sparkline background */}
      {sparkData.length > 2 && (
        <div style={{
          position: 'absolute', bottom: 0, right: 0,
          opacity: 0.15, pointerEvents: 'none',
        }}>
          <Sparkline data={sparkData} color={iconColor} width={100} height={50}/>
        </div>
      )}

      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '36px', height: '36px',
            background: `${iconColor}15`,
            borderRadius: '10px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
            border: `1px solid ${iconColor}25`,
          }}>
            <SensorIcon type={sensor.sensor_type} size={22} color={iconColor} animated={status === 'HEALTHY'}/>
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '12px', letterSpacing: '0.3px', color: '#e2eaf6' }}>
              {sensor.id}
            </div>
            <div style={{ color: '#3d5278', fontSize: '10px', marginTop: '1px' }}>
              {sensor.name ?? sensor.sensor_type} · {sensor.unit}
            </div>
          </div>
        </div>
        <StatusBadge value={status} size="sm"/>
      </div>

      {/* Health score + bar */}
      {score !== null ? (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', alignItems: 'center' }}>
            <span style={{ fontSize: '10px', color: '#3d5278', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Health</span>
            <span style={{
              fontSize: '14px', fontWeight: 800,
              color: score >= 70 ? '#22d3a0' : score >= 50 ? '#f0b429' : '#ff4d6a',
              fontVariantNumeric: 'tabular-nums',
            }}>
              {score.toFixed(1)}
            </span>
          </div>
          <HealthBar score={score}/>

          {/* Sparkline visible row */}
          {sparkData.length > 3 && (
            <div style={{ marginTop: '8px', opacity: 0.7 }}>
              <Sparkline data={sparkData} color={iconColor} width={160} height={32}/>
            </div>
          )}
        </>
      ) : (
        <div style={{ fontSize: '11px', color: '#1e2d45', fontStyle: 'italic', marginTop: '4px' }}>
          Awaiting data…
        </div>
      )}

      {health?.reason && (
        <div style={{
          fontSize: '9px', color: '#3d5278', marginTop: '6px',
          lineHeight: 1.5, paddingTop: '6px',
          borderTop: '1px solid #1a2640',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}>
          {health.reason}
        </div>
      )}
    </div>
  )
}
