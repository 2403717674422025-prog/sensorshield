import React from 'react'
import type { Alert } from '../types'
import { StatusBadge } from './StatusBadge'

interface Props {
  alert: Alert
  onAck: (id: number) => void
}

const SEVERITY_LEFT: Record<string, string> = {
  CRITICAL: '#ff4d6a',
  WARNING:  '#f0b429',
  INFO:     '#4da6ff',
}

const FAULT_ICONS: Record<string, string> = {
  DRIFT:                    '📉',
  BIAS:                     '⚖️',
  NOISE:                    '〰️',
  STUCK:                    '🔒',
  MISSING:                  '❌',
  INTERMITTENT:             '⚡',
  SPIKE:                    '📈',
  SAMPLING_FAILURE:         '🔄',
  CROSS_SENSOR_INCONSISTENCY:'🔀',
  LOW_RELIABILITY:          '⚠️',
  LOW_PREDICTION_TRUST:     '🤔',
  UNKNOWN:                  '❓',
}

export function AlertRow({ alert, onAck }: Props) {
  const ts    = new Date(alert.timestamp).toLocaleString()
  const left  = SEVERITY_LEFT[alert.severity] ?? '#4da6ff'
  const icon  = FAULT_ICONS[alert.fault_type] ?? '⚠️'

  return (
    <div style={{
      background:   'linear-gradient(135deg, #0d1520 0%, #111926 100%)',
      border:       '1px solid #1e2d45',
      borderLeft:   `3px solid ${left}`,
      borderRadius: '12px',
      padding:      '16px 18px',
      display:      'grid',
      gridTemplateColumns: '1fr auto',
      gap:          '16px',
      alignItems:   'start',
      transition:   'border-color 0.2s',
      position:     'relative',
      overflow:     'hidden',
    }}>
      {/* Glow */}
      <div style={{
        position: 'absolute', top: 0, right: 0,
        width: '100px', height: '100px',
        background: left,
        opacity: 0.02,
        filter: 'blur(30px)',
        borderRadius: '50%',
      }} />

      <div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '16px' }}>{icon}</span>
          <StatusBadge value={alert.severity} size="sm" />
          <span style={{ fontWeight: 700, fontSize: '13px', color: '#e2eaf6' }}>
            {alert.sensor_id ?? alert.machine_id ?? 'System'}
          </span>
          <span style={{
            fontSize: '11px', color: '#6b82a8',
            background: '#1a2640', padding: '2px 8px',
            borderRadius: '20px', border: '1px solid #1e2d45',
          }}>
            {alert.fault_type}
          </span>
          <span style={{ color: '#3d5278', fontSize: '11px', marginLeft: 'auto' }}>{ts}</span>
        </div>

        <div style={{ fontSize: '13px', color: '#c5d5e8', marginBottom: '6px', lineHeight: 1.5 }}>
          {alert.description}
        </div>

        {alert.impact && (
          <div style={{
            fontSize: '12px', color: '#6b82a8', marginTop: '6px',
            padding: '8px 10px',
            background: '#0d1520',
            borderRadius: '8px',
            border: '1px solid #1e2d45',
            lineHeight: 1.5,
          }}>
            <span style={{ color: '#4da6ff', fontWeight: 600 }}>Impact: </span>
            {alert.impact}
          </div>
        )}

        {alert.recommended_action && (
          <div style={{
            fontSize: '12px', color: '#6b82a8', marginTop: '6px',
            padding: '8px 10px',
            background: '#0d1520',
            borderRadius: '8px',
            border: '1px solid #1e2d45',
            lineHeight: 1.5,
          }}>
            <span style={{ color: '#22d3a0', fontWeight: 600 }}>Action: </span>
            {alert.recommended_action}
          </div>
        )}

        {alert.health_score_at_alert != null && (
          <div style={{ fontSize: '11px', color: '#3d5278', marginTop: '6px' }}>
            Health at alert: <span style={{ color: '#6b82a8', fontWeight: 600 }}>{alert.health_score_at_alert.toFixed(1)}</span>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'flex-end' }}>
        <StatusBadge value={alert.status} size="sm" />
        {alert.status === 'ACTIVE' && (
          <button
            onClick={() => onAck(alert.id)}
            style={{
              background:   'transparent',
              border:       '1px solid #243352',
              borderRadius: '8px',
              color:        '#6b82a8',
              padding:      '5px 12px',
              fontSize:     '11px',
              fontWeight:   600,
              letterSpacing:'0.3px',
              cursor:       'pointer',
              transition:   'all 0.15s',
              whiteSpace:   'nowrap',
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = '#4da6ff'
              ;(e.currentTarget as HTMLButtonElement).style.color = '#4da6ff'
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = '#243352'
              ;(e.currentTarget as HTMLButtonElement).style.color = '#6b82a8'
            }}
          >
            Acknowledge
          </button>
        )}
      </div>
    </div>
  )
}
