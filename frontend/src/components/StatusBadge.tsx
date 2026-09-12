import React from 'react'
import type { SensorStatus, AlertSeverity, TrustStatus } from '../types'

const STATUS_CONFIG: Record<string, { color: string; bg: string; dot?: boolean }> = {
  HEALTHY:      { color: '#22d3a0', bg: 'rgba(34,211,160,0.12)',  dot: true },
  DRIFTING:     { color: '#f0b429', bg: 'rgba(240,180,41,0.12)' },
  NOISY:        { color: '#f0b429', bg: 'rgba(240,180,41,0.12)' },
  STUCK:        { color: '#ff4d6a', bg: 'rgba(255,77,106,0.12)' },
  INTERMITTENT: { color: '#f0b429', bg: 'rgba(240,180,41,0.12)' },
  MISSING:      { color: '#ff4d6a', bg: 'rgba(255,77,106,0.12)' },
  FAILED:       { color: '#ff4d6a', bg: 'rgba(255,77,106,0.12)' },
  UNKNOWN:      { color: '#6b82a8', bg: 'rgba(107,130,168,0.12)' },
  INFO:         { color: '#4da6ff', bg: 'rgba(77,166,255,0.12)' },
  WARNING:      { color: '#f0b429', bg: 'rgba(240,180,41,0.12)' },
  CRITICAL:     { color: '#ff4d6a', bg: 'rgba(255,77,106,0.12)' },
  TRUSTED:      { color: '#22d3a0', bg: 'rgba(34,211,160,0.12)',  dot: true },
  CAUTION:      { color: '#f0b429', bg: 'rgba(240,180,41,0.12)' },
  UNTRUSTED:    { color: '#ff4d6a', bg: 'rgba(255,77,106,0.12)' },
  ACTIVE:       { color: '#ff4d6a', bg: 'rgba(255,77,106,0.12)' },
  ACKNOWLEDGED: { color: '#f0b429', bg: 'rgba(240,180,41,0.12)' },
  RESOLVED:     { color: '#22d3a0', bg: 'rgba(34,211,160,0.12)' },
  ONLINE:       { color: '#22d3a0', bg: 'rgba(34,211,160,0.12)',  dot: true },
  OFFLINE:      { color: '#6b82a8', bg: 'rgba(107,130,168,0.12)' },
}

interface Props {
  value: SensorStatus | AlertSeverity | TrustStatus | string
  size?: 'sm' | 'md'
}

export function StatusBadge({ value, size = 'md' }: Props) {
  const cfg = STATUS_CONFIG[value] ?? { color: '#6b82a8', bg: 'rgba(107,130,168,0.12)' }
  const pad = size === 'sm' ? '3px 8px' : '4px 10px'
  const fs  = size === 'sm' ? '10px' : '11px'

  return (
    <span style={{
      display:      'inline-flex',
      alignItems:   'center',
      gap:          '5px',
      padding:      pad,
      borderRadius: '20px',
      fontSize:     fs,
      fontWeight:   700,
      letterSpacing:'0.5px',
      color:        cfg.color,
      background:   cfg.bg,
      border:       `1px solid ${cfg.color}30`,
      whiteSpace:   'nowrap',
      textTransform:'uppercase',
    }}>
      {cfg.dot && (
        <span style={{
          width: '5px', height: '5px',
          borderRadius: '50%',
          background: cfg.color,
          display: 'inline-block',
          boxShadow: `0 0 6px ${cfg.color}`,
          animation: value === 'HEALTHY' || value === 'TRUSTED' || value === 'ONLINE'
            ? 'pulse 2s ease-in-out infinite' : 'none',
        }} />
      )}
      {value}
    </span>
  )
}
