import React from 'react'

interface Props { score: number }

export function HealthBar({ score }: Props) {
  const pct   = Math.max(0, Math.min(100, score))
  const color = pct >= 70 ? '#22d3a0' : pct >= 50 ? '#f0b429' : '#ff4d6a'
  const glow  = pct >= 70
    ? '0 0 8px rgba(34,211,160,0.5)'
    : pct >= 50
    ? '0 0 8px rgba(240,180,41,0.5)'
    : '0 0 8px rgba(255,77,106,0.5)'

  return (
    <div style={{
      height:       '5px',
      background:   '#1a2640',
      borderRadius: '10px',
      overflow:     'hidden',
    }}>
      <div style={{
        width:        `${pct}%`,
        height:       '100%',
        background:   `linear-gradient(90deg, ${color}88, ${color})`,
        borderRadius: '10px',
        boxShadow:    glow,
        transition:   'width 0.6s ease',
      }} />
    </div>
  )
}
