import React from 'react'

interface Props {
  label:  string
  value:  string | number
  color?: string
  sub?:   string
  icon?:  string
}

const COLOR_GLOW: Record<string, string> = {
  '#22d3a0': '0 0 24px rgba(34,211,160,0.2)',
  '#f0b429': '0 0 24px rgba(240,180,41,0.2)',
  '#ff4d6a': '0 0 24px rgba(255,77,106,0.2)',
  '#4da6ff': '0 0 24px rgba(77,166,255,0.2)',
  '#b57bff': '0 0 24px rgba(181,123,255,0.2)',
}

export function StatCard({ label, value, color = '#4da6ff', sub, icon }: Props) {
  const glow = COLOR_GLOW[color] ?? `0 0 24px ${color}33`

  return (
    <div style={{
      background:   'linear-gradient(135deg, #0d1520 0%, #121c2e 100%)',
      border:       `1px solid #1e2d45`,
      borderRadius: '14px',
      padding:      '20px 22px',
      minWidth:     '150px',
      flex:         1,
      position:     'relative',
      overflow:     'hidden',
      transition:   'transform 0.2s, box-shadow 0.2s',
      cursor:       'default',
    }}
    onMouseEnter={e => {
      (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)'
      ;(e.currentTarget as HTMLDivElement).style.boxShadow = glow
    }}
    onMouseLeave={e => {
      (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)'
      ;(e.currentTarget as HTMLDivElement).style.boxShadow = 'none'
    }}
    >
      {/* Top accent line */}
      <div style={{
        position:   'absolute',
        top:        0, left: 0, right: 0,
        height:     '2px',
        background: `linear-gradient(90deg, transparent, ${color}, transparent)`,
        opacity:    0.7,
      }} />

      {/* Glow blob */}
      <div style={{
        position:   'absolute',
        bottom:     '-20px', right: '-20px',
        width:      '80px', height: '80px',
        borderRadius: '50%',
        background: color,
        opacity:    0.04,
        filter:     'blur(20px)',
      }} />

      <div style={{ fontSize: '11px', color: '#6b82a8', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>
        {icon && <span style={{ marginRight: '5px' }}>{icon}</span>}
        {label}
      </div>
      <div style={{ fontSize: '34px', fontWeight: 700, color, lineHeight: 1, fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: '11px', color: '#6b82a8', marginTop: '6px' }}>{sub}</div>}
    </div>
  )
}
