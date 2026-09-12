import React from 'react'

interface Props {
  data:   number[]
  color?: string
  height?: number
  width?:  number
}

export function Sparkline({ data, color = '#4da6ff', height = 40, width = 120 }: Props) {
  if (!data || data.length < 2) return null

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1

  const pad = 3
  const w = width - pad * 2
  const h = height - pad * 2

  const points = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * w
    const y = pad + (1 - (v - min) / range) * h
    return `${x},${y}`
  })

  const pathD = `M ${points.join(' L ')}`

  // Fill area under the line
  const fillD = `M ${pad},${pad + h} L ${points.join(' L ')} L ${pad + w},${pad + h} Z`

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible' }}>
      <defs>
        <linearGradient id={`spark-fill-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3"/>
          <stop offset="100%" stopColor={color} stopOpacity="0"/>
        </linearGradient>
      </defs>
      {/* Fill */}
      <path d={fillD} fill={`url(#spark-fill-${color.replace('#', '')})`}/>
      {/* Line */}
      <path d={pathD} stroke={color} strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
      {/* Last point dot */}
      <circle
        cx={parseFloat(points[points.length - 1].split(',')[0])}
        cy={parseFloat(points[points.length - 1].split(',')[1])}
        r="2.5"
        fill={color}
      />
    </svg>
  )
}
