import React from 'react'

interface Props {
  type: string
  size?: number
  color?: string
  animated?: boolean
}

export function SensorIcon({ type, size = 32, color = '#4da6ff', animated = false }: Props) {
  const pulse = animated ? {
    animation: 'pulse 2s ease-in-out infinite',
  } : {}

  const icons: Record<string, JSX.Element> = {
    temperature: (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <circle cx="24" cy="36" r="8" fill={color} opacity="0.9" style={pulse}/>
        <rect x="20" y="8" width="8" height="24" rx="4" fill={color} opacity="0.6"/>
        <rect x="21" y="20" width="6" height="16" rx="3" fill={color} opacity="0.9"/>
        <line x1="28" y1="16" x2="34" y2="16" stroke={color} strokeWidth="2" strokeLinecap="round"/>
        <line x1="28" y1="22" x2="32" y2="22" stroke={color} strokeWidth="2" strokeLinecap="round"/>
        <line x1="28" y1="28" x2="34" y2="28" stroke={color} strokeWidth="2" strokeLinecap="round"/>
        <circle cx="24" cy="36" r="4" fill="white" opacity="0.4"/>
      </svg>
    ),
    pressure: (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <circle cx="24" cy="24" r="16" stroke={color} strokeWidth="2.5" opacity="0.5"/>
        <circle cx="24" cy="24" r="10" stroke={color} strokeWidth="2" opacity="0.7"/>
        <path d="M24 14 L24 20" stroke={color} strokeWidth="2.5" strokeLinecap="round"/>
        <path d="M34 24 L28 24" stroke={color} strokeWidth="2.5" strokeLinecap="round"/>
        <path d="M31.07 16.93 L26.83 21.17" stroke={color} strokeWidth="2" strokeLinecap="round"/>
        <circle cx="24" cy="24" r="3" fill={color} style={pulse}/>
        <path d="M24 24 L29 18" stroke={color} strokeWidth="2" strokeLinecap="round" opacity="0.8"/>
      </svg>
    ),
    vibration: (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <path d="M6 24 Q10 14 14 24 Q18 34 22 24 Q26 14 30 24 Q34 34 38 24 Q42 14 44 24"
          stroke={color} strokeWidth="2.5" fill="none" strokeLinecap="round"
          style={animated ? { animation: 'shimmer 1.5s ease-in-out infinite' } : {}}/>
        <circle cx="8" cy="24" r="2" fill={color} opacity="0.6"/>
        <circle cx="40" cy="24" r="2" fill={color} opacity="0.6"/>
        <line x1="24" y1="8" x2="24" y2="40" stroke={color} strokeWidth="1" opacity="0.2" strokeDasharray="3 3"/>
      </svg>
    ),
    current: (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <path d="M26 8 L14 26 L22 26 L20 40 L34 22 L26 22 Z"
          fill={color} opacity="0.85" style={pulse}/>
        <circle cx="24" cy="24" r="20" stroke={color} strokeWidth="1.5" opacity="0.15"/>
      </svg>
    ),
    flow: (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <path d="M8 24 Q16 14 24 24 Q32 34 40 24" stroke={color} strokeWidth="2.5"
          fill="none" strokeLinecap="round" opacity="0.9"/>
        <path d="M8 30 Q16 20 24 30 Q32 40 40 30" stroke={color} strokeWidth="2"
          fill="none" strokeLinecap="round" opacity="0.5"/>
        <path d="M8 18 Q16 8 24 18 Q32 28 40 18" stroke={color} strokeWidth="1.5"
          fill="none" strokeLinecap="round" opacity="0.3"/>
        <circle cx="40" cy="24" r="3" fill={color} style={pulse}/>
      </svg>
    ),
    humidity: (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <path d="M24 8 Q30 18 34 26 A10 10 0 0 1 14 26 Q18 18 24 8Z"
          fill={color} opacity="0.7" style={pulse}/>
        <path d="M18 28 Q22 24 26 28" stroke="white" strokeWidth="1.5"
          fill="none" strokeLinecap="round" opacity="0.5"/>
        <circle cx="24" cy="28" r="3" fill="white" opacity="0.3"/>
      </svg>
    ),
    voltage: (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <rect x="14" y="10" width="20" height="28" rx="4" stroke={color} strokeWidth="2" opacity="0.6"/>
        <rect x="19" y="6" width="10" height="6" rx="2" fill={color} opacity="0.7"/>
        <line x1="20" y1="20" x2="28" y2="20" stroke={color} strokeWidth="2" strokeLinecap="round"/>
        <line x1="20" y1="26" x2="28" y2="26" stroke={color} strokeWidth="2" strokeLinecap="round"/>
        <line x1="22" y1="32" x2="26" y2="32" stroke={color} strokeWidth="2" strokeLinecap="round"/>
        <circle cx="24" cy="26" r="2" fill={color} opacity="0.8" style={pulse}/>
      </svg>
    ),
    speed: (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <circle cx="24" cy="24" r="16" stroke={color} strokeWidth="2" opacity="0.4"/>
        <path d="M12 36 A16 16 0 0 1 36 36" stroke={color} strokeWidth="2.5"
          fill="none" strokeLinecap="round" opacity="0.8"/>
        <path d="M24 24 L32 16" stroke={color} strokeWidth="2.5" strokeLinecap="round"
          style={animated ? { transformOrigin: '24px 24px', animation: 'spin 2s linear infinite' } : {}}/>
        <circle cx="24" cy="24" r="3" fill={color} style={pulse}/>
        <circle cx="16" cy="36" r="2" fill={color} opacity="0.5"/>
        <circle cx="32" cy="36" r="2" fill={color} opacity="0.5"/>
        <circle cx="12" cy="26" r="1.5" fill={color} opacity="0.4"/>
        <circle cx="36" cy="26" r="1.5" fill={color} opacity="0.4"/>
      </svg>
    ),
    unknown: (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <circle cx="24" cy="24" r="16" stroke={color} strokeWidth="2" opacity="0.5"/>
        <circle cx="24" cy="34" r="2" fill={color} opacity="0.8"/>
        <path d="M18 18 A6 6 0 0 1 30 18 Q30 24 24 26 L24 30"
          stroke={color} strokeWidth="2.5" fill="none" strokeLinecap="round"/>
      </svg>
    ),
  }

  return icons[type] ?? icons['unknown']
}
