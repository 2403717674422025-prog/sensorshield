import React, { useState } from 'react'
import type { Sensor, SensorHealth, Machine } from '../types'
import { SensorCard } from './SensorCard'
import { StatusBadge } from './StatusBadge'

interface Props {
  machine:        Machine
  sensors:        Sensor[]
  latestHealth:   Record<string, SensorHealth>
  selectedSensor: string | null
  onSelect:       (id: string) => void
}

const MACHINE_ICONS: Record<string, string> = {
  M001: '🔧',
  M002: '⚙️',
  M003: '🏭',
  M004: '💨',
}

const MACHINE_COLORS: Record<string, string> = {
  M001: '#4da6ff',
  M002: '#22d3a0',
  M003: '#f0b429',
  M004: '#b57bff',
}

export function MachinePanel({ machine, sensors, latestHealth, selectedSensor, onSelect }: Props) {
  const [collapsed, setCollapsed] = useState(false)
  const color = MACHINE_COLORS[machine.id] ?? '#4da6ff'
  const icon  = MACHINE_ICONS[machine.id] ?? '🔩'

  const healthScores  = sensors.map(s => latestHealth[s.id]?.health_score).filter(Boolean) as number[]
  const avgHealth     = healthScores.length > 0
    ? Math.round(healthScores.reduce((a, b) => a + b, 0) / healthScores.length)
    : null
  const healthyCount  = sensors.filter(s => (latestHealth[s.id]?.status ?? s.status) === 'HEALTHY').length
  const criticalCount = sensors.filter(s => ['FAILED','MISSING','STUCK'].includes(latestHealth[s.id]?.status ?? s.status)).length

  return (
    <div style={{
      background:   'linear-gradient(135deg, #0a1628 0%, #0d1c30 100%)',
      border:       `1px solid ${color}25`,
      borderRadius: '16px',
      overflow:     'hidden',
      marginBottom: '20px',
    }}>
      {/* Machine header */}
      <div
        onClick={() => setCollapsed(c => !c)}
        style={{
          padding:    '18px 22px',
          display:    'flex',
          alignItems: 'center',
          gap:        '14px',
          cursor:     'pointer',
          borderBottom: collapsed ? 'none' : `1px solid ${color}15`,
          background: `linear-gradient(135deg, ${color}08 0%, transparent 100%)`,
          position:   'relative',
          overflow:   'hidden',
        }}
      >
        {/* Top accent */}
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: '2px',
          background: `linear-gradient(90deg, transparent, ${color}, transparent)`,
          opacity: 0.6,
        }}/>

        {/* Icon */}
        <div style={{
          width:        '44px', height: '44px',
          borderRadius: '12px',
          background:   `${color}18`,
          border:       `1px solid ${color}30`,
          display:      'flex', alignItems: 'center', justifyContent: 'center',
          fontSize:     '22px',
          flexShrink:   0,
        }}>
          {icon}
        </div>

        {/* Info */}
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontWeight: 800, fontSize: '14px', color: '#e2eaf6' }}>{machine.id}</span>
            <span style={{ fontSize: '13px', color: '#6b82a8' }}>—</span>
            <span style={{ fontSize: '13px', color: '#c5d5e8', fontWeight: 600 }}>{machine.name}</span>
            <StatusBadge value={machine.status} size="sm" />
          </div>
          <div style={{ fontSize: '11px', color: '#3d5278', marginTop: '2px' }}>
            📍 {machine.location ?? 'Unknown location'}
          </div>
        </div>

        {/* Stats */}
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '18px', fontWeight: 800, color }}>
              {avgHealth ?? '—'}
            </div>
            <div style={{ fontSize: '10px', color: '#3d5278', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Health</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '18px', fontWeight: 800, color: '#22d3a0' }}>{healthyCount}</div>
            <div style={{ fontSize: '10px', color: '#3d5278', textTransform: 'uppercase', letterSpacing: '0.5px' }}>OK</div>
          </div>
          {criticalCount > 0 && (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '18px', fontWeight: 800, color: '#ff4d6a' }}>{criticalCount}</div>
              <div style={{ fontSize: '10px', color: '#3d5278', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Critical</div>
            </div>
          )}
          <div style={{
            width: '28px', height: '28px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#3d5278', fontSize: '14px',
            transition: 'transform 0.2s',
            transform: collapsed ? 'rotate(-90deg)' : 'rotate(0deg)',
          }}>
            ▼
          </div>
        </div>
      </div>

      {/* Sensors grid */}
      {!collapsed && (
        <div style={{
          display:             'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
          gap:                 '12px',
          padding:             '16px 18px',
        }}>
          {sensors.map(s => (
            <SensorCard
              key={s.id}
              sensor={s}
              health={latestHealth[s.id]}
              active={selectedSensor === s.id}
              onClick={() => onSelect(s.id)}
              accentColor={color}
            />
          ))}
        </div>
      )}
    </div>
  )
}
