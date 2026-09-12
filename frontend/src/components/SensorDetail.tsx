import React, { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
} from 'recharts'
import type { Sensor, SensorHealth, SensorReading } from '../types'
import { api } from '../api'
import { StatusBadge } from './StatusBadge'

interface Props {
  sensor: Sensor
}

function fmt(ts: string) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function SensorDetail({ sensor }: Props) {
  const [readings, setReadings] = useState<SensorReading[]>([])
  const [health,   setHealth]   = useState<SensorHealth[]>([])
  const [loading,  setLoading]  = useState(true)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      api.sensorHistory(sensor.id, 60),
      api.sensorHealth(sensor.id, 60),
    ]).then(([r, h]) => {
      setReadings([...r].reverse())
      setHealth([...h].reverse())
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [sensor.id])

  if (loading) return <div style={{ color: '#8b949e', padding: '20px' }}>Loading…</div>

  const readingData = readings.map(r => ({ time: fmt(r.timestamp), value: r.value }))
  const healthData  = health.map(h => ({
    time:        fmt(h.timestamp),
    health:      h.health_score,
    anomaly:     h.anomaly_score,
    drift:       h.drift_score,
    noise:       h.noise_score,
  }))

  const latest = health[health.length - 1]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: '16px' }}>{sensor.id}</div>
          <div style={{ color: '#8b949e', fontSize: '13px' }}>{sensor.name} · {sensor.unit}</div>
        </div>
        {latest && <StatusBadge value={latest.status} />}
      </div>

      {/* Sub-scores */}
      {latest && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(130px,1fr))', gap: '10px' }}>
          {[
            { label: 'Health',      val: latest.health_score },
            { label: 'Drift',       val: latest.drift_score },
            { label: 'Noise',       val: latest.noise_score },
            { label: 'Consistency', val: latest.consistency_score },
            { label: 'Missing',     val: latest.missing_data_score },
            { label: 'Anomaly',     val: latest.anomaly_score },
          ].map(({ label, val }) => val != null && (
            <div key={label} style={{
              background: '#21262d', borderRadius: '6px', padding: '10px 12px', textAlign: 'center',
            }}>
              <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>{label}</div>
              <div style={{
                fontSize: '20px', fontWeight: 700,
                color: val >= 70 ? '#3fb950' : val >= 50 ? '#d29922' : '#f85149',
              }}>
                {val.toFixed(0)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Raw readings chart */}
      {readingData.length > 0 && (
        <div>
          <div style={{ fontSize: '12px', color: '#8b949e', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Raw Readings
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={readingData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#8b949e' }} />
              <YAxis tick={{ fontSize: 10, fill: '#8b949e' }} />
              <Tooltip
                contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '6px' }}
                labelStyle={{ color: '#8b949e' }}
              />
              <Line type="monotone" dataKey="value" stroke="#58a6ff" dot={false} strokeWidth={1.5} />
              {sensor.min_valid_value != null && (
                <ReferenceLine y={sensor.min_valid_value} stroke="#f85149" strokeDasharray="4 2" strokeWidth={1} />
              )}
              {sensor.max_valid_value != null && (
                <ReferenceLine y={sensor.max_valid_value} stroke="#f85149" strokeDasharray="4 2" strokeWidth={1} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Health score trend */}
      {healthData.length > 0 && (
        <div>
          <div style={{ fontSize: '12px', color: '#8b949e', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Health Score Trend
          </div>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={healthData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#8b949e' }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#8b949e' }} />
              <Tooltip
                contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '6px' }}
                labelStyle={{ color: '#8b949e' }}
              />
              <ReferenceLine y={70} stroke="#3fb950" strokeDasharray="4 2" strokeWidth={1} />
              <ReferenceLine y={50} stroke="#d29922" strokeDasharray="4 2" strokeWidth={1} />
              <ReferenceLine y={30} stroke="#f85149" strokeDasharray="4 2" strokeWidth={1} />
              <Line type="monotone" dataKey="health" stroke="#3fb950"  dot={false} strokeWidth={1.5} name="Health" />
              <Line type="monotone" dataKey="drift"  stroke="#d29922"  dot={false} strokeWidth={1}   name="Drift" />
              <Line type="monotone" dataKey="noise"  stroke="#bc8cff"  dot={false} strokeWidth={1}   name="Noise" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {latest?.reason && (
        <div style={{
          background: '#161b22', border: '1px solid #30363d',
          borderRadius: '6px', padding: '12px 14px',
          fontSize: '13px', color: '#8b949e', lineHeight: 1.5,
        }}>
          <strong style={{ color: '#e6edf3' }}>Diagnostics:</strong> {latest.reason}
        </div>
      )}
    </div>
  )
}
