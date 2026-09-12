import React, { useState, useEffect, useCallback, useRef } from 'react'
import type { Sensor, SensorHealth, Alert, Prediction, WsMessage, SystemSummary, Machine } from './types'
import { api } from './api'
import { useWebSocket } from './hooks/useWebSocket'
import { usePolling }   from './hooks/usePolling'
import { StatCard }     from './components/StatCard'
import { SensorCard }   from './components/SensorCard'
import { SensorDetail } from './components/SensorDetail'
import { AlertRow }     from './components/AlertRow'
import { MachinePanel } from './components/MachinePanel'

// ---------------------------------------------------------------------------
type Tab = 'overview' | 'sensors' | 'alerts'

const NAV: { id: Tab; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'sensors',  label: 'Sensors'  },
  { id: 'alerts',   label: 'Alerts'   },
]

// ---------------------------------------------------------------------------
export default function App() {
  const [tab,            setTab]           = useState<Tab>('overview')
  const [sensors,        setSensors]       = useState<Sensor[]>([])
  const [machines,       setMachines]      = useState<Machine[]>([])
  const [latestHealth,   setLatestHealth]  = useState<Record<string, SensorHealth>>({})
  const [alerts,         setAlerts]        = useState<Alert[]>([])
  const [predictions,    setPredictions]   = useState<Prediction[]>([])
  const [selectedSensor, setSelected]      = useState<string | null>(null)
  const [wsStatus,       setWsStatus]      = useState<'connecting' | 'live' | 'offline'>('connecting')
  const [liveEvents,     setLiveEvents]    = useState(0)
  const liveEventsRef    = useRef(0)

  // ── REST data loaders ──────────────────────────────────────────────────
  const loadSensors = useCallback(async () => {
    try {
      const s = await api.sensors()
      setSensors(s)
      const m = await api.machines()
      setMachines(m)
      // Fetch latest health for every sensor
      const healthMap: Record<string, SensorHealth> = {}
      await Promise.allSettled(
        s.map(async (sensor) => {
          const h = await api.sensorHealth(sensor.id, 1)
          if (h.length > 0) healthMap[sensor.id] = h[0]
        })
      )
      setLatestHealth(prev => ({ ...prev, ...healthMap }))
    } catch { /* backend not yet connected */ }
  }, [])

  const loadAlerts = useCallback(async () => {
    try {
      const active = await api.alerts('ACTIVE')
      const acked  = await api.alerts('ACKNOWLEDGED')
      setAlerts([...active, ...acked].sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      ))
    } catch { /* ignore */ }
  }, [])

  const loadPredictions = useCallback(async () => {
    try {
      const allMachines = await api.machines()
      if (allMachines.length === 0) return
      const preds = await api.predictions(allMachines[0].id, 20)
      setPredictions(preds)
    } catch { /* ignore */ }
  }, [])

  // Initial load + polling
  usePolling(loadSensors,     15_000)
  usePolling(loadAlerts,      10_000)
  usePolling(loadPredictions, 20_000)

  // ── WebSocket ──────────────────────────────────────────────────────────
  const handleWsMessage = useCallback((msg: WsMessage) => {
    setWsStatus('live')
    liveEventsRef.current += 1
    setLiveEvents(liveEventsRef.current)

    if (msg.type === 'health_update') {
      const d = msg.data as Record<string, unknown>
      const sid = d.sensor_id as string
      setLatestHealth(prev => ({
        ...prev,
        [sid]: {
          ...(prev[sid] ?? {}),
          sensor_id:           sid,
          timestamp:           d.timestamp as string,
          health_score:        d.health_score as number ?? prev[sid]?.health_score ?? 0,
          status:              d.status as string ?? 'UNKNOWN',
          anomaly_score:       d.anomaly_score as number ?? null,
          failure_probability: d.failure_probability as number ?? null,
        } as SensorHealth,
      }))
      // Update sensor status in sensor list
      setSensors(prev =>
        prev.map(s =>
          s.id === sid ? { ...s, status: d.status as Sensor['status'] ?? s.status } : s
        )
      )
    }

    if (msg.type === 'alert') {
      const a = msg.data as unknown as Alert
      setAlerts(prev => [a, ...prev.filter(x => x.id !== a.id)])
    }

    if (msg.type === 'prediction_update') {
      const p = msg.data as unknown as Prediction
      setPredictions(prev => [p, ...prev.slice(0, 19)])
    }
  }, [])

  useWebSocket(handleWsMessage)

  // Mark as offline after 3s without a message (WS reconnecting)
  useEffect(() => {
    const t = setTimeout(() => {
      if (wsStatus === 'connecting') setWsStatus('offline')
    }, 3000)
    return () => clearTimeout(t)
  }, [wsStatus])

  // ── Computed summary ───────────────────────────────────────────────────
  const summary: SystemSummary = {
    totalSensors:      sensors.length,
    healthySensors:    sensors.filter(s => (latestHealth[s.id]?.status ?? s.status) === 'HEALTHY').length,
    warningSensors:    sensors.filter(s => ['DRIFTING','NOISY','INTERMITTENT'].includes(latestHealth[s.id]?.status ?? s.status)).length,
    criticalSensors:   sensors.filter(s => ['STUCK','MISSING','FAILED'].includes(latestHealth[s.id]?.status ?? s.status)).length,
    activeAlerts:      alerts.filter(a => a.status === 'ACTIVE').length,
    systemReliability: sensors.length === 0 ? 0
      : Math.round(
          Object.values(latestHealth).reduce((s, h) => s + h.health_score, 0) /
          Math.max(Object.values(latestHealth).length, 1)
        ),
  }

  const selectedSensorObj = sensors.find(s => s.id === selectedSensor) ?? null

  // ── Alert action ───────────────────────────────────────────────────────
  const acknowledgeAlert = async (id: number) => {
    try {
      await api.acknowledgeAlert(id)
      setAlerts(prev => prev.map(a => a.id === id ? { ...a, status: 'ACKNOWLEDGED' } : a))
    } catch { /* ignore */ }
  }

  // ── Latest prediction ─────────────────────────────────────────────────
  const latestPred = predictions[0]

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>

      {/* Top bar */}
      <header style={{
        background: 'rgba(8,12,20,0.95)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid #1e2d45',
        padding: '0 28px',
        display: 'flex',
        alignItems: 'center',
        gap: '28px',
        height: '56px',
        flexShrink: 0,
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '30px', height: '30px',
            background: 'linear-gradient(135deg, #4da6ff, #b57bff)',
            borderRadius: '8px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 16px rgba(77,166,255,0.3)',
          }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L3 7v5c0 5.25 3.75 10.15 9 11.35C17.25 22.15 21 17.25 21 12V7L12 2z"
                fill="white" opacity="0.95" />
            </svg>
          </div>
          <span style={{ fontWeight: 800, fontSize: '15px', letterSpacing: '-0.4px', background: 'linear-gradient(135deg, #e2eaf6, #8bafd4)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            SensorShield
          </span>
        </div>

        {/* Nav */}
        <nav style={{ display: 'flex', gap: '4px' }}>
          {NAV.map(n => (
            <button
              key={n.id}
              onClick={() => { setTab(n.id); setSelected(null) }}
              style={{
                background:   tab === n.id ? 'rgba(77,166,255,0.1)' : 'transparent',
                border:       `1px solid ${tab === n.id ? 'rgba(77,166,255,0.3)' : 'transparent'}`,
                borderRadius: '8px',
                color:        tab === n.id ? '#4da6ff' : '#6b82a8',
                padding:      '5px 16px',
                fontSize:     '13px',
                fontWeight:   tab === n.id ? 700 : 500,
                cursor:       'pointer',
                transition:   'all 0.15s',
              }}
            >
              {n.label}
              {n.id === 'alerts' && summary.activeAlerts > 0 && (
                <span style={{
                  marginLeft: '7px',
                  background: 'linear-gradient(135deg, #ff4d6a, #ff2d50)',
                  color: '#fff',
                  borderRadius: '20px',
                  padding: '1px 7px',
                  fontSize: '10px',
                  fontWeight: 800,
                  boxShadow: '0 0 8px rgba(255,77,106,0.4)',
                }}>
                  {summary.activeAlerts}
                </span>
              )}
            </button>
          ))}
        </nav>

        {/* Status */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: '7px',
            background: wsStatus === 'live' ? 'rgba(34,211,160,0.08)' : 'rgba(107,130,168,0.08)',
            border: `1px solid ${wsStatus === 'live' ? 'rgba(34,211,160,0.2)' : 'rgba(107,130,168,0.15)'}`,
            borderRadius: '20px',
            padding: '4px 12px',
          }}>
            <span style={{
              width: '7px', height: '7px', borderRadius: '50%',
              background: wsStatus === 'live' ? '#22d3a0' : wsStatus === 'connecting' ? '#f0b429' : '#6b82a8',
              display: 'inline-block',
              boxShadow: wsStatus === 'live' ? '0 0 8px #22d3a0' : 'none',
              animation: wsStatus === 'live' ? 'pulse 2s ease-in-out infinite' : 'none',
            }} />
            <span style={{ fontSize: '12px', color: wsStatus === 'live' ? '#22d3a0' : '#6b82a8', fontWeight: 600 }}>
              {wsStatus === 'live' ? `Live · ${liveEvents}` : wsStatus === 'connecting' ? 'Connecting…' : 'Offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main style={{ flex: 1, overflow: 'auto', padding: '28px 32px' }}>

        {/* ── OVERVIEW ───────────────────────────────────────────────── */}
        {tab === 'overview' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', maxWidth: '1200px' }}>
            <div>
              <h2 style={{ fontSize: '20px', fontWeight: 800, marginBottom: '4px', letterSpacing: '-0.3px' }}>System Overview</h2>
              <div style={{ color: '#6b82a8', fontSize: '13px' }}>
                Real-time sensor reliability monitoring
              </div>
            </div>

            {/* Summary stats */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '14px' }}>
              <StatCard label="Total Sensors"    value={summary.totalSensors}    color="#4da6ff"  icon="📡" />
              <StatCard label="Healthy"          value={summary.healthySensors}  color="#22d3a0"  icon="✅" />
              <StatCard label="Warning"          value={summary.warningSensors}  color="#f0b429"  icon="⚠️" />
              <StatCard label="Critical"         value={summary.criticalSensors} color="#ff4d6a"  icon="🔴" />
              <StatCard label="Active Alerts"    value={summary.activeAlerts}    color="#ff4d6a"  icon="🔔" />
              <StatCard
                label="System Reliability"
                value={`${summary.systemReliability}%`}
                color={summary.systemReliability >= 70 ? '#22d3a0' : summary.systemReliability >= 50 ? '#f0b429' : '#ff4d6a'}
                sub="avg health score"
                icon="🛡️"
              />
            </div>

            {/* Latest prediction */}
            {latestPred && (
              <div style={{
                background: 'linear-gradient(135deg, #0d1520 0%, #121c2e 100%)',
                border: '1px solid #1e2d45',
                borderRadius: '16px',
                padding: '22px 26px',
                position: 'relative',
                overflow: 'hidden',
              }}>
                <div style={{
                  position: 'absolute', top: 0, left: 0, right: 0, height: '2px',
                  background: 'linear-gradient(90deg, transparent, #b57bff, #4da6ff, transparent)',
                  opacity: 0.6,
                }} />
                <div style={{ position: 'absolute', top: '-40px', right: '-40px', width: '150px', height: '150px', borderRadius: '50%', background: '#b57bff', opacity: 0.03, filter: 'blur(40px)' }} />

                <div style={{ fontSize: '11px', color: '#6b82a8', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: '16px', fontWeight: 600 }}>
                  🤖 Latest Failure Prediction
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '28px', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '11px', color: '#6b82a8', marginBottom: '4px' }}>Failure Probability</div>
                    <div style={{
                      fontSize: '36px', fontWeight: 800, letterSpacing: '-1px',
                      color: latestPred.failure_probability > 0.7 ? '#ff4d6a'
                           : latestPred.failure_probability > 0.4 ? '#f0b429' : '#22d3a0',
                    }}>
                      {(latestPred.failure_probability * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '11px', color: '#6b82a8', marginBottom: '4px' }}>Trust Status</div>
                    <div style={{ marginTop: '4px' }}>
                      <span style={{
                        color: latestPred.trust_status === 'TRUSTED' ? '#22d3a0'
                             : latestPred.trust_status === 'CAUTION' ? '#f0b429' : '#ff4d6a',
                        fontWeight: 800, fontSize: '16px', letterSpacing: '0.5px',
                      }}>
                        {latestPred.trust_status}
                      </span>
                    </div>
                  </div>
                  {latestPred.model_uncertainty != null && (
                    <div>
                      <div style={{ fontSize: '11px', color: '#6b82a8', marginBottom: '4px' }}>Model Uncertainty</div>
                      <div style={{ fontSize: '22px', fontWeight: 700, color: '#b57bff' }}>
                        ±{(latestPred.model_uncertainty * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}
                  <div style={{ marginLeft: 'auto', fontSize: '11px', color: '#3d5278' }}>
                    {new Date(latestPred.timestamp).toLocaleString()}
                  </div>
                </div>
                {latestPred.trust_reason && (
                  <div style={{ fontSize: '12px', color: '#6b82a8', marginTop: '12px', paddingTop: '12px', borderTop: '1px solid #1e2d45' }}>
                    {latestPred.trust_reason}
                  </div>
                )}
              </div>
            )}

            {/* Machine panels with sensors grouped */}
            {sensors.length > 0 && (
              <div>
                <div style={{ fontSize: '11px', color: '#6b82a8', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: '16px', fontWeight: 600 }}>
                  🏭 Machines & Sensors
                </div>
                {Object.entries(
                  sensors.reduce((acc, s) => {
                    if (!acc[s.machine_id]) acc[s.machine_id] = []
                    acc[s.machine_id].push(s)
                    return acc
                  }, {} as Record<string, Sensor[]>)
                ).map(([machineId, machineSensors]) => {
                  const machine = machines.find(m => m.id === machineId) ?? {
                    id: machineId, name: machineId, location: null, status: 'ONLINE'
                  }
                  return (
                    <MachinePanel
                      key={machineId}
                      machine={machine}
                      sensors={machineSensors}
                      latestHealth={latestHealth}
                      selectedSensor={selectedSensor}
                      onSelect={(id) => { setTab('sensors'); setSelected(id) }}
                    />
                  )
                })}
              </div>
            )}

            {sensors.length === 0 && (
              <div style={{
                background: 'linear-gradient(135deg, #0d1520 0%, #121c2e 100%)',
                border: '1px solid #1e2d45',
                borderRadius: '16px', padding: '60px', textAlign: 'center', color: '#6b82a8',
              }}>
                <div style={{ fontSize: '40px', marginBottom: '12px' }}>📡</div>
                <div style={{ fontSize: '16px', marginBottom: '8px', color: '#e2eaf6', fontWeight: 600 }}>No sensors registered</div>
                <div style={{ fontSize: '13px' }}>
                  Run <code style={{ background: '#1a2640', padding: '2px 8px', borderRadius: '6px', color: '#4da6ff' }}>
                    python scripts/seed_database.py
                  </code> to seed the database
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── SENSORS ────────────────────────────────────────────────── */}
        {tab === 'sensors' && (
          <div style={{ display: 'flex', gap: '20px', maxWidth: '1300px', height: 'calc(100vh - 112px)' }}>
            {/* Sensor list */}
            <div style={{ width: '290px', flexShrink: 0, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px', paddingRight: '4px' }}>
              {sensors.length === 0 ? (
                <div style={{ color: '#6b82a8', fontSize: '13px', padding: '16px' }}>No sensors yet.</div>
              ) : sensors.map(s => (
                <SensorCard
                  key={s.id}
                  sensor={s}
                  health={latestHealth[s.id]}
                  active={selectedSensor === s.id}
                  onClick={() => setSelected(s.id)}
                />
              ))}
            </div>

            {/* Sensor detail panel */}
            <div style={{
              flex: 1, overflowY: 'auto',
              background: 'linear-gradient(135deg, #0d1520 0%, #111926 100%)',
              border: '1px solid #1e2d45',
              borderRadius: '16px', padding: '28px',
            }}>
              {selectedSensorObj ? (
                <SensorDetail sensor={selectedSensorObj} />
              ) : (
                <div style={{ color: '#3d5278', textAlign: 'center', paddingTop: '80px', fontSize: '14px' }}>
                  <div style={{ fontSize: '40px', marginBottom: '12px' }}>👆</div>
                  Select a sensor to view details
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── ALERTS ─────────────────────────────────────────────────── */}
        {tab === 'alerts' && (
          <div style={{ maxWidth: '960px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <h2 style={{ fontSize: '20px', fontWeight: 800, letterSpacing: '-0.3px' }}>Alert Center</h2>
                <div style={{ color: '#6b82a8', fontSize: '13px', marginTop: '3px' }}>
                  {summary.activeAlerts} active alert{summary.activeAlerts !== 1 ? 's' : ''}
                </div>
              </div>
              <button
                onClick={loadAlerts}
                style={{
                  background:   'transparent',
                  border:       '1px solid #1e2d45',
                  borderRadius: '10px',
                  color:        '#6b82a8',
                  padding:      '8px 18px',
                  fontSize:     '13px',
                  fontWeight:   600,
                  cursor:       'pointer',
                  display:      'flex',
                  alignItems:   'center',
                  gap:          '6px',
                }}
              >
                🔄 Refresh
              </button>
            </div>

            {alerts.length === 0 ? (
              <div style={{
                background: 'linear-gradient(135deg, #0d1520 0%, #121c2e 100%)',
                border: '1px solid #1e2d45',
                borderRadius: '16px', padding: '60px', textAlign: 'center',
              }}>
                <div style={{ fontSize: '40px', marginBottom: '12px' }}>✅</div>
                <div style={{ color: '#22d3a0', fontSize: '16px', fontWeight: 600 }}>All systems nominal</div>
                <div style={{ color: '#6b82a8', fontSize: '13px', marginTop: '6px' }}>No active alerts — all sensors are healthy</div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {alerts.map(a => (
                  <AlertRow key={a.id} alert={a} onAck={acknowledgeAlert} />
                ))}
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  )
}
