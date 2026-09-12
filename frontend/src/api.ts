// Typed API client — thin wrappers around fetch
import type { Sensor, SensorHealth, Alert, Prediction, Machine, SensorReading } from './types'

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`)
  return r.json() as Promise<T>
}

async function patch<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { method: 'PATCH' })
  if (!r.ok) throw new Error(`PATCH ${path} → ${r.status}`)
  return r.json() as Promise<T>
}

export const api = {
  machines:   () => get<Machine[]>('/machines'),
  sensors:    (machineId?: string) =>
    get<Sensor[]>(machineId ? `/sensors?machine_id=${machineId}` : '/sensors'),

  sensorHistory: (sensorId: string, limit = 60) =>
    get<SensorReading[]>(`/sensors/${sensorId}/history?limit=${limit}`),

  sensorHealth: (sensorId: string, limit = 60) =>
    get<SensorHealth[]>(`/sensors/${sensorId}/health?limit=${limit}`),

  alerts: (status?: string) =>
    get<Alert[]>(`/alerts${status ? `?status=${status}` : ''}`),

  acknowledgeAlert: (alertId: number) =>
    patch<Alert>(`/alerts/${alertId}/acknowledge`),

  predictions: (machineId: string, limit = 20) =>
    get<Prediction[]>(`/predictions/${machineId}?limit=${limit}`),

  systemHealth: () => get<Record<string, unknown>>('/system/health'),
}
