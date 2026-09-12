// Shared TypeScript types matching the backend Pydantic schemas

export type SensorStatus =
  | 'HEALTHY' | 'DRIFTING' | 'NOISY' | 'STUCK'
  | 'INTERMITTENT' | 'MISSING' | 'FAILED' | 'UNKNOWN'

export type AlertSeverity = 'INFO' | 'WARNING' | 'CRITICAL'
export type AlertStatus   = 'ACTIVE' | 'ACKNOWLEDGED' | 'RESOLVED'
export type TrustStatus   = 'TRUSTED' | 'CAUTION' | 'UNTRUSTED' | 'UNKNOWN'

export interface Sensor {
  id: string
  machine_id: string
  sensor_type: string
  name: string | null
  unit: string
  status: SensorStatus
  min_valid_value: number | null
  max_valid_value: number | null
}

export interface SensorReading {
  id: number
  sensor_id: string
  machine_id: string
  timestamp: string
  value: number
  is_valid: boolean
  quality_flag: string | null
}

export interface SensorHealth {
  id: number
  sensor_id: string
  timestamp: string
  health_score: number
  anomaly_score: number | null
  drift_score: number | null
  noise_score: number | null
  missing_data_score: number | null
  consistency_score: number | null
  reconstruction_error: number | null
  status: SensorStatus
  status_confidence: number | null
  reason: string | null
}

export interface Alert {
  id: number
  sensor_id: string | null
  machine_id: string | null
  timestamp: string
  severity: AlertSeverity
  fault_type: string
  status: AlertStatus
  description: string
  impact: string | null
  recommended_action: string | null
  health_score_at_alert: number | null
  trust_score_at_alert: number | null
  resolved_at: string | null
}

export interface Prediction {
  id: number
  machine_id: string
  timestamp: string
  failure_probability: number
  model_uncertainty: number | null
  avg_sensor_reliability: number | null
  min_sensor_reliability: number | null
  trust_score: number | null
  trust_status: TrustStatus
  trust_reason: string | null
}

export interface Machine {
  id: string
  name: string
  location: string | null
  status: string
}

// WebSocket message envelope
export type WsMessageType = 'health_update' | 'alert' | 'prediction_update'
export interface WsMessage {
  type: WsMessageType
  data: Record<string, unknown>
}

// Dashboard summary computed on the frontend
export interface SystemSummary {
  totalSensors: number
  healthySensors: number
  warningSensors: number
  criticalSensors: number
  activeAlerts: number
  systemReliability: number   // 0–100
}
