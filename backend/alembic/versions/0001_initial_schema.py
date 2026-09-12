"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-07

Creates all SensorShield tables:
  machines, sensors, sensor_readings, sensor_health,
  predictions, alerts, reconstructed_readings
"""

from alembic import op
import sqlalchemy as sa

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # machines
    # ------------------------------------------------------------------ #
    op.create_table(
        'machines',
        sa.Column('id',          sa.String(50),  primary_key=True),
        sa.Column('name',        sa.String(100), nullable=False),
        sa.Column('location',    sa.String(200), nullable=True),
        sa.Column('description', sa.Text(),      nullable=True),
        sa.Column('status',
                  sa.Enum('ONLINE','OFFLINE','DEGRADED','MAINTENANCE','UNKNOWN',
                          name='machinestatus'),
                  nullable=False, server_default='ONLINE'),
        sa.Column('created_at',  sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
        sa.Column('updated_at',  sa.DateTime(timezone=True),
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ------------------------------------------------------------------ #
    # sensors
    # ------------------------------------------------------------------ #
    op.create_table(
        'sensors',
        sa.Column('id',         sa.String(50),  primary_key=True),
        sa.Column('machine_id', sa.String(50),
                  sa.ForeignKey('machines.id', ondelete='CASCADE'), nullable=False),
        sa.Column('sensor_type',
                  sa.Enum('temperature','pressure','vibration','current',
                          'flow','humidity','voltage','speed','unknown',
                          name='sensortype'),
                  nullable=False),
        sa.Column('name',            sa.String(100), nullable=True),
        sa.Column('unit',            sa.String(20),  nullable=False),
        sa.Column('min_valid_value', sa.Float(),     nullable=True),
        sa.Column('max_valid_value', sa.Float(),     nullable=True),
        sa.Column('status',
                  sa.Enum('HEALTHY','DRIFTING','NOISY','STUCK',
                          'INTERMITTENT','MISSING','FAILED','UNKNOWN',
                          name='sensorstatus'),
                  nullable=False, server_default='HEALTHY'),
        sa.Column('installation_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ------------------------------------------------------------------ #
    # sensor_readings
    # ------------------------------------------------------------------ #
    op.create_table(
        'sensor_readings',
        sa.Column('id',           sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('sensor_id',    sa.String(50),
                  sa.ForeignKey('sensors.id', ondelete='CASCADE'), nullable=False),
        sa.Column('machine_id',   sa.String(50),  nullable=False),
        sa.Column('timestamp',    sa.DateTime(timezone=True), nullable=False),
        sa.Column('value',        sa.Float(),     nullable=False),
        sa.Column('is_valid',     sa.Boolean(),   nullable=False, server_default='true'),
        sa.Column('quality_flag', sa.String(50),  nullable=True),
    )
    op.create_index('ix_sensor_readings_sensor_ts',  'sensor_readings', ['sensor_id', 'timestamp'])
    op.create_index('ix_sensor_readings_machine_ts', 'sensor_readings', ['machine_id', 'timestamp'])

    # ------------------------------------------------------------------ #
    # sensor_health
    # ------------------------------------------------------------------ #
    op.create_table(
        'sensor_health',
        sa.Column('id',                   sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('sensor_id',            sa.String(50),
                  sa.ForeignKey('sensors.id', ondelete='CASCADE'), nullable=False),
        sa.Column('timestamp',            sa.DateTime(timezone=True), nullable=False),
        sa.Column('health_score',         sa.Float(), nullable=False),
        sa.Column('anomaly_score',        sa.Float(), nullable=True),
        sa.Column('drift_score',          sa.Float(), nullable=True),
        sa.Column('noise_score',          sa.Float(), nullable=True),
        sa.Column('missing_data_score',   sa.Float(), nullable=True),
        sa.Column('consistency_score',    sa.Float(), nullable=True),
        sa.Column('reconstruction_error', sa.Float(), nullable=True),
        sa.Column('status',
                  sa.Enum('HEALTHY','DRIFTING','NOISY','STUCK',
                          'INTERMITTENT','MISSING','FAILED','UNKNOWN',
                          name='sensorstatus'),
                  nullable=False),
        sa.Column('status_confidence', sa.Float(),  nullable=True),
        sa.Column('reason',            sa.Text(),   nullable=True),
    )
    op.create_index('ix_sensor_health_sensor_ts', 'sensor_health', ['sensor_id', 'timestamp'])

    # ------------------------------------------------------------------ #
    # predictions
    # ------------------------------------------------------------------ #
    op.create_table(
        'predictions',
        sa.Column('id',         sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('machine_id', sa.String(50),
                  sa.ForeignKey('machines.id', ondelete='CASCADE'), nullable=False),
        sa.Column('timestamp',                sa.DateTime(timezone=True), nullable=False),
        sa.Column('failure_probability',      sa.Float(), nullable=False),
        sa.Column('model_uncertainty',        sa.Float(), nullable=True),
        sa.Column('avg_sensor_reliability',   sa.Float(), nullable=True),
        sa.Column('min_sensor_reliability',   sa.Float(), nullable=True),
        sa.Column('trust_score',              sa.Float(), nullable=True),
        sa.Column('trust_status',
                  sa.Enum('TRUSTED','CAUTION','UNTRUSTED','UNKNOWN',
                          name='truststatus'),
                  nullable=False, server_default='UNKNOWN'),
        sa.Column('trust_reason', sa.Text(), nullable=True),
    )
    op.create_index('ix_predictions_machine_ts', 'predictions', ['machine_id', 'timestamp'])

    # ------------------------------------------------------------------ #
    # alerts
    # ------------------------------------------------------------------ #
    op.create_table(
        'alerts',
        sa.Column('id',         sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('sensor_id',  sa.String(50),
                  sa.ForeignKey('sensors.id', ondelete='CASCADE'), nullable=True),
        sa.Column('machine_id', sa.String(50), nullable=True),
        sa.Column('timestamp',  sa.DateTime(timezone=True), nullable=False),
        sa.Column('severity',
                  sa.Enum('INFO','WARNING','CRITICAL', name='alertseverity'),
                  nullable=False),
        sa.Column('fault_type',
                  sa.Enum('DRIFT','BIAS','NOISE','STUCK','MISSING','INTERMITTENT',
                          'SPIKE','SAMPLING_FAILURE','CROSS_SENSOR_INCONSISTENCY',
                          'LOW_RELIABILITY','LOW_PREDICTION_TRUST','UNKNOWN',
                          name='faulttype'),
                  nullable=False),
        sa.Column('status',
                  sa.Enum('ACTIVE','ACKNOWLEDGED','RESOLVED', name='alertstatus'),
                  nullable=False, server_default='ACTIVE'),
        sa.Column('description',           sa.Text(),   nullable=False),
        sa.Column('impact',                sa.Text(),   nullable=True),
        sa.Column('recommended_action',    sa.Text(),   nullable=True),
        sa.Column('health_score_at_alert', sa.Float(),  nullable=True),
        sa.Column('trust_score_at_alert',  sa.Float(),  nullable=True),
        sa.Column('resolved_at',           sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_alerts_sensor_ts',  'alerts', ['sensor_id', 'timestamp'])
    op.create_index('ix_alerts_machine_ts', 'alerts', ['machine_id', 'timestamp'])
    op.create_index('ix_alerts_status',     'alerts', ['status'])

    # ------------------------------------------------------------------ #
    # reconstructed_readings
    # ------------------------------------------------------------------ #
    op.create_table(
        'reconstructed_readings',
        sa.Column('id',                   sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('sensor_id',            sa.String(50),
                  sa.ForeignKey('sensors.id', ondelete='CASCADE'), nullable=False),
        sa.Column('timestamp',            sa.DateTime(timezone=True), nullable=False),
        sa.Column('observed_value',       sa.Float(), nullable=True),
        sa.Column('reconstructed_value',  sa.Float(), nullable=False),
        sa.Column('confidence',           sa.Float(), nullable=False),
        sa.Column('method',               sa.String(50), nullable=True),
    )
    op.create_index('ix_reconstructed_sensor_ts', 'reconstructed_readings', ['sensor_id', 'timestamp'])


def downgrade() -> None:
    op.drop_table('reconstructed_readings')
    op.drop_table('alerts')
    op.drop_table('predictions')
    op.drop_table('sensor_health')
    op.drop_table('sensor_readings')
    op.drop_table('sensors')
    op.drop_table('machines')

    # Drop custom enum types
    for enum_name in ['machinestatus', 'sensortype', 'sensorstatus',
                      'truststatus', 'alertseverity', 'faulttype', 'alertstatus']:
        op.execute(f'DROP TYPE IF EXISTS {enum_name}')
