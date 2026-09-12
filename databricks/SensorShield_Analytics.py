# Databricks notebook source
# SensorShield Analytics - Apache Spark + MLflow + Delta Lake
# Processes 52,000 sensor readings, runs EDA, and saves Delta tables
# for Power BI consumption.
#
# Setup:
#   1. Upload data/raw/sensor_data.csv to Unity Catalog Volume
#   2. Run all cells in order

# COMMAND ----------

# Cell 1: Load sensor data from Unity Catalog Volume
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

df = spark.read.csv(
    "/Volumes/workspace/default/sensorshield/sensor_data.csv",
    header=True,
    inferSchema=True
)

print(f"Rows: {df.count():,}")
print(f"Columns: {len(df.columns)}")
df.printSchema()
display(df.limit(5))

# COMMAND ----------

# Cell 2: Rename columns to sensor IDs and save as Delta table
sensor_cols = ["timestamp", "sensor_00", "sensor_01", "sensor_02",
               "sensor_03", "sensor_04", "sensor_05"]

rename_map = {
    "sensor_00": "TEMP_01",
    "sensor_01": "PRESSURE_01",
    "sensor_02": "VIBRATION_01",
    "sensor_03": "CURRENT_01",
    "sensor_04": "FLOW_01",
    "sensor_05": "HUMIDITY_01"
}

df_clean = df.select(sensor_cols)
for old, new in rename_map.items():
    df_clean = df_clean.withColumnRenamed(old, new)

df_clean = df_clean.dropna()

print(f"Clean rows: {df_clean.count():,}")
display(df_clean.limit(5))

df_clean.write.format("delta").mode("overwrite").saveAsTable("sensorshield_readings")
print("Delta table created: sensorshield_readings")

# COMMAND ----------

# Cell 3: Register temp view and run SQL analytics
df_clean.createOrReplaceTempView("sensor_data")

summary = spark.sql("""
    SELECT
        ROUND(AVG(TEMP_01), 2)         AS avg_temperature,
        ROUND(AVG(PRESSURE_01), 2)     AS avg_pressure,
        ROUND(AVG(VIBRATION_01), 2)    AS avg_vibration,
        ROUND(AVG(CURRENT_01), 2)      AS avg_current,
        ROUND(AVG(FLOW_01), 2)         AS avg_flow,
        ROUND(AVG(HUMIDITY_01), 2)     AS avg_humidity,
        ROUND(STDDEV(TEMP_01), 2)      AS stddev_temperature,
        ROUND(STDDEV(VIBRATION_01), 2) AS stddev_vibration,
        COUNT(*)                        AS total_readings
    FROM sensor_data
""")
display(summary)

# COMMAND ----------

# Cell 4: Detect anomalies using statistical thresholds
anomalies = spark.sql("""
    SELECT
        timestamp,
        TEMP_01, PRESSURE_01, VIBRATION_01,
        CURRENT_01, FLOW_01, HUMIDITY_01,
        CASE
            WHEN TEMP_01 > 4.0        THEN 'HIGH_TEMP'
            WHEN VIBRATION_01 > 4.0   THEN 'HIGH_VIBRATION'
            WHEN PRESSURE_01 > 4.0    THEN 'HIGH_PRESSURE'
            WHEN CURRENT_01 > 4.0     THEN 'HIGH_CURRENT'
            WHEN FLOW_01 < 1.5        THEN 'LOW_FLOW'
            ELSE 'NORMAL'
        END AS fault_label
    FROM sensor_data
""")

fault_counts = anomalies.groupBy("fault_label").count().orderBy("count", ascending=False)
display(fault_counts)

anomalies.write.format("delta").mode("overwrite").saveAsTable("sensorshield_anomalies")
print("Delta table created: sensorshield_anomalies")

# COMMAND ----------

# Cell 5: MLflow experiment tracking
import mlflow

mlflow.set_experiment("/SensorShield_Analytics")

row = spark.sql("""
    SELECT
        AVG(TEMP_01)      as t,
        AVG(PRESSURE_01)  as p,
        AVG(VIBRATION_01) as v,
        AVG(CURRENT_01)   as c,
        AVG(FLOW_01)      as f,
        AVG(HUMIDITY_01)  as h,
        COUNT(*)          as n
    FROM sensor_data
""").first()

with mlflow.start_run(run_name="SensorShield_EDA"):
    mlflow.log_param("dataset",     "sensor_data.csv")
    mlflow.log_param("num_sensors", 6)
    mlflow.log_param("total_rows",  int(row["n"]))
    mlflow.log_metric("mean_temperature",  round(float(row["t"]), 4))
    mlflow.log_metric("mean_pressure",     round(float(row["p"]), 4))
    mlflow.log_metric("mean_vibration",    round(float(row["v"]), 4))
    mlflow.log_metric("mean_current",      round(float(row["c"]), 4))
    mlflow.log_metric("mean_flow",         round(float(row["f"]), 4))
    mlflow.log_metric("mean_humidity",     round(float(row["h"]), 4))
    print(f"MLflow run logged! Run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# Cell 6: Create sensor summary Delta table for Power BI
summary_df = spark.sql("""
    SELECT 'TEMP_01' AS sensor_id, 'temperature' AS sensor_type,
        ROUND(AVG(TEMP_01), 3) AS mean_value,
        ROUND(MIN(TEMP_01), 3) AS min_value,
        ROUND(MAX(TEMP_01), 3) AS max_value,
        ROUND(STDDEV(TEMP_01), 3) AS stddev_value,
        COUNT(TEMP_01) AS reading_count
    FROM sensor_data
    UNION ALL
    SELECT 'PRESSURE_01', 'pressure',
        ROUND(AVG(PRESSURE_01), 3), ROUND(MIN(PRESSURE_01), 3),
        ROUND(MAX(PRESSURE_01), 3), ROUND(STDDEV(PRESSURE_01), 3),
        COUNT(PRESSURE_01) FROM sensor_data
    UNION ALL
    SELECT 'VIBRATION_01', 'vibration',
        ROUND(AVG(VIBRATION_01), 3), ROUND(MIN(VIBRATION_01), 3),
        ROUND(MAX(VIBRATION_01), 3), ROUND(STDDEV(VIBRATION_01), 3),
        COUNT(VIBRATION_01) FROM sensor_data
    UNION ALL
    SELECT 'CURRENT_01', 'current',
        ROUND(AVG(CURRENT_01), 3), ROUND(MIN(CURRENT_01), 3),
        ROUND(MAX(CURRENT_01), 3), ROUND(STDDEV(CURRENT_01), 3),
        COUNT(CURRENT_01) FROM sensor_data
    UNION ALL
    SELECT 'FLOW_01', 'flow',
        ROUND(AVG(FLOW_01), 3), ROUND(MIN(FLOW_01), 3),
        ROUND(MAX(FLOW_01), 3), ROUND(STDDEV(FLOW_01), 3),
        COUNT(FLOW_01) FROM sensor_data
    UNION ALL
    SELECT 'HUMIDITY_01', 'humidity',
        ROUND(AVG(HUMIDITY_01), 3), ROUND(MIN(HUMIDITY_01), 3),
        ROUND(MAX(HUMIDITY_01), 3), ROUND(STDDEV(HUMIDITY_01), 3),
        COUNT(HUMIDITY_01) FROM sensor_data
""")

display(summary_df)
summary_df.write.format("delta").mode("overwrite").saveAsTable("sensorshield_sensor_summary")
print("Delta table created: sensorshield_sensor_summary")

# COMMAND ----------

# Cell 7: View all Delta tables
tables = spark.sql("SHOW TABLES").toPandas()
print(tables.to_string())
print("\nDatabricks integration complete!")
print("Delta tables ready for Power BI connection.")
