"""
main.py
=======
FastAPI application factory for SensorShield.

Lifespan pattern (FastAPI 0.93+):
  - startup: configure logging, verify DB connection, log config summary
  - shutdown: close DB connection pool gracefully

All routers are registered here. The order matters for OpenAPI tag ordering.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import check_database_connection, engine
from app.core.logging import configure_logging, get_logger

# Import all models so SQLAlchemy metadata is populated before any route runs
import app.models  # noqa: F401

from app.api.routes import system, machines, sensors, alerts, predictions, faults, websocket
from app.services.sensor_pipeline import model_registry
from app.services.mqtt_subscriber import start_mqtt_subscriber, stop_mqtt_subscriber

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs on startup.
    Code after `yield` runs on shutdown.
    """
    # Startup
    configure_logging()
    logger.info("=" * 60)
    logger.info("SensorShield v%s starting up", settings.APP_VERSION)
    logger.info("Environment : %s", settings.APP_ENV)
    logger.info("Database    : %s", settings.DATABASE_URL.split("@")[-1])  # hide creds
    logger.info("=" * 60)

    db_ok = await check_database_connection()
    if db_ok:
        logger.info("Database connection: OK")
    else:
        logger.warning(
            "Database connection: FAILED — "
            "start PostgreSQL and check DATABASE_URL in .env"
        )

    # Load all ML models into memory once (non-blocking — failures are logged)
    logger.info("Loading ML models...")
    try:
        model_registry.load_all()
        logger.info("ML models loaded.")
    except Exception as exc:
        logger.warning("ML model loading encountered errors: %s", exc)

    # Start MQTT subscriber (non-blocking — skips gracefully if broker unreachable)
    import asyncio
    asyncio.create_task(start_mqtt_subscriber())

    yield

    # Shutdown
    logger.info("SensorShield shutting down — closing DB connection pool")
    stop_mqtt_subscriber()
    await engine.dispose()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="SensorShield",
        description=(
            "AI-Powered Sensor Reliability & Prediction Trust Platform.\n\n"
            "Detects sensor degradation, scores reliability, and determines "
            "whether downstream AI predictions can be trusted."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # CORS
    # FastAPI CORS middleware must be added BEFORE routers.
    # In production, replace with your actual frontend origin.
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(system.router)
    app.include_router(machines.router)
    app.include_router(sensors.router)
    app.include_router(alerts.router)
    app.include_router(predictions.router)
    app.include_router(faults.router)
    app.include_router(websocket.router)

    return app


app = create_app()
