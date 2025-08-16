# app/main.py

# Standard Library Imports
import asyncio
import logging
import random
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List

# Third-Party Imports
import psutil
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (Boolean, Column, create_engine, DateTime, Float, Integer,
                        String)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
# Imports for Prometheus
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge


# --- Standard Logger Configuration ---
# Create a formatter
log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Handler for console output (for `docker-compose logs`)
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# Handler for file output (for Promtail to scrape)
file_handler = RotatingFileHandler(
    "/var/log/app.log", maxBytes=10*1024*1024, backupCount=5
)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# --- <<< NEW: INTEGRATE UVICORN LOGS >>> ---
# Get the Uvicorn access logger and add our file handler to it.
# This will make uvicorn's access logs go to /var/log/app.log as well.
logging.getLogger("uvicorn.access").addHandler(file_handler)

# --- Database Setup ---
SQLITE_DATABASE_URL = "sqlite:///./chaos_backend.db"
engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Database Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class ChaosEvent(Base):
    __tablename__ = "chaos_events"
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration = Column(Float)
    impact_level = Column(String)

class APIMetrics(Base):
    __tablename__ = "api_metrics"
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, index=True)
    method = Column(String)
    response_time = Column(Float)
    status_code = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)


# --- Pydantic Models ---
class UserCreate(BaseModel):
    username: str
    email: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool

class ChaosEventResponse(BaseModel):
    id: int
    event_type: str
    description: str
    timestamp: datetime
    duration: float
    impact_level: str

class MetricsResponse(BaseModel):
    endpoint: str
    method: str
    response_time: float
    status_code: int
    timestamp: datetime


# --- Custom Prometheus Metrics ---
CHAOS_EVENTS_TOTAL = Counter(
    "chaos_events_total",
    "Total number of chaos events triggered",
    ["event_type"]
)
SYSTEM_CPU_USAGE_PERCENT = Gauge(
    "system_cpu_usage_percent",
    "Current system-wide CPU utilization as a percentage"
)
SYSTEM_MEMORY_USAGE_PERCENT = Gauge(
    "system_memory_usage_percent",
    "Current system-wide memory utilization as a percentage"
)


# --- Chaos Engineering Core ---
class ChaosMonkey:
    def __init__(self):
        self.chaos_enabled = True
        self.chaos_probability = 0.1
        self.active_chaos = {}

    def should_cause_chaos(self) -> bool:
        return self.chaos_enabled and random.random() < self.chaos_probability

    async def async_latency_chaos(self, min_delay: float = 1.0, max_delay: float = 5.0):
        if self.should_cause_chaos():
            delay = random.uniform(min_delay, max_delay)
            logging.warning(f"Chaos Monkey: Introducing {delay:.2f}s async latency")
            CHAOS_EVENTS_TOTAL.labels(event_type="latency").inc()
            await asyncio.sleep(delay)
            return delay
        return 0

    def memory_chaos(self):
        if self.should_cause_chaos():
            logging.warning("Chaos Monkey: Consuming memory")
            CHAOS_EVENTS_TOTAL.labels(event_type="memory").inc()
            memory_hog = bytearray(100 * 1024 * 1024)
            time.sleep(2)
            del memory_hog
            return True
        return False

    def cpu_chaos(self, duration: float = 2.0):
        if self.should_cause_chaos():
            logging.warning(f"Chaos Monkey: Creating CPU load for {duration}s")
            CHAOS_EVENTS_TOTAL.labels(event_type="cpu_load").inc()
            end_time = time.time() + duration
            while time.time() < end_time:
                pass
            return duration
        return 0

    def exception_chaos(self):
        if self.should_cause_chaos():
            CHAOS_EVENTS_TOTAL.labels(event_type="exception").inc()
            exceptions = [
                HTTPException(status_code=500, detail="Chaos Monkey: Random server error"),
                HTTPException(status_code=503, detail="Chaos Monkey: Service temporarily unavailable"),
                HTTPException(status_code=429, detail="Chaos Monkey: Rate limit exceeded"),
            ]
            raise random.choice(exceptions)

    def database_chaos(self):
        if self.should_cause_chaos():
            logging.warning("Chaos Monkey: Simulating database issues")
            CHAOS_EVENTS_TOTAL.labels(event_type="database_issue").inc()
            time.sleep(random.uniform(0.5, 2.0))
            if random.random() < 0.3:
                raise HTTPException(status_code=503, detail="Chaos Monkey: Database connection failed")

chaos_monkey = ChaosMonkey()


# --- System Utilities ---
class SystemMetrics:
    @staticmethod
    def get_system_stats():
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": datetime.utcnow()
        }

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- FastAPI Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Application startup sequence initiated...")
    
    # Start background tasks
    metrics_task = asyncio.create_task(update_system_metrics())
    chaos_task = asyncio.create_task(background_chaos_events())
    
    yield
    
    # Shutdown
    logging.info("Shutting down application. Cleaning up tasks.")
    metrics_task.cancel()
    chaos_task.cancel()


# --- Main App Creation ---
app = FastAPI(
    title="Chaos Engineering Backend API",
    description="A FastAPI backend with comprehensive chaos engineering features",
    version="1.0.0",
    lifespan=lifespan 
)

Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Background Tasks ---
async def update_system_metrics():
    """Periodically update the system metrics gauges."""
    while True:
        try:
            stats = SystemMetrics.get_system_stats()
            SYSTEM_CPU_USAGE_PERCENT.set(stats["cpu_percent"])
            SYSTEM_MEMORY_USAGE_PERCENT.set(stats["memory_percent"])
        except Exception as e:
            logging.error(f"Failed to update system metrics: {e}")
        await asyncio.sleep(5)

async def background_chaos_events():
    """Run periodic chaos events in the background"""
    while True:
        try:
            await asyncio.sleep(random.uniform(30, 120))

            if not chaos_monkey.chaos_enabled:
                continue

            event_types = ["cpu_spike", "memory_spike"]
            event_type = random.choice(event_types)

            db = SessionLocal()
            try:
                duration = 0
                if event_type == "cpu_spike":
                    duration = chaos_monkey.cpu_chaos(duration=3.0)
                    if duration > 0:
                        chaos_event = ChaosEvent(
                            event_type="background_cpu_spike",
                            description=f"Background CPU spike for {duration}s",
                            duration=duration, impact_level="medium"
                        )
                        db.add(chaos_event)
                        db.commit()
                elif event_type == "memory_spike":
                    if chaos_monkey.memory_chaos():
                        chaos_event = ChaosEvent(
                            event_type="background_memory_spike",
                            description="Background memory consumption",
                            duration=2.0, impact_level="low"
                        )
                        db.add(chaos_event)
                        db.commit()
            finally:
                db.close()
        except Exception as e:
            logging.error(f"Background chaos event failed: {e}")


# --- API Endpoints ---
@app.get("/")
async def root():
    await chaos_monkey.async_latency_chaos(0.1, 1.0)
    chaos_monkey.exception_chaos()
    return {"message": "Welcome to Chaos Engineering Backend API", "version": "1.0.0", "chaos_enabled": chaos_monkey.chaos_enabled}

@app.get("/health")
async def health_check():
    # original_prob = chaos_monkey.chaos_probability
    # chaos_monkey.chaos_probability = 0.05
    # try:
    #     chaos_monkey.exception_chaos()
    #     system_stats = SystemMetrics.get_system_stats()
    #     return {"status": "healthy", "timestamp": datetime.utcnow(), "system_stats": system_stats}
    # finally:
    #     chaos_monkey.chaos_probability = original_prob
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# --- User Management Endpoints ---
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user with chaos engineering."""
    await chaos_monkey.async_latency_chaos(0.2, 2.0)
    chaos_monkey.database_chaos()
    chaos_monkey.exception_chaos()
    chaos_monkey.memory_chaos()

    existing_user = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="User with that username or email already exists")

    db_user = User(username=user.username, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/", response_model=List[UserResponse])
async def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all users with chaos engineering."""
    await chaos_monkey.async_latency_chaos(0.1, 1.5)
    chaos_monkey.database_chaos()
    chaos_monkey.exception_chaos()

    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get a specific user by ID with chaos engineering."""
    await chaos_monkey.async_latency_chaos(0.1, 1.0)
    chaos_monkey.database_chaos()
    chaos_monkey.exception_chaos()

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.delete("/users/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete a user by ID with chaos engineering."""
    await chaos_monkey.async_latency_chaos(0.2, 2.0)
    chaos_monkey.database_chaos()
    chaos_monkey.exception_chaos()

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

# --- Load and Stress Testing Endpoints ---
@app.get("/load-test")
async def load_test_endpoint():
    """Endpoint specifically for load testing with simulated processing."""
    processing_type = random.choice(["light", "medium", "heavy"])

    if processing_type == "light":
        await asyncio.sleep(random.uniform(0.1, 0.5))
    elif processing_type == "medium":
        await asyncio.sleep(random.uniform(0.5, 1.5))
        chaos_monkey.cpu_chaos(duration=1.0)
    else:  # heavy
        await asyncio.sleep(random.uniform(1.0, 3.0))
        chaos_monkey.cpu_chaos(duration=2.0)
        chaos_monkey.memory_chaos()

    return {
        "processing_type": processing_type,
        "timestamp": datetime.utcnow(),
        "random_data": [random.randint(1, 1000) for _ in range(100)]
    }

@app.post("/stress/cpu/{duration}")
async def stress_cpu(duration: float):
    """Manually stress CPU for a specified duration."""
    if not 0 < duration <= 30:
        raise HTTPException(status_code=400, detail="Duration must be between 0 and 30 seconds.")

    start_time = time.time()
    count = 0
    while time.time() < start_time + duration:
        count += 1

    return {
        "message": f"CPU stress test completed",
        "duration_seconds": duration,
        "iterations": count,
        "timestamp": datetime.utcnow()
    }

@app.post("/stress/memory/{megabytes}")
async def stress_memory(megabytes: int):
    """Manually stress memory by allocating a specified amount in MB."""
    if not 0 < megabytes <= 500:
        raise HTTPException(status_code=400, detail="Memory allocation must be between 0 and 500MB.")

    logging.info(f"Allocating {megabytes}MB of memory for a stress test...")
    memory_hog = bytearray(megabytes * 1024 * 1024)
    await asyncio.sleep(5)  # Hold memory for 5 seconds
    del memory_hog
    logging.info("Memory released.")

    return {
        "message": "Memory stress test completed",
        "allocated_mb": megabytes,
        "duration_seconds": 5,
        "timestamp": datetime.utcnow()
    }

# --- Chaos Control Endpoints ---
@app.post("/chaos/enable")
async def enable_chaos():
    chaos_monkey.chaos_enabled = True
    logging.info("Chaos engineering has been ENABLED.")
    return {"message": "Chaos engineering enabled", "enabled": True}

@app.post("/chaos/disable")
async def disable_chaos():
    chaos_monkey.chaos_enabled = False
    logging.info("Chaos engineering has been DISABLED.")
    return {"message": "Chaos engineering disabled", "enabled": False}

@app.post("/chaos/probability/{probability}")
async def set_chaos_probability(probability: float):
    if not 0.0 <= probability <= 1.0:
        raise HTTPException(status_code=400, detail="Probability must be between 0.0 and 1.0")

    chaos_monkey.chaos_probability = probability
    logging.info(f"Chaos probability set to {probability}")
    return {"message": f"Chaos probability set to {probability}", "probability": probability}

@app.get("/chaos/status")
async def chaos_status():
    return {
        "enabled": chaos_monkey.chaos_enabled,
        "probability": chaos_monkey.chaos_probability,
        "active_chaos": chaos_monkey.active_chaos
    }

# --- Metrics and Monitoring Endpoints ---
@app.get("/metrics/api", response_model=List[MetricsResponse])
async def get_api_metrics(limit: int = 100, db: Session = Depends(get_db)):
    """Get API performance metrics."""
    metrics = db.query(APIMetrics).order_by(APIMetrics.timestamp.desc()).limit(limit).all()
    return metrics

@app.get("/metrics/chaos", response_model=List[ChaosEventResponse])
async def get_chaos_events(limit: int = 100, db: Session = Depends(get_db)):
    """Get recorded chaos engineering events."""
    events = db.query(ChaosEvent).order_by(ChaosEvent.timestamp.desc()).limit(limit).all()
    return events

@app.get("/metrics/system")
async def get_system_metrics():
    """Get current system metrics."""
    return SystemMetrics.get_system_stats()

# --- WebSocket Endpoint ---
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            stats = SystemMetrics.get_system_stats()
            data_to_send = {
                "type": "system_metrics",
                "data": {
                    "cpu_percent": stats["cpu_percent"],
                    "memory_percent": stats["memory_percent"],
                    "disk_percent": stats["disk_percent"],
                    "timestamp": stats["timestamp"].isoformat(),
                    "chaos_enabled": chaos_monkey.chaos_enabled,
                    "chaos_probability": chaos_monkey.chaos_probability
                }
            }
            await websocket.send_json(data_to_send)
            await asyncio.sleep(5)
    except Exception as e:
        logging.warning(f"WebSocket disconnected: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)