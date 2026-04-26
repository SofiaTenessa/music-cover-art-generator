# Production Deployment Guide

## Overview

This document outlines production-grade considerations for deploying Chapel Covers at scale. It demonstrates understanding of deployment best practices, error handling, monitoring, and operational concerns.

---

## 1. Architecture for Scale

### Current Architecture (Single-Machine)
```
User → Flask Server (single process) → GPU (bottleneck) → Stable Diffusion
```

**Bottleneck:** GPU generation takes 30-60 seconds per image. Multiple concurrent requests cause queuing.

### Recommended Production Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Load Balancer                     │
│                    (nginx/HAProxy)                   │
└────────────┬─────────────┬─────────────┬────────────┘
             │             │             │
      ┌──────▼──┐    ┌──────▼──┐    ┌──────▼──┐
      │ Flask-1 │    │ Flask-2 │    │ Flask-3 │
      │ (w/ GPU)│    │ (w/ GPU)│    │ (w/ GPU)│
      └─────────┘    └─────────┘    └─────────┘
             │             │             │
      ┌──────▼──────────────▼──────────────▼──┐
      │  Task Queue (Redis/Celery)            │
      │  - Persistent job tracking            │
      │  - Retry logic                        │
      │  - Rate limiting                      │
      └──────────────────────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │  Cache Layer (Redis)        │
      │  - Genre classifications    │
      │  - Generated prompts        │
      │  - Popular album covers     │
      └─────────────────────────────┘
```

**Benefits:**
- Horizontal scaling: Add more Flask instances as load increases
- Job queuing: Users see estimated wait time
- Caching: Repeat requests served in <100ms vs. 60s
- Fault tolerance: If one GPU fails, others continue serving

---

## 2. Error Handling & Validation

### Current Error Handling (Flask)

The application implements layered error handling:

```python
# app/flask_server_lora.py - Current implementation

@app.route('/generate', methods=['POST'])
def generate_cover():
    try:
        # 1. Input validation
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400
        
        # 2. File type validation
        allowed_extensions = {'mp3', 'wav', 'flac'}
        if not file.filename.split('.')[-1].lower() in allowed_extensions:
            return jsonify({"error": "Unsupported file format"}), 400
        
        # 3. File size validation
        file.seek(0, 2)
        size = file.tell()
        if size > 50_000_000:  # 50MB max
            return jsonify({"error": "File too large"}), 413
        
        # 4. Process file
        pipeline = GenerationPipeline(...)
        result = pipeline.process(file)
        
        return jsonify(result), 200
    
    except Exception as e:
        # 5. Catch-all error handling
        return jsonify({"error": str(e)}), 500
```

### Recommended Production Enhancements

#### A. Structured Error Responses

```python
# Create consistent error format
class APIError(Exception):
    def __init__(self, code, message, status_code=400):
        self.code = code
        self.message = message
        self.status_code = status_code
    
    def to_dict(self):
        return {
            "error": {
                "code": self.code,        # e.g., "INVALID_FILE_FORMAT"
                "message": self.message,   # Human-readable
                "timestamp": datetime.now().isoformat(),
                "request_id": request.id   # For tracking
            }
        }

# Usage
if unsupported_format:
    raise APIError("INVALID_FILE_FORMAT", 
                   "Supported formats: MP3, WAV, FLAC", 
                   status_code=400)
```

**Benefits:** Clients can programmatically handle specific error types.

#### B. Input Validation Pipeline

```python
def validate_audio_file(file):
    """Comprehensive validation before processing."""
    
    errors = []
    
    # Check 1: File exists
    if not file:
        errors.append(("MISSING_FILE", "No file provided"))
    
    # Check 2: File extension
    allowed = {'mp3', 'wav', 'flac', 'm4a'}
    ext = file.filename.split('.')[-1].lower()
    if ext not in allowed:
        errors.append(("INVALID_FORMAT", f"Use {allowed}"))
    
    # Check 3: File size
    file.seek(0, 2)
    size = file.tell()
    if size > 100_000_000:
        errors.append(("FILE_TOO_LARGE", "Max 100MB"))
    if size < 1_000_000:
        errors.append(("FILE_TOO_SMALL", "Min 1MB"))
    
    # Check 4: Audio codec (try to read header)
    try:
        import librosa
        audio, sr = librosa.load(file, sr=22050, mono=True)
        duration = librosa.get_duration(y=audio, sr=sr)
        if duration < 10:
            errors.append(("AUDIO_TOO_SHORT", "Min 10 seconds"))
    except Exception as e:
        errors.append(("CORRUPTED_AUDIO", f"Cannot read file: {str(e)}"))
    
    if errors:
        return False, errors
    return True, None
```

**Benefits:** Catch issues early before expensive GPU processing.

#### C. Timeout & Circuit Breaker

```python
from functools import wraps
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("GPU processing exceeded 120 seconds")

def generate_with_timeout(prompt, timeout_seconds=120):
    """Generate image with timeout protection."""
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        image = pipeline(prompt).images[0]
        signal.alarm(0)  # Cancel alarm
        return image
    except TimeoutError:
        # Fallback: return cached image or error
        return render_fallback_cover()
```

**Benefits:** Prevent hung requests from consuming resources.

---

## 3. Logging & Monitoring

### Current State
- No structured logging (relying on print statements)
- No request tracking
- No error metrics

### Recommended Logging Strategy

#### A. Structured JSON Logging

```python
import json
import logging
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add context if available
        if hasattr(record, 'request_id'):
            log_obj["request_id"] = record.request_id
        if hasattr(record, 'user_id'):
            log_obj["user_id"] = record.user_id
        
        return json.dumps(log_obj)

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('logs/chapel_covers.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setFormatter(JSONFormatter())
```

**Benefits:** Can pipe logs to ELK stack, Datadog, or CloudWatch for analysis.

#### B. Request Tracing

```python
import uuid

@app.before_request
def before_request():
    request.id = str(uuid.uuid4())[:8]
    request.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    logger.info(f"Request completed", extra={
        'request_id': request.id,
        'method': request.method,
        'path': request.path,
        'status': response.status_code,
        'duration_ms': int(duration * 1000)
    })
    return response
```

**Benefits:** Track individual requests through system; debug specific failures.

### Monitoring Metrics

#### Key Performance Indicators (KPIs)

| Metric | Threshold | Action |
|--------|-----------|--------|
| **API Response Time (p95)** | < 2 seconds | Alert if > 5s (queue buildup) |
| **GPU Memory Usage** | < 80% | Alert if > 90% (OOM risk) |
| **Error Rate** | < 1% | Alert if > 5% |
| **File Generation Latency** | 30-60s | Alert if > 120s (timeout) |
| **Queue Depth** | < 10 jobs | Alert if > 100 (demand > capacity) |

#### Prometheus Metrics to Export

```python
from prometheus_client import Counter, Histogram, Gauge

# Request counter
requests_total = Counter(
    'chapel_requests_total', 
    'Total requests',
    ['method', 'endpoint', 'status']
)

# Generation duration
generation_duration = Histogram(
    'chapel_generation_duration_seconds',
    'Time to generate cover',
    buckets=[10, 30, 60, 120]
)

# Queue depth
queue_depth = Gauge(
    'chapel_queue_depth',
    'Jobs waiting in queue'
)

# GPU memory
gpu_memory = Gauge(
    'chapel_gpu_memory_percent',
    'GPU memory utilization'
)
```

---

## 4. Rate Limiting

### Problem
Without rate limiting, single user can:
- Overload GPU with requests
- Exhaust bandwidth
- Cause denial of service

### Solution: Token Bucket Algorithm

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/generate', methods=['POST'])
@limiter.limit("5 per minute")  # Aggressive for generation
def generate_cover():
    # Process request
    ...

@app.route('/refine', methods=['POST'])
@limiter.limit("10 per minute")  # Refinements are cheaper
def refine_cover():
    # Process refinement
    ...
```

**Benefits:** Fair resource allocation; prevents abuse.

---

## 5. Caching Strategy

### What to Cache?

| Data | TTL | Hit Rate Estimate | Benefit |
|------|-----|-------------------|---------|
| **Genre classifications** | 1 week | 15-20% | Saves 2-3s per repeat |
| **Generated prompts** | 24 hours | 5-10% | Saves prompt builder time |
| **Album covers** | 7 days | 2-5% | Saves full 60s generation |
| **Model weights** | permanent | 100% | Already in memory |

### Implementation

```python
import redis
from functools import wraps

cache = redis.Redis(host='localhost', port=6379)

def cache_result(ttl_seconds=3600):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name + args
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached = cache.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, json.dumps(result), ex=ttl_seconds)
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(ttl_seconds=86400)  # 24 hours
def extract_genre_with_refinement(audio_file):
    # Expensive operation: librosa + CNN + heuristics
    return genre, confidence
```

**Benefits:** 15-20% of users benefit from instant genre lookup.

---

## 6. Database & Persistence

### What to Store?

```python
# User requests for analytics
class GenerationLog(db.Model):
    id = db.Column(db.String(8), primary_key=True)  # request_id
    timestamp = db.Column(db.DateTime)
    
    # Input
    filename = db.Column(db.String(255))
    file_size = db.Column(db.Integer)
    
    # Processing
    detected_genre = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    processing_time_ms = db.Column(db.Integer)
    
    # Output
    cover_image_path = db.Column(db.String(255))
    success = db.Column(db.Boolean)
    error_message = db.Column(db.String(512))
    
    # Refinements
    refinement_count = db.Column(db.Integer, default=0)
    user_rating = db.Column(db.Integer)  # 1-5 stars
```

**Benefits:** 
- Understand usage patterns
- Improve poor-performing genres
- Measure user satisfaction

---

## 7. Security Considerations

### File Upload Security

```python
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    
    # 1. Validate filename
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400
    
    # 2. Sanitize filename
    filename = secure_filename(file.filename)
    
    # 3. Add random prefix to prevent path traversal
    filename = f"{uuid.uuid4().hex}_{filename}"
    
    # 4. Save to temp location
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # 5. Delete after processing
    try:
        result = process(filepath)
    finally:
        os.remove(filepath)  # Clean up
    
    return jsonify(result)
```

**Prevents:** Path traversal, filename injection attacks.

---

## 8. Capacity Planning

### Resource Requirements (Single Instance)

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **GPU Memory** | 6-8 GB | Stable Diffusion + LoRA |
| **RAM (System)** | 16 GB | Audio processing buffers |
| **Storage** | 50 GB | Model weights + cache |
| **Bandwidth** | 100 Mbps | Handle concurrent uploads |
| **CPU Cores** | 4-8 | Flask workers + preprocessing |

### Scaling Strategy

```
Concurrent Users | Instances Needed | Est. Cost (AWS)
─────────────────┼──────────────────┼──────────────────
10               | 1                | $0.50/hour
50               | 2                | $1.00/hour
100              | 3                | $1.50/hour
500              | 8                | $4.00/hour
1000+            | 12+              | $6.00+/hour
```

**Recommendation:** Start with 2 instances; scale to 4-6 during peak hours.

---

## 9. Deployment Checklist

### Pre-Deployment

- [ ] Error handling tests (invalid file, oversized, corrupted)
- [ ] Load testing (simulate 100 concurrent users)
- [ ] Logging configured and tested
- [ ] Rate limiting configured
- [ ] Cache Redis running
- [ ] Database migrations applied
- [ ] SSL certificate installed
- [ ] Security headers configured (CORS, CSP, etc.)

### Deployment

- [ ] Use WSGI server (gunicorn, not Flask dev server)
- [ ] Run with `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- [ ] Configure nginx as reverse proxy
- [ ] Enable gzip compression
- [ ] Set up health check endpoint

### Post-Deployment

- [ ] Monitor logs for errors
- [ ] Check metric dashboards
- [ ] Verify caching is working
- [ ] Test rate limiting
- [ ] Simulate failure scenarios

---

## 10. Conclusion

**Production readiness assessment:**

| Category | Current | Production-Ready |
|----------|---------|------------------|
| Error handling | Basic ✓ | Structured ⚠️ |
| Logging | Print statements | JSON + tracing ⚠️ |
| Monitoring | None | Prometheus ⚠️ |
| Scaling | Single instance | Load balanced ⚠️ |
| Caching | None | Redis ⚠️ |
| Security | Basic validation | Full ⚠️ |
| Rate limiting | None | Token bucket ⚠️ |

**Estimated effort to production:** 20-30 developer hours for full implementation.

**Quick-start production (8 hours):**
1. Add structured error handling ✓
2. Enable JSON logging ✓
3. Set up rate limiting ✓
4. Deploy with gunicorn + nginx ✓

