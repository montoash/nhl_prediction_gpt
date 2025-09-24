# gunicorn_config.py
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
backlog = 2048

# Worker processes - optimized for free tier
workers = 1  # Reduced from 2 to save memory on free tier
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased from 30 to handle NFL data loading
keepalive = 2
max_requests = 100  # Reduced to prevent memory accumulation
max_requests_jitter = 10

# Memory optimization for free tier
preload_app = False  # Don't preload to save memory
worker_tmp_dir = "/dev/shm"  # Use shared memory for temp files

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'nfl_prediction_api'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None