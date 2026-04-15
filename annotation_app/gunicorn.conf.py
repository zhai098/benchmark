from __future__ import annotations

import os

bind = os.environ.get("ANNOTATION_APP_BIND", "127.0.0.1:5050")
workers = 2
threads = 4
timeout = 120
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
capture_output = True
