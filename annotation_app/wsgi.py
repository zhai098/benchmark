from __future__ import annotations

from annotation_app.app import app, configure_app_logging, ensure_dirs

ensure_dirs()
configure_app_logging()

application = app
