"""Scraper test config — disable startup autostart/scheduler side effects."""

import os

# Must be set before the scraper app's lifespan runs (TestClient triggers it).
os.environ["HRF_SCRAPER_AUTOSTART"] = "false"
os.environ["HRF_SCHEDULER_ENABLED"] = "false"
