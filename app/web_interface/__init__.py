"""Web Interface Package."""

from .app import create_app, launch_app
from .routes import setup_routes

__all__ = ['create_app', 'launch_app', 'setup_routes']
