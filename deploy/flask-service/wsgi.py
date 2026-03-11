"""WSGI entry point for Gunicorn (multi-worker). Use: gunicorn -w 4 wsgi:app"""
from app import create_app
from app.config import DefaultConfig

app = create_app(DefaultConfig)
application = app  # alias for wsgi:application

