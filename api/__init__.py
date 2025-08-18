"""
스마트팩토리 에너지 관리 시스템 API 모듈
"""

from .routes import app, router
from .middleware import setup_middleware

__all__ = ['app', 'router', 'setup_middleware']