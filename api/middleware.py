# api/middleware.py
"""
API 미들웨어
"""
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import logging

logger = logging.getLogger("api_middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """요청/응답 로깅 미들웨어"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 요청 로깅
        logger.info(f"📥 {request.method} {request.url.path}")

        # 요청 처리
        response = await call_next(request)

        # 응답 시간 계산
        process_time = time.time() - start_time

        # 응답 로깅
        logger.info(f"📤 {response.status_code} - {process_time:.3f}초")

        # 응답 헤더에 처리 시간 추가
        response.headers["X-Process-Time"] = str(process_time)

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """요청 제한 미들웨어"""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()

        # 클라이언트별 요청 기록 정리
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < self.window_seconds
            ]
        else:
            self.requests[client_ip] = []

        # 요청 제한 확인
        if len(self.requests[client_ip]) >= self.max_requests:
            return Response(
                content="Too Many Requests",
                status_code=429
            )

        # 요청 기록
        self.requests[client_ip].append(current_time)

        return await call_next(request)


def setup_middleware(app):
    """미들웨어 설정"""
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

    logger.info("✅ API 미들웨어 설정 완료")