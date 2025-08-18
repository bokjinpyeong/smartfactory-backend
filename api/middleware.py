"""
스마트팩토리 에너지 관리 시스템 - API 미들웨어

API 보안, 로깅, 모니터링 미들웨어
- 요청/응답 로깅
- 인증 및 권한 확인
- 요청 제한 (Rate Limiting)
- 성능 모니터링
- 오류 추적
"""

import time
import json
import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS, HTTP_401_UNAUTHORIZED
import jwt
from collections import defaultdict, deque

# Core 모듈
from core.config import get_config
from core.logger import get_logger, get_audit_logger, get_performance_logger
from core.exceptions import APIError, safe_execute


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """요청/응답 로깅 미들웨어"""

    def __init__(self, app, log_level: str = "INFO"):
        super().__init__(app)
        self.logger = get_logger("api.requests")
        self.audit_logger = get_audit_logger()
        self.log_level = log_level

        # 로깅 제외 경로
        self.exclude_paths = {"/health", "/docs", "/openapi.json"}

        # 민감한 필드 마스킹
        self.sensitive_fields = {"password", "token", "api_key", "secret"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """요청 처리"""
        start_time = time.time()
        request_id = self._generate_request_id()

        # 요청 정보 로깅
        if request.url.path not in self.exclude_paths:
            await self._log_request(request, request_id)

        # 요청 처리
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time

            # 응답 정보 로깅
            if request.url.path not in self.exclude_paths:
                await self._log_response(request, response, request_id, processing_time)

            # 성능 로깅
            if processing_time > 1.0:  # 1초 이상 걸린 요청
                self.logger.warning(
                    f"느린 요청 감지 - {request.method} {request.url.path}: {processing_time:.3f}초",
                    extra={
                        "request_id": request_id,
                        "processing_time": processing_time,
                        "method": request.method,
                        "path": request.url.path
                    }
                )

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            await self._log_error(request, e, request_id, processing_time)
            raise

    def _generate_request_id(self) -> str:
        """요청 ID 생성"""
        import uuid
        return str(uuid.uuid4())[:8]

    async def _log_request(self, request: Request, request_id: str):
        """요청 로깅"""
        # 요청 바디 읽기 (POST, PUT 등)
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    body = json.loads(body_bytes.decode())
                    body = self._mask_sensitive_data(body)
            except:
                body = "<binary_data>"

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "body": body,
            "timestamp": datetime.now().isoformat()
        }

        # 민감한 헤더 마스킹
        log_data["headers"] = self._mask_sensitive_data(log_data["headers"])

        self.logger.info(f"요청 수신 - {request.method} {request.url.path}", extra=log_data)

        # 감사 로그 (보안 관련 요청)
        if self._is_security_relevant(request):
            self.audit_logger.info("보안 관련 요청", extra=log_data)

    async def _log_response(self, request: Request, response: Response,
                            request_id: str, processing_time: float):
        """응답 로깅"""
        log_data = {
            "request_id": request_id,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "response_headers": dict(response.headers),
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(
            f"요청 완료 - {request.method} {request.url.path} [{response.status_code}] {processing_time:.3f}초",
            extra=log_data
        )

    async def _log_error(self, request: Request, error: Exception,
                         request_id: str, processing_time: float):
        """오류 로깅"""
        log_data = {
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.error(
            f"요청 오류 - {request.method} {request.url.path}: {error}",
            extra=log_data
        )

    def _mask_sensitive_data(self, data: Dict) -> Dict:
        """민감한 데이터 마스킹"""
        if not isinstance(data, dict):
            return data

        masked_data = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                masked_data[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked_data[key] = self._mask_sensitive_data(value)
            else:
                masked_data[key] = value

        return masked_data

    def _is_security_relevant(self, request: Request) -> bool:
        """보안 관련 요청 여부 확인"""
        security_paths = ["/auth", "/login", "/admin", "/system"]
        return any(path in request.url.path for path in security_paths)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """요청 제한 미들웨어"""

    def __init__(self, app,
                 requests_per_minute: int = 60,
                 requests_per_hour: int = 1000,
                 burst_limit: int = 10):
        super().__init__(app)
        self.logger = get_logger("api.rate_limit")

        # 제한 설정
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit

        # 요청 추적
        self.minute_requests = defaultdict(deque)  # IP별 1분간 요청
        self.hour_requests = defaultdict(deque)  # IP별 1시간간 요청

        # 제외 경로
        self.exclude_paths = {"/health", "/docs", "/openapi.json"}

        # 정리 작업
        self._cleanup_task = None
        self._start_cleanup_task()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """요청 처리"""
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # 요청 제한 확인
        if not self._is_allowed(client_ip, current_time):
            self.logger.warning(f"요청 제한 초과: {client_ip} - {request.url.path}")
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Too Many Requests",
                    "message": "요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )

        # 요청 기록
        self._record_request(client_ip, current_time)

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 주소 추출"""
        # X-Forwarded-For 헤더 확인 (프록시 환경)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # X-Real-IP 헤더 확인
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # 직접 연결
        return request.client.host if request.client else "unknown"

    def _is_allowed(self, client_ip: str, current_time: float) -> bool:
        """요청 허용 여부 확인"""
        minute_requests = self.minute_requests[client_ip]
        hour_requests = self.hour_requests[client_ip]

        # 오래된 요청 제거
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600

        while minute_requests and minute_requests[0] < minute_cutoff:
            minute_requests.popleft()

        while hour_requests and hour_requests[0] < hour_cutoff:
            hour_requests.popleft()

        # 제한 확인
        if len(minute_requests) >= self.requests_per_minute:
            return False

        if len(hour_requests) >= self.requests_per_hour:
            return False

        # 버스트 제한 (최근 10초)
        burst_cutoff = current_time - 10
        recent_requests = sum(1 for t in minute_requests if t > burst_cutoff)
        if recent_requests >= self.burst_limit:
            return False

        return True

    def _record_request(self, client_ip: str, current_time: float):
        """요청 기록"""
        self.minute_requests[client_ip].append(current_time)
        self.hour_requests[client_ip].append(current_time)

    def _start_cleanup_task(self):
        """정리 작업 시작"""

        async def cleanup():
            while True:
                await asyncio.sleep(300)  # 5분마다 정리
                current_time = time.time()

                # 오래된 데이터 정리
                cutoff_time = current_time - 3600  # 1시간 전

                for ip in list(self.minute_requests.keys()):
                    if not self.minute_requests[ip]:
                        del self.minute_requests[ip]

                for ip in list(self.hour_requests.keys()):
                    requests = self.hour_requests[ip]
                    while requests and requests[0] < cutoff_time:
                        requests.popleft()
                    if not requests:
                        del self.hour_requests[ip]

        self._cleanup_task = asyncio.create_task(cleanup())


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """인증 미들웨어"""

    def __init__(self, app, secret_key: str = None):
        super().__init__(app)
        self.logger = get_logger("api.auth")
        self.secret_key = secret_key or "your-secret-key"  # 실제로는 환경변수에서

        # 인증 제외 경로
        self.public_paths = {
            "/health", "/docs", "/openapi.json", "/redoc",
            "/api/v1/auth/login", "/api/v1/auth/register"
        }

        # 관리자만 접근 가능한 경로
        self.admin_paths = {
            "/api/v1/system", "/api/v1/admin"
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """요청 처리"""
        # 공개 경로는 인증 생략
        if any(request.url.path.startswith(path) for path in self.public_paths):
            return await call_next(request)

        # 인증 토큰 확인
        auth_result = await self._authenticate_request(request)
        if not auth_result["authenticated"]:
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Authentication Required",
                    "message": auth_result["message"]
                }
            )

        # 관리자 권한 확인
        if any(request.url.path.startswith(path) for path in self.admin_paths):
            if not auth_result.get("is_admin", False):
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Forbidden",
                        "message": "관리자 권한이 필요합니다"
                    }
                )

        # 사용자 정보를 요청에 추가
        request.state.user = auth_result["user"]

        return await call_next(request)

    async def _authenticate_request(self, request: Request) -> Dict:
        """요청 인증"""
        # Authorization 헤더 확인
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return {
                "authenticated": False,
                "message": "Authorization 헤더가 필요합니다"
            }

        # Bearer 토큰 추출
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                return {
                    "authenticated": False,
                    "message": "Bearer 토큰이 필요합니다"
                }
        except ValueError:
            return {
                "authenticated": False,
                "message": "잘못된 Authorization 헤더 형식"
            }

        # JWT 토큰 검증
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            # 토큰 만료 확인
            if payload.get("exp", 0) < time.time():
                return {
                    "authenticated": False,
                    "message": "토큰이 만료되었습니다"
                }

            return {
                "authenticated": True,
                "user": {
                    "user_id": payload.get("user_id"),
                    "username": payload.get("username"),
                    "is_admin": payload.get("is_admin", False)
                },
                "is_admin": payload.get("is_admin", False)
            }

        except jwt.InvalidTokenError as e:
            self.logger.warning(f"잘못된 JWT 토큰: {e}")
            return {
                "authenticated": False,
                "message": "잘못된 토큰입니다"
            }


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """성능 모니터링 미들웨어"""

    def __init__(self, app):
        super().__init__(app)
        self.performance_logger = get_performance_logger()

        # 성능 통계
        self.stats = {
            "request_count": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "slow_requests": 0,
            "error_count": 0
        }

        # 경로별 통계
        self.path_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "errors": 0
        })

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """요청 처리"""
        start_time = time.time()
        path = request.url.path

        try:
            response = await call_next(request)
            processing_time = time.time() - start_time

            # 통계 업데이트
            self._update_stats(path, processing_time, response.status_code >= 400)

            # 성능 로깅
            if processing_time > 2.0:  # 2초 이상
                self.performance_logger.warning(
                    f"매우 느린 요청: {request.method} {path} - {processing_time:.3f}초"
                )

            # 응답 헤더에 처리 시간 추가
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(path, processing_time, True)

            self.performance_logger.error(
                f"요청 처리 오류: {request.method} {path} - {processing_time:.3f}초 - {e}"
            )
            raise

    def _update_stats(self, path: str, processing_time: float, is_error: bool):
        """통계 업데이트"""
        # 전체 통계
        self.stats["request_count"] += 1
        self.stats["total_time"] += processing_time
        self.stats["average_time"] = self.stats["total_time"] / self.stats["request_count"]

        if processing_time > 1.0:
            self.stats["slow_requests"] += 1

        if is_error:
            self.stats["error_count"] += 1

        # 경로별 통계
        path_stat = self.path_stats[path]
        path_stat["count"] += 1
        path_stat["total_time"] += processing_time
        path_stat["average_time"] = path_stat["total_time"] / path_stat["count"]

        if is_error:
            path_stat["errors"] += 1

    def get_statistics(self) -> Dict:
        """성능 통계 반환"""
        return {
            "global_stats": self.stats.copy(),
            "path_stats": dict(self.path_stats),
            "timestamp": datetime.now().isoformat()
        }


class CORSMiddleware(BaseHTTPMiddleware):
    """CORS 미들웨어 (개선된 버전)"""

    def __init__(self, app,
                 allowed_origins: List[str] = None,
                 allowed_methods: List[str] = None,
                 allowed_headers: List[str] = None):
        super().__init__(app)

        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or ["*"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """요청 처리"""
        # Preflight 요청 처리
        if request.method == "OPTIONS":
            return self._create_preflight_response(request)

        # 실제 요청 처리
        response = await call_next(request)

        # CORS 헤더 추가
        self._add_cors_headers(response, request)

        return response

    def _create_preflight_response(self, request: Request) -> Response:
        """Preflight 응답 생성"""
        response = Response(status_code=200)
        self._add_cors_headers(response, request)
        return response

    def _add_cors_headers(self, response: Response, request: Request):
        """CORS 헤더 추가"""
        origin = request.headers.get("origin")

        if self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin or "*"

        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "86400"  # 24시간

    def _is_origin_allowed(self, origin: str) -> bool:
        """허용된 오리진인지 확인"""
        if not origin:
            return True

        if "*" in self.allowed_origins:
            return True

        return origin in self.allowed_origins


# 미들웨어 팩토리 함수들
def create_request_logging_middleware(log_level: str = "INFO"):
    """요청 로깅 미들웨어 생성"""
    return RequestLoggingMiddleware(None, log_level)


def create_rate_limiting_middleware(requests_per_minute: int = 60,
                                    requests_per_hour: int = 1000,
                                    burst_limit: int = 10):
    """요청 제한 미들웨어 생성"""
    return RateLimitingMiddleware(None, requests_per_minute, requests_per_hour, burst_limit)


def create_authentication_middleware(secret_key: str = None):
    """인증 미들웨어 생성"""
    return AuthenticationMiddleware(None, secret_key)


def create_performance_monitoring_middleware():
    """성능 모니터링 미들웨어 생성"""
    return PerformanceMonitoringMiddleware(None)


def create_cors_middleware(allowed_origins: List[str] = None):
    """CORS 미들웨어 생성"""
    return CORSMiddleware(None, allowed_origins)


# 사용 예시
if __name__ == "__main__":
    from fastapi import FastAPI

    app = FastAPI()

    # 미들웨어 추가 (순서 중요)
    app.add_middleware(PerformanceMonitoringMiddleware)
    app.add_middleware(AuthenticationMiddleware, secret_key="test-secret")
    app.add_middleware(RateLimitingMiddleware, requests_per_minute=30)
    app.add_middleware(RequestLoggingMiddleware, log_level="INFO")
    app.add_middleware(CORSMiddleware, allowed_origins=["http://localhost:3000"])


    @app.get("/test")
    async def test_endpoint():
        return {"message": "테스트 성공"}


    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)