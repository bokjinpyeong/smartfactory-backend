"""
스마트팩토리 에너지 관리 시스템 로깅 모듈
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
import json


class ColorFormatter(logging.Formatter):
    """컬러 포맷터"""

    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[94m',  # 파랑
        'INFO': '\033[92m',  # 초록
        'WARNING': '\033[93m',  # 노랑
        'ERROR': '\033[91m',  # 빨강
        'CRITICAL': '\033[95m',  # 마젠타
        'RESET': '\033[0m'  # 리셋
    }

    def format(self, record):
        if hasattr(record, 'color') and record.color:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON 포맷터"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # 추가 필드가 있으면 포함
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # 예외 정보 포함
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class SmartFactoryLogger:
    """스마트팩토리 전용 로거"""

    def __init__(self, name: str = "smartfactory", level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # 핸들러 중복 방지
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """핸들러 설정"""
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 컬러 포맷터 (개발 환경용)
        console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_formatter = ColorFormatter(console_format)
        console_handler.setFormatter(console_formatter)

        # 파일 핸들러
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # 일반 로그 파일
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)

        file_format = "%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s"
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)

        # JSON 로그 파일 (구조화된 로그)
        json_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name}_structured.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JSONFormatter())

        # 에러 전용 파일
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name}_errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        # 핸들러 추가
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
        self.logger.addHandler(error_handler)

    def debug(self, message: str, **kwargs):
        """디버그 로그"""
        extra = {'color': True, 'extra_fields': kwargs} if kwargs else {'color': True}
        self.logger.debug(message, extra=extra)

    def info(self, message: str, **kwargs):
        """정보 로그"""
        extra = {'color': True, 'extra_fields': kwargs} if kwargs else {'color': True}
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs):
        """경고 로그"""
        extra = {'color': True, 'extra_fields': kwargs} if kwargs else {'color': True}
        self.logger.warning(message, extra=extra)

    def error(self, message: str, **kwargs):
        """에러 로그"""
        extra = {'color': True, 'extra_fields': kwargs} if kwargs else {'color': True}
        self.logger.error(message, extra=extra)

    def critical(self, message: str, **kwargs):
        """치명적 에러 로그"""
        extra = {'color': True, 'extra_fields': kwargs} if kwargs else {'color': True}
        self.logger.critical(message, extra=extra)

    def exception(self, message: str, **kwargs):
        """예외 로그 (스택 트레이스 포함)"""
        extra = {'color': True, 'extra_fields': kwargs} if kwargs else {'color': True}
        self.logger.exception(message, extra=extra)


class PerformanceLogger:
    """성능 측정 로거"""

    def __init__(self, logger: SmartFactoryLogger):
        self.logger = logger
        self.timers: Dict[str, datetime] = {}

    def start_timer(self, name: str):
        """타이머 시작"""
        self.timers[name] = datetime.now()
        self.logger.debug(f"타이머 시작: {name}")

    def end_timer(self, name: str) -> Optional[float]:
        """타이머 종료 및 경과 시간 반환"""
        if name not in self.timers:
            self.logger.warning(f"타이머를 찾을 수 없습니다: {name}")
            return None

        start_time = self.timers.pop(name)
        elapsed = (datetime.now() - start_time).total_seconds()

        self.logger.info(f"타이머 종료: {name}", elapsed_seconds=elapsed)
        return elapsed

    def log_performance(self, operation: str, **metrics):
        """성능 지표 로그"""
        self.logger.info(f"성능 측정: {operation}", **metrics)


class AuditLogger:
    """감사 로거"""

    def __init__(self, logger: SmartFactoryLogger):
        self.logger = logger

    def log_model_action(self, action: str, model_name: str, **details):
        """모델 관련 액션 로그"""
        self.logger.info(
            f"모델 액션: {action}",
            action_type="model",
            model_name=model_name,
            **details
        )

    def log_data_action(self, action: str, data_source: str, **details):
        """데이터 관련 액션 로그"""
        self.logger.info(
            f"데이터 액션: {action}",
            action_type="data",
            data_source=data_source,
            **details
        )

    def log_system_action(self, action: str, component: str, **details):
        """시스템 관련 액션 로그"""
        self.logger.info(
            f"시스템 액션: {action}",
            action_type="system",
            component=component,
            **details
        )

    def log_alert(self, alert_type: str, severity: str, message: str, **details):
        """알림 로그"""
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(
            f"알림: {alert_type} - {message}",
            alert_type=alert_type,
            severity=severity,
            **details
        )


# 전역 로거 인스턴스들
_loggers: Dict[str, SmartFactoryLogger] = {}
_performance_loggers: Dict[str, PerformanceLogger] = {}
_audit_loggers: Dict[str, AuditLogger] = {}


def get_logger(name: str = "smartfactory", level: str = "INFO") -> SmartFactoryLogger:
    """로거 인스턴스 반환 (싱글톤)"""
    if name not in _loggers:
        _loggers[name] = SmartFactoryLogger(name, level)
    return _loggers[name]


def get_performance_logger(name: str = "smartfactory") -> PerformanceLogger:
    """성능 로거 반환"""
    if name not in _performance_loggers:
        logger = get_logger(name)
        _performance_loggers[name] = PerformanceLogger(logger)
    return _performance_loggers[name]


def get_audit_logger(name: str = "smartfactory") -> AuditLogger:
    """감사 로거 반환"""
    if name not in _audit_loggers:
        logger = get_logger(name)
        _audit_loggers[name] = AuditLogger(logger)
    return _audit_loggers[name]


def setup_logging(level: str = "INFO", name: str = "smartfactory"):
    """로깅 시스템 초기화"""
    logger = get_logger(name, level)
    logger.info("스마트팩토리 에너지 관리 시스템 로깅 초기화 완료")
    return logger


# 컨텍스트 매니저로 성능 측정
class timer:
    """성능 측정 컨텍스트 매니저"""

    def __init__(self, name: str, logger_name: str = "smartfactory"):
        self.name = name
        self.perf_logger = get_performance_logger(logger_name)

    def __enter__(self):
        self.perf_logger.start_timer(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.perf_logger.end_timer(self.name)
        if exc_type:
            logger = get_logger()
            logger.error(f"타이머 '{self.name}' 실행 중 오류 발생",
                         elapsed_seconds=elapsed,
                         exception_type=str(exc_type))
        return False


# 데코레이터로 성능 측정
def log_performance(func_name: str = None, logger_name: str = "smartfactory"):
    """함수 성능 측정 데코레이터"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"
            with timer(name, logger_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator