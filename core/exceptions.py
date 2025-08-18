"""
스마트팩토리 에너지 관리 시스템 커스텀 예외 모듈
"""
from typing import Any, Dict, Optional, Union


class SmartFactoryException(Exception):
    """스마트팩토리 기본 예외 클래스"""

    def __init__(
            self,
            message: str,
            error_code: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
            cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause

    def __str__(self):
        if self.details:
            return f"{self.message} (상세: {self.details})"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """예외 정보를 딕셔너리로 변환"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'cause': str(self.cause) if self.cause else None
        }


# =============================================================================
# 데이터 관련 예외
# =============================================================================

class DataException(SmartFactoryException):
    """데이터 관련 기본 예외"""
    pass


class DataValidationError(DataException):
    """데이터 검증 오류"""
    pass


class DataProcessingError(DataException):
    """데이터 처리 오류"""
    pass


class DataNotFoundError(DataException):
    """데이터를 찾을 수 없음"""
    pass


class DataCorruptionError(DataException):
    """데이터 손상 오류"""
    pass


class InsufficientDataError(DataException):
    """데이터 부족 오류"""
    pass


class SensorDataError(DataException):
    """센서 데이터 오류"""
    pass


# =============================================================================
# 모델 관련 예외
# =============================================================================

class ModelException(SmartFactoryException):
    """모델 관련 기본 예외"""
    pass


class ModelNotFoundError(ModelException):
    """모델을 찾을 수 없음"""
    pass


class ModelLoadError(ModelException):
    """모델 로드 오류"""
    pass


class ModelSaveError(ModelException):
    """모델 저장 오류"""
    pass


class ModelTrainingError(ModelException):
    """모델 학습 오류"""
    pass


class ModelPredictionError(ModelException):
    """모델 예측 오류"""
    pass


class ModelValidationError(ModelException):
    """모델 검증 오류"""
    pass


class ModelPerformanceError(ModelException):
    """모델 성능 오류"""
    pass


# =============================================================================
# 최적화 관련 예외
# =============================================================================

class OptimizationException(SmartFactoryException):
    """최적화 관련 기본 예외"""
    pass


class SchedulingError(OptimizationException):
    """스케줄링 오류"""
    pass


class ConstraintViolationError(OptimizationException):
    """제약 조건 위반 오류"""
    pass


class InfeasibleSolutionError(OptimizationException):
    """실행 불가능한 해"""
    pass


class ConvergenceError(OptimizationException):
    """수렴 오류"""
    pass


class PeakPowerExceededError(OptimizationException):
    """피크 전력 초과 오류"""
    pass


class TOUPricingError(OptimizationException):
    """TOU 요금제 오류"""
    pass


# =============================================================================
# 시스템 관련 예외
# =============================================================================

class SystemException(SmartFactoryException):
    """시스템 관련 기본 예외"""
    pass


class ConfigurationError(SystemException):
    """설정 오류"""
    pass


class ConnectionError(SystemException):
    """연결 오류"""
    pass


class DatabaseError(SystemException):
    """데이터베이스 오류"""
    pass


class MQTTError(SystemException):
    """MQTT 오류"""
    pass


class APIError(SystemException):
    """API 오류"""
    pass


class AuthenticationError(SystemException):
    """인증 오류"""
    pass


class AuthorizationError(SystemException):
    """권한 오류"""
    pass


class ResourceNotAvailableError(SystemException):
    """리소스 사용 불가"""
    pass


class ServiceUnavailableError(SystemException):
    """서비스 사용 불가"""
    pass


# =============================================================================
# IoT 관련 예외
# =============================================================================

class IoTException(SmartFactoryException):
    """IoT 관련 기본 예외"""
    pass


class SensorConnectionError(IoTException):
    """센서 연결 오류"""
    pass


class SensorReadError(IoTException):
    """센서 읽기 오류"""
    pass


class SensorCalibrationError(IoTException):
    """센서 보정 오류"""
    pass


class DeviceOfflineError(IoTException):
    """기기 오프라인 오류"""
    pass


class CommunicationTimeoutError(IoTException):
    """통신 타임아웃 오류"""
    pass


# =============================================================================
# 실시간 처리 관련 예외
# =============================================================================

class RealTimeException(SmartFactoryException):
    """실시간 처리 관련 기본 예외"""
    pass


class StreamProcessingError(RealTimeException):
    """스트림 처리 오류"""
    pass


class BufferOverflowError(RealTimeException):
    """버퍼 오버플로우 오류"""
    pass


class LatencyExceededError(RealTimeException):
    """지연 시간 초과 오류"""
    pass


class QueueFullError(RealTimeException):
    """큐 가득참 오류"""
    pass


# =============================================================================
# 예외 핸들러 및 유틸리티
# =============================================================================

class ExceptionHandler:
    """예외 처리 핸들러"""

    def __init__(self, logger=None):
        from .logger import get_logger
        self.logger = logger or get_logger()

    def handle_exception(
            self,
            exception: Exception,
            context: Optional[str] = None,
            reraise: bool = True,
            default_return: Any = None
    ) -> Any:
        """예외 처리"""

        # 로깅
        if context:
            self.logger.error(f"{context}: {str(exception)}")
        else:
            self.logger.error(f"예외 발생: {str(exception)}")

        # 스택 트레이스 로깅 (SmartFactoryException이 아닌 경우)
        if not isinstance(exception, SmartFactoryException):
            self.logger.exception("상세 오류 정보:")

        # 재발생 여부에 따른 처리
        if reraise:
            raise exception
        else:
            return default_return

    def wrap_function(self, func, context: str = None, default_return: Any = None):
        """함수를 예외 처리로 래핑"""

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return self.handle_exception(
                    e,
                    context=context or f"{func.__name__}",
                    reraise=False,
                    default_return=default_return
                )

        return wrapper


def safe_execute(
        func,
        *args,
        context: str = None,
        default_return: Any = None,
        logger=None,
        **kwargs
) -> Any:
    """안전한 함수 실행"""
    handler = ExceptionHandler(logger)
    return handler.wrap_function(func, context, default_return)(*args, **kwargs)


def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
    """필수 필드 검증"""
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]

    if missing_fields:
        raise DataValidationError(
            f"필수 필드가 누락되었습니다",
            details={'missing_fields': missing_fields, 'provided_fields': list(data.keys())}
        )


def validate_data_types(data: Dict[str, Any], type_specs: Dict[str, type]) -> None:
    """데이터 타입 검증"""
    type_errors = []

    for field, expected_type in type_specs.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], expected_type):
                type_errors.append({
                    'field': field,
                    'expected_type': expected_type.__name__,
                    'actual_type': type(data[field]).__name__,
                    'value': str(data[field])
                })

    if type_errors:
        raise DataValidationError(
            f"데이터 타입이 올바르지 않습니다",
            details={'type_errors': type_errors}
        )


def validate_range(value: Union[int, float], min_val: float = None, max_val: float = None,
                   field_name: str = "value") -> None:
    """값 범위 검증"""
    if min_val is not None and value < min_val:
        raise DataValidationError(
            f"{field_name}이 최소값보다 작습니다",
            details={'field': field_name, 'value': value, 'min_allowed': min_val}
        )

    if max_val is not None and value > max_val:
        raise DataValidationError(
            f"{field_name}이 최대값보다 큽니다",
            details={'field': field_name, 'value': value, 'max_allowed': max_val}
        )


# =============================================================================
# 데코레이터
# =============================================================================

def exception_handler(
        exception_type: type = Exception,
        context: str = None,
        reraise: bool = True,
        default_return: Any = None,
        logger=None
):
    """예외 처리 데코레이터"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                handler = ExceptionHandler(logger)
                return handler.handle_exception(
                    e,
                    context=context or f"{func.__module__}.{func.__name__}",
                    reraise=reraise,
                    default_return=default_return
                )

        return wrapper

    return decorator


def validate_input(**validation_rules):
    """입력 검증 데코레이터"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # kwargs 검증
            if 'required' in validation_rules:
                validate_required_fields(kwargs, validation_rules['required'])

            if 'types' in validation_rules:
                validate_data_types(kwargs, validation_rules['types'])

            if 'ranges' in validation_rules:
                for field, (min_val, max_val) in validation_rules['ranges'].items():
                    if field in kwargs and kwargs[field] is not None:
                        validate_range(kwargs[field], min_val, max_val, field)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# 예외 생성 헬퍼 함수
# =============================================================================

def create_data_error(message: str, **details) -> DataException:
    """데이터 예외 생성 헬퍼"""
    return DataException(message, details=details)


def create_model_error(message: str, model_name: str = None, **details) -> ModelException:
    """모델 예외 생성 헬퍼"""
    if model_name:
        details['model_name'] = model_name
    return ModelException(message, details=details)


def create_optimization_error(message: str, **details) -> OptimizationException:
    """최적화 예외 생성 헬퍼"""
    return OptimizationException(message, details=details)


def create_system_error(message: str, component: str = None, **details) -> SystemException:
    """시스템 예외 생성 헬퍼"""
    if component:
        details['component'] = component
    return SystemException(message, details=details)


def create_iot_error(message: str, device_id: str = None, **details) -> IoTException:
    """IoT 예외 생성 헬퍼"""
    if device_id:
        details['device_id'] = device_id
    return IoTException(message, details=details)