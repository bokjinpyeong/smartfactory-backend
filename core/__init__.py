"""
스마트팩토리 에너지 관리 시스템 Core 패키지

이 패키지는 시스템의 핵심 기능들을 제공합니다:
- 설정 관리 (config)
- 로깅 시스템 (logger)
- 예외 처리 (exceptions)
"""

# 버전 정보
__version__ = "1.0.0"
__author__ = "SmartFactory Energy Team"

# 핵심 모듈 임포트
from .config import (
    ConfigManager,
    get_config,
    update_config,
    ModelConfig,
    DataConfig,
    SystemConfig,
    PowerConfig
)

from .logger import (
    SmartFactoryLogger,
    PerformanceLogger,
    AuditLogger,
    get_logger,
    get_performance_logger,
    get_audit_logger,
    setup_logging,
    timer,
    log_performance
)

from .exceptions import (
    # 기본 예외
    SmartFactoryException,

    # 데이터 관련
    DataException,
    DataValidationError,
    DataProcessingError,
    DataNotFoundError,
    InsufficientDataError,
    SensorDataError,

    # 모델 관련
    ModelException,
    ModelNotFoundError,
    ModelLoadError,
    ModelSaveError,
    ModelTrainingError,
    ModelPredictionError,

    # 최적화 관련
    OptimizationException,
    SchedulingError,
    ConstraintViolationError,
    PeakPowerExceededError,
    TOUPricingError,

    # 시스템 관련
    SystemException,
    ConfigurationError,
    DatabaseError,
    APIError,

    # IoT 관련
    IoTException,
    SensorConnectionError,
    DeviceOfflineError,

    # 유틸리티 함수
    safe_execute,
    validate_required_fields,
    validate_data_types,
    validate_range,
    exception_handler,
    validate_input
)


# 전역 초기화
def initialize_core_system(env: str = "development", log_level: str = "INFO"):
    """Core 시스템 초기화"""

    # 설정 시스템 초기화
    config = update_config(env=env)

    # 로깅 시스템 초기화
    logger = setup_logging(level=log_level)

    logger.info("🚀 스마트팩토리 에너지 관리 시스템 Core 초기화 완료")
    logger.info(f"환경: {env}, 로그 레벨: {log_level}")

    return config, logger


# 시스템 상태 확인
def get_system_status():
    """시스템 상태 정보 반환"""
    config = get_config()
    logger = get_logger()

    status = {
        'core_version': __version__,
        'config_env': config.env,
        'system_ready': True,
        'components': {
            'config_manager': True,
            'logging_system': True,
            'exception_handler': True
        }
    }

    logger.debug("시스템 상태 조회", **status)
    return status


# 편의 함수들
def quick_setup(env: str = "development"):
    """빠른 설정 (개발용)"""
    return initialize_core_system(env=env, log_level="DEBUG")


def production_setup():
    """운영 환경 설정"""
    return initialize_core_system(env="production", log_level="INFO")


# 패키지 레벨에서 사용할 주요 객체들
__all__ = [
    # 버전 정보
    '__version__',
    '__author__',

    # 초기화 함수
    'initialize_core_system',
    'get_system_status',
    'quick_setup',
    'production_setup',

    # 설정 관련
    'ConfigManager',
    'get_config',
    'update_config',
    'ModelConfig',
    'DataConfig',
    'SystemConfig',
    'PowerConfig',

    # 로깅 관련
    'SmartFactoryLogger',
    'PerformanceLogger',
    'AuditLogger',
    'get_logger',
    'get_performance_logger',
    'get_audit_logger',
    'setup_logging',
    'timer',
    'log_performance',

    # 예외 관련
    'SmartFactoryException',
    'DataException',
    'DataValidationError',
    'DataProcessingError',
    'ModelException',
    'ModelTrainingError',
    'ModelPredictionError',
    'OptimizationException',
    'SystemException',
    'IoTException',
    'safe_execute',
    'exception_handler',
    'validate_input'
]