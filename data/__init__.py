"""
스마트팩토리 에너지 관리 시스템 Data 패키지

이 패키지는 데이터 처리 관련 기능들을 제공합니다:
- 데이터 수집 (collector)
- 데이터 전처리 (processor)
- 데이터 검증 (validator)
"""

# 버전 정보
__version__ = "1.0.0"

# 데이터 처리 모듈 임포트
from .processor import (
    DataProcessor,
    create_processor,
    quick_clean_data,
    quick_feature_engineering,
    quick_data_preparation
)

# Core 패키지 의존성
try:
    from ..core import get_logger, get_config

    logger = get_logger("data_package")
    logger.info("Data 패키지 로드 완료")
except ImportError:
    print("Warning: Core 패키지를 먼저 초기화해주세요")

# 데이터 처리 파이프라인 설정
DEFAULT_PIPELINE_CONFIG = {
    'cleaning': {
        'remove_negative_power': True,
        'temperature_range': (-50, 200),
        'remove_duplicates': True,
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5
    },
    'feature_engineering': {
        'create_time_features': True,
        'create_interaction_features': True,
        'create_derived_features': True,
        'encode_categorical': True
    },
    'scaling': {
        'method': 'adaptive',  # adaptive, standard, minmax, robust
        'exclude_columns': ['Machine_ID', 'Event_Sequence_Number']
    },
    'splitting': {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'method': 'stratified'  # stratified, temporal, random
    }
}


def create_data_pipeline(config=None):
    """데이터 처리 파이프라인 생성"""
    if config is None:
        config = DEFAULT_PIPELINE_CONFIG

    processor = create_processor()

    def pipeline(data, target_col='Power_Consumption_Realistic'):
        """전체 데이터 처리 파이프라인 실행"""
        logger = get_logger("data_pipeline")
        logger.info("데이터 처리 파이프라인 시작")

        try:
            # 1단계: 데이터 정리
            if config['cleaning']['remove_negative_power']:
                data_clean = processor.clean_data(data)
            else:
                data_clean = data.copy()

            # 2단계: 현실적 전력 모델 생성
            data_with_realistic = processor.create_realistic_power_model(data_clean)

            # 3단계: 피처 엔지니어링
            if config['feature_engineering']['create_time_features']:
                data_engineered = processor.engineer_features(data_with_realistic)
            else:
                data_engineered = data_with_realistic

            # 4단계: 스케일링
            data_scaled = processor.scale_features(
                data_engineered,
                fit=True
            )

            # 5단계: 데이터 분할
            split_result = processor.split_data(
                data_scaled,
                target_col,
                split_method=config['splitting']['method']
            )

            logger.info("데이터 처리 파이프라인 완료")
            return split_result

        except Exception as e:
            logger.error(f"데이터 처리 파이프라인 실패: {str(e)}")
            raise

    return pipeline


def validate_data_quality(data):
    """데이터 품질 검증"""
    processor = create_processor()
    return processor.validate_data(data)


def get_data_summary(data):
    """데이터 요약 정보 생성"""
    processor = create_processor()
    return processor.create_data_summary(data)


# 편의 함수들
def quick_pipeline(data, target_col='Power_Consumption_Realistic'):
    """빠른 데이터 처리 (기본 설정)"""
    pipeline = create_data_pipeline()
    return pipeline(data, target_col)


def custom_pipeline(data, target_col='Power_Consumption_Realistic', **kwargs):
    """커스텀 설정으로 데이터 처리"""
    config = DEFAULT_PIPELINE_CONFIG.copy()

    # 사용자 설정으로 업데이트
    for key, value in kwargs.items():
        if key in config:
            if isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

    pipeline = create_data_pipeline(config)
    return pipeline(data, target_col)


# 데이터 검증 체크리스트
DATA_QUALITY_CHECKLIST = {
    'completeness': {
        'check': 'missing_values',
        'threshold': 0.1,  # 10% 이하 결측치
        'description': '결측치 비율이 10% 이하여야 함'
    },
    'validity': {
        'check': 'value_ranges',
        'power_min': 0,
        'power_max': 10000,
        'temp_min': -50,
        'temp_max': 200,
        'description': '값이 유효한 범위 내에 있어야 함'
    },
    'consistency': {
        'check': 'duplicates',
        'threshold': 0.01,  # 1% 이하 중복
        'description': '중복 데이터가 1% 이하여야 함'
    },
    'accuracy': {
        'check': 'outliers',
        'method': 'iqr',
        'threshold': 0.05,  # 5% 이하 이상치
        'description': '이상치가 5% 이하여야 함'
    }
}


def run_quality_checks(data):
    """데이터 품질 체크리스트 실행"""
    logger = get_logger("data_quality")
    results = {}

    for check_name, check_config in DATA_QUALITY_CHECKLIST.items():
        try:
            if check_config['check'] == 'missing_values':
                missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
                passed = missing_ratio <= check_config['threshold']
                results[check_name] = {
                    'passed': passed,
                    'value': missing_ratio,
                    'threshold': check_config['threshold'],
                    'description': check_config['description']
                }

            elif check_config['check'] == 'duplicates':
                duplicate_ratio = data.duplicated().sum() / len(data)
                passed = duplicate_ratio <= check_config['threshold']
                results[check_name] = {
                    'passed': passed,
                    'value': duplicate_ratio,
                    'threshold': check_config['threshold'],
                    'description': check_config['description']
                }

            # 다른 체크들도 구현...

        except Exception as e:
            logger.warning(f"품질 체크 '{check_name}' 실패: {str(e)}")
            results[check_name] = {
                'passed': False,
                'error': str(e),
                'description': check_config['description']
            }

    # 전체 결과 요약
    total_checks = len(results)
    passed_checks = sum(1 for r in results.values() if r.get('passed', False))

    summary = {
        'overall_score': passed_checks / total_checks,
        'passed_checks': passed_checks,
        'total_checks': total_checks,
        'details': results
    }

    logger.info(f"데이터 품질 체크 완료: {passed_checks}/{total_checks} 통과")
    return summary


# 패키지 레벨 설정
def configure_data_processing(**kwargs):
    """데이터 처리 설정 업데이트"""
    global DEFAULT_PIPELINE_CONFIG

    for key, value in kwargs.items():
        if key in DEFAULT_PIPELINE_CONFIG:
            if isinstance(value, dict):
                DEFAULT_PIPELINE_CONFIG[key].update(value)
            else:
                DEFAULT_PIPELINE_CONFIG[key] = value


__all__ = [
    # 버전 정보
    '__version__',

    # 주요 클래스
    'DataProcessor',

    # 팩토리 함수
    'create_processor',
    'create_data_pipeline',

    # 빠른 처리 함수
    'quick_clean_data',
    'quick_feature_engineering',
    'quick_data_preparation',
    'quick_pipeline',
    'custom_pipeline',

    # 품질 관리
    'validate_data_quality',
    'get_data_summary',
    'run_quality_checks',

    # 설정
    'DEFAULT_PIPELINE_CONFIG',
    'DATA_QUALITY_CHECKLIST',
    'configure_data_processing'
]