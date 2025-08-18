"""
스마트팩토리 에너지 관리 시스템 Models 패키지 (향상된 버전)

이 패키지는 AI 모델 관련 기능들을 제공합니다:
- 기본 모델 프레임워크 (base_model)
- 이상탐지 모델 (anomaly_detector, enhanced_anomaly_detector)
- 전력 예측 모델 (power_predictor)
- 통합 예측 시스템 (integrated_prediction_system)
- 고급 데이터 생성기 (advanced_data_generator)
"""

# 버전 정보
__version__ = "2.0.0"

# GPU 사용 가능 여부 확인
import torch
import warnings

warnings.filterwarnings('ignore')

GPU_AVAILABLE = torch.cuda.is_available()

# 기본 모델 프레임워크 임포트
from .base_model import (
    BaseModel,
    RegressionModel,
    ClassificationModel,
    EnsembleModel,
    ModelManager,
    get_model_manager,
    register_model,
    get_model
)

# 기본 이상탐지 모델 임포트
from .anomaly_detector import (
    AnomalyDetector,
    RealTimeAnomalyDetector,
    PowerAnomalyDetector,
    create_anomaly_detector,
    create_power_anomaly_detector,
    create_realtime_detector
)

# 향상된 이상탐지 모델 임포트
from .enhanced_anomaly_detector import (
    EnhancedAnomalyDetector,
    RealTimeEnhancedDetector,
    IntelligentAnomalyLabeler,
    create_enhanced_anomaly_detector,
    create_realtime_enhanced_detector,
    create_intelligent_labeler
)

# 전력 예측 모델 임포트
from .power_predictor import (
    LSTMPowerPredictor,
    XGBoostPowerPredictor,
    RandomForestPowerPredictor,
    MLPPowerPredictor,
    PowerPredictionEnsemble,
    create_lstm_predictor,
    create_xgboost_predictor,
    create_rf_predictor,
    create_mlp_predictor,
    create_ensemble_predictor
)

# 통합 예측 시스템 임포트
from .integrated_prediction_system import (
    IntegratedPredictionSystem,
    BatchPredictionProcessor,
    create_integrated_system,
    create_batch_processor,
    run_complete_pipeline
)

# 고급 데이터 생성기 임포트
from .advanced_data_generator import (
    AdvancedDataGenerator,
    MachineType,
    OperationalMode,
    Shift,
    OperatorSkill,
    MachineProfile,
    EnvironmentalConditions,
    create_data_generator,
    generate_sample_data,
    generate_machine_data,
    create_electrical_features
)

# Core 패키지 의존성
try:
    from ..core import get_logger, get_config

    logger = get_logger("models_package")
    logger.info(f"Models 패키지 로드 완료 v{__version__} (GPU: {'사용가능' if GPU_AVAILABLE else '사용불가'})")
except ImportError:
    print(f"Warning: Core 패키지를 먼저 초기화해주세요 (Models v{__version__})")

# 확장된 모델 타입 레지스트리
MODEL_REGISTRY = {
    # 기본 이상탐지 모델
    'anomaly_detector': AnomalyDetector,
    'power_anomaly_detector': PowerAnomalyDetector,
    'realtime_anomaly_detector': RealTimeAnomalyDetector,

    # 향상된 이상탐지 모델 (NEW)
    'enhanced_anomaly_detector': EnhancedAnomalyDetector,
    'realtime_enhanced_detector': RealTimeEnhancedDetector,
    'intelligent_labeler': IntelligentAnomalyLabeler,

    # 예측 모델
    'lstm_predictor': LSTMPowerPredictor,
    'xgboost_predictor': XGBoostPowerPredictor,
    'rf_predictor': RandomForestPowerPredictor,
    'mlp_predictor': MLPPowerPredictor,
    'ensemble_predictor': PowerPredictionEnsemble,

    # 통합 시스템 (NEW)
    'integrated_system': IntegratedPredictionSystem,
    'batch_processor': BatchPredictionProcessor,

    # 데이터 생성기 (NEW)
    'data_generator': AdvancedDataGenerator
}

# 향상된 기본 모델 설정
DEFAULT_MODEL_CONFIG = {
    'anomaly_detection': {
        'algorithm': 'isolation_forest',
        'contamination': 0.05,
        'n_estimators': 100,
        'max_samples': 1000,
        'enhanced_mode': True,  # NEW: 향상된 모드 사용
        'intelligent_labeling': True,  # NEW: 지능형 라벨링 사용
        'gpu_acceleration': GPU_AVAILABLE  # NEW: GPU 가속 사용
    },
    'power_prediction': {
        'ensemble_models': ['xgboost', 'rf', 'mlp'] + (['lstm'] if GPU_AVAILABLE else []),
        'ensemble_method': 'averaging',
        'enable_gpu': GPU_AVAILABLE,
        'safe_sampling': True,  # NEW: 안전한 샘플링 사용
        'optimization_level': 'high'  # NEW: 최적화 레벨
    },
    'integrated_system': {  # NEW: 통합 시스템 설정
        'enable_anomaly_detection': True,
        'enable_power_prediction': True,
        'risk_assessment': True,
        'real_time_mode': False,
        'batch_processing': True
    },
    'data_generation': {  # NEW: 데이터 생성 설정
        'physics_based': True,
        'include_3phase': True,
        'realistic_anomalies': True,
        'temporal_features': True,
        'environmental_factors': True
    },
    'training': {
        'validation_split': 0.2,
        'early_stopping': True,
        'cross_validation': False,
        'sample_for_speed': True,  # NEW: 대용량 데이터 샘플링
        'gpu_memory_management': True  # NEW: GPU 메모리 관리
    }
}


def create_model(model_type: str, model_name: str = None, **kwargs):
    """향상된 모델 팩토리 함수"""
    if model_type not in MODEL_REGISTRY:
        available_types = list(MODEL_REGISTRY.keys())
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}. "
                         f"사용 가능한 타입: {available_types}")

    model_class = MODEL_REGISTRY[model_type]

    # 기본 이름 설정
    if model_name is None:
        model_name = f"{model_type}_{len(get_model_manager().list_models()) + 1}"

    # 모델 생성 (타입별 특수 처리)
    if model_type in ['integrated_system']:
        model = model_class(model_name=model_name)
    elif model_type in ['batch_processor']:
        # BatchProcessor는 integrated_system이 필요
        integrated_system = kwargs.get('integrated_system')
        if integrated_system is None:
            raise ValueError("batch_processor 생성 시 integrated_system이 필요합니다")
        model = model_class(integrated_system)
    elif model_type in ['data_generator']:
        random_seed = kwargs.get('random_seed', 42)
        model = model_class(random_seed)
    elif model_type in ['ensemble_predictor']:
        device = kwargs.get('device')
        model = model_class(model_name=model_name, device=device)
    else:
        model = model_class(model_name=model_name)

    # 모델 매니저에 등록 (일부 유틸리티 클래스 제외)
    if model_type not in ['batch_processor', 'data_generator']:
        register_model(model)

    logger = get_logger("model_factory")
    logger.info(f"모델 생성 완료: {model_name} ({model_type})")

    return model


def create_enhanced_anomaly_system(config=None):
    """향상된 이상탐지 시스템 생성"""
    if config is None:
        config = DEFAULT_MODEL_CONFIG['anomaly_detection']

    systems = {}

    # 기본 이상탐지 모델 (호환성 유지)
    if not config.get('enhanced_mode', True):
        systems['basic_detector'] = create_model(
            'anomaly_detector',
            'basic_anomaly_detector'
        )

    # 향상된 이상탐지 모델 (권장)
    if config.get('enhanced_mode', True):
        systems['enhanced_detector'] = create_model(
            'enhanced_anomaly_detector',
            'enhanced_anomaly_detector'
        )

    # 전력 특화 이상탐지 모델
    systems['power_detector'] = create_model(
        'power_anomaly_detector',
        'power_anomaly_detector'
    )

    # 실시간 이상탐지 모델
    if config.get('enhanced_mode', True):
        systems['realtime_detector'] = create_realtime_enhanced_detector(
            contamination=config['contamination']
        )
    else:
        systems['realtime_detector'] = create_realtime_detector(
            contamination=config['contamination']
        )

    # 지능형 라벨러
    if config.get('intelligent_labeling', True):
        systems['intelligent_labeler'] = create_intelligent_labeler(
            contamination_rate=config['contamination']
        )

    return systems


def create_enhanced_prediction_system(config=None):
    """향상된 전력 예측 시스템 생성"""
    if config is None:
        config = DEFAULT_MODEL_CONFIG['power_prediction']

    # 개별 예측 모델들
    models = {}

    if 'xgboost' in config['ensemble_models']:
        models['xgboost'] = create_model('xgboost_predictor', 'xgboost_power_model')

    if 'rf' in config['ensemble_models']:
        models['rf'] = create_model('rf_predictor', 'rf_power_model')

    if 'mlp' in config['ensemble_models']:
        models['mlp'] = create_model('mlp_predictor', 'mlp_power_model')

    if 'lstm' in config['ensemble_models'] and GPU_AVAILABLE:
        models['lstm'] = create_model('lstm_predictor', 'lstm_power_model')

    # 앙상블 모델
    ensemble = create_model(
        'ensemble_predictor',
        'power_prediction_ensemble',
        device=torch.device('cuda' if config['enable_gpu'] and GPU_AVAILABLE else 'cpu')
    )

    return {
        'individual_models': models,
        'ensemble_model': ensemble
    }


def create_complete_integrated_system(config=None):
    """완전한 통합 AI 시스템 생성 (NEW)"""
    logger = get_logger("integrated_system")
    logger.info("완전한 통합 AI 시스템 생성 시작")

    if config is None:
        config = DEFAULT_MODEL_CONFIG

    # 통합 예측 시스템 (이상탐지 + 전력예측)
    integrated_system = create_model('integrated_system', 'main_integrated_system')

    # 배치 처리기
    batch_processor = create_model(
        'batch_processor',
        'main_batch_processor',
        integrated_system=integrated_system
    )

    # 데이터 생성기
    data_generator = create_model('data_generator', 'main_data_generator')

    # 개별 시스템들 (옵션)
    individual_systems = {}

    if config.get('anomaly_detection', {}).get('create_individual', False):
        individual_systems['anomaly_system'] = create_enhanced_anomaly_system(
            config.get('anomaly_detection')
        )

    if config.get('power_prediction', {}).get('create_individual', False):
        individual_systems['prediction_system'] = create_enhanced_prediction_system(
            config.get('power_prediction')
        )

    complete_system = {
        'integrated_system': integrated_system,
        'batch_processor': batch_processor,
        'data_generator': data_generator,
        'individual_systems': individual_systems,
        'model_manager': get_model_manager(),
        'system_info': {
            'gpu_available': GPU_AVAILABLE,
            'total_models': len(get_model_manager().list_models()),
            'version': __version__,
            'enhanced_features': True,
            'physics_based_generation': True,
            'intelligent_anomaly_detection': True
        }
    }

    logger.info("완전한 통합 AI 시스템 생성 완료")
    logger.info(f"총 {complete_system['system_info']['total_models']}개 모델 등록")

    return complete_system


def create_anomaly_system(config=None):
    """기존 이상탐지 시스템 생성 (하위 호환성)"""
    return create_enhanced_anomaly_system(config)


def create_prediction_system(config=None):
    """기존 예측 시스템 생성 (하위 호환성)"""
    return create_enhanced_prediction_system(config)


def create_complete_ai_system():
    """기존 AI 시스템 생성 (하위 호환성)"""
    return create_complete_integrated_system()


def train_all_models(X_train, y_train, X_val=None, y_val=None, **kwargs):
    """등록된 모든 모델 일괄 학습 (향상된 버전)"""
    logger = get_logger("model_training")
    manager = get_model_manager()

    results = {}

    # 안전한 샘플링 옵션
    sample_for_speed = kwargs.get('sample_for_speed', True)

    for model_name in manager.list_models():
        try:
            model = manager.get_model(model_name)

            # 모델 타입별 학습 방식
            if isinstance(model, (AnomalyDetector, PowerAnomalyDetector, EnhancedAnomalyDetector)):
                # 이상탐지 모델은 y_train 없이 학습
                result = model.train(X_train, **kwargs)
            elif isinstance(model, IntegratedPredictionSystem):
                # 통합 시스템은 특별한 파라미터 사용
                result = model.train(X_train, y_train, X_val, y_val,
                                   sample_for_speed=sample_for_speed, **kwargs)
            else:
                # 일반 예측 모델
                result = model.train(X_train, y_train, X_val, y_val, **kwargs)

            results[model_name] = {
                'success': True,
                'training_time': result.get('training_time', 0),
                'model_type': model.model_type,
                'enhanced_features': getattr(model, 'use_gpu', False)
            }

            logger.info(f"모델 학습 완료: {model_name}")

        except Exception as e:
            logger.error(f"모델 학습 실패: {model_name} - {str(e)}")
            results[model_name] = {
                'success': False,
                'error': str(e),
                'model_type': getattr(model, 'model_type', 'unknown')
            }

    # 결과 요약
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    enhanced_count = sum(1 for r in results.values() if r.get('enhanced_features', False))

    logger.info(f"일괄 학습 완료: {successful}/{total} 성공 (향상된 모델: {enhanced_count}개)")

    return results


def evaluate_all_models(X_test, y_test, **kwargs):
    """등록된 모든 예측 모델 평가 (향상된 버전)"""
    logger = get_logger("model_evaluation")
    manager = get_model_manager()

    # 예측 모델만 평가
    prediction_models = []
    integrated_models = []

    for model_name in manager.list_models():
        model = manager.get_model(model_name)
        if isinstance(model, IntegratedPredictionSystem) and model.is_trained:
            integrated_models.append(model_name)
        elif isinstance(model, (RegressionModel, EnsembleModel)) and model.is_trained:
            prediction_models.append(model_name)

    results = {}

    # 일반 예측 모델 평가
    if prediction_models:
        comparison_df = manager.compare_models(X_test, y_test, prediction_models)
        results['prediction_models'] = comparison_df.to_dict('records') if not comparison_df.empty else {}

    # 통합 모델 평가
    for model_name in integrated_models:
        try:
            model = manager.get_model(model_name)

            # 이상치 라벨 생성 (평가용)
            y_anomaly_test = kwargs.get('y_anomaly_test')
            if y_anomaly_test is None:
                labeler = create_intelligent_labeler()
                y_anomaly_test = labeler.create_intelligent_labels(X_test, y_test)

            evaluation_result = model.evaluate_integrated(X_test, y_test, y_anomaly_test)
            results[f'integrated_{model_name}'] = evaluation_result

        except Exception as e:
            logger.error(f"통합 모델 평가 실패: {model_name} - {str(e)}")

    logger.info(f"모델 평가 완료: 일반 {len(prediction_models)}개, 통합 {len(integrated_models)}개")

    return results


def get_enhanced_system_status():
    """향상된 AI 시스템 상태 조회"""
    manager = get_model_manager()
    models = manager.list_models()

    status = {
        'models_package_version': __version__,
        'gpu_available': GPU_AVAILABLE,
        'total_models': len(models),
        'trained_models': 0,
        'enhanced_models': 0,
        'integrated_systems': 0,
        'model_types': {},
        'model_details': {},
        'capabilities': {
            'intelligent_anomaly_detection': True,
            'physics_based_data_generation': True,
            'integrated_prediction': True,
            'batch_processing': True,
            'gpu_acceleration': GPU_AVAILABLE,
            'real_time_detection': True
        }
    }

    for model_name in models:
        try:
            model = manager.get_model(model_name)
            model_info = model.get_model_info()

            status['model_details'][model_name] = model_info

            if model.is_trained:
                status['trained_models'] += 1

            # 향상된 모델 카운트
            if isinstance(model, (EnhancedAnomalyDetector, IntegratedPredictionSystem)):
                status['enhanced_models'] += 1

            if isinstance(model, IntegratedPredictionSystem):
                status['integrated_systems'] += 1

            model_type = model.model_type
            if model_type not in status['model_types']:
                status['model_types'][model_type] = 0
            status['model_types'][model_type] += 1

        except Exception as e:
            logger = get_logger("system_status")
            logger.warning(f"모델 상태 조회 실패: {model_name} - {str(e)}")

    return status


def get_system_status():
    """시스템 상태 조회 (하위 호환성)"""
    return get_enhanced_system_status()


# 데이터 생성 편의 함수들
def quick_generate_data(
    n_samples: int = 10000,
    anomaly_rate: float = 0.05,
    machine_type: str = None
) -> pd.DataFrame:
    """빠른 데이터 생성"""
    if machine_type:
        return generate_machine_data(machine_type, n_samples, anomaly_rate)
    else:
        return generate_sample_data(n_samples, anomaly_rate)


def create_realistic_dataset(
    n_samples: int = 50000,
    anomaly_rate: float = 0.03,
    include_3phase: bool = True,
    environmental_variation: bool = True
) -> pd.DataFrame:
    """현실적인 데이터셋 생성"""
    generator = create_data_generator()

    env_conditions = None
    if environmental_variation:
        env_conditions = EnvironmentalConditions(
            ambient_temp_range=(18, 35),
            humidity_range=(30, 85),
            pressure_range=(995, 1030),
            seasonal_factor=np.random.uniform(0.8, 1.2)
        )

    return generator.generate_enhanced_industrial_data(
        n_samples=n_samples,
        anomaly_rate=anomaly_rate,
        environmental_conditions=env_conditions,
        include_3phase=include_3phase
    )


# 설정 업데이트 함수 (향상된 버전)
def configure_models(**kwargs):
    """모델 설정 업데이트 (향상된 버전)"""
    global DEFAULT_MODEL_CONFIG

    for key, value in kwargs.items():
        if key in DEFAULT_MODEL_CONFIG:
            if isinstance(value, dict):
                DEFAULT_MODEL_CONFIG[key].update(value)
            else:
                DEFAULT_MODEL_CONFIG[key] = value

    logger = get_logger("model_config")
    logger.info(f"모델 설정 업데이트: {list(kwargs.keys())}")


# 전체 모듈 내보내기 리스트 (확장)
__all__ = [
    # 버전 정보
    '__version__',
    'GPU_AVAILABLE',

    # 기본 모델 클래스
    'BaseModel',
    'RegressionModel',
    'ClassificationModel',
    'EnsembleModel',

    # 기본 이상탐지 모델
    'AnomalyDetector',
    'RealTimeAnomalyDetector',
    'PowerAnomalyDetector',

    # 향상된 이상탐지 모델 (NEW)
    'EnhancedAnomalyDetector',
    'RealTimeEnhancedDetector',
    'IntelligentAnomalyLabeler',

    # 예측 모델
    'LSTMPowerPredictor',
    'XGBoostPowerPredictor',
    'RandomForestPowerPredictor',
    'MLPPowerPredictor',
    'PowerPredictionEnsemble',

    # 통합 시스템 (NEW)
    'IntegratedPredictionSystem',
    'BatchPredictionProcessor',

    # 데이터 생성기 (NEW)
    'AdvancedDataGenerator',
    'MachineType',
    'OperationalMode',
    'Shift',
    'OperatorSkill',
    'MachineProfile',
    'EnvironmentalConditions',

    # 모델 관리
    'ModelManager',
    'get_model_manager',
    'register_model',
    'get_model',

    # 팩토리 함수들
    'create_model',
    'create_anomaly_detector',
    'create_power_anomaly_detector',
    'create_realtime_detector',
    'create_enhanced_anomaly_detector',  # NEW
    'create_realtime_enhanced_detector',  # NEW
    'create_intelligent_labeler',  # NEW
    'create_lstm_predictor',
    'create_xgboost_predictor',
    'create_rf_predictor',
    'create_mlp_predictor',
    'create_ensemble_predictor',
    'create_integrated_system',  # NEW
    'create_batch_processor',  # NEW
    'create_data_generator',  # NEW

    # 시스템 레벨 함수들
    'create_anomaly_system',
    'create_prediction_system',
    'create_complete_ai_system',
    'create_enhanced_anomaly_system',  # NEW
    'create_enhanced_prediction_system',  # NEW
    'create_complete_integrated_system',  # NEW
    'train_all_models',
    'evaluate_all_models',
    'get_system_status',
    'get_enhanced_system_status',  # NEW
    'run_complete_pipeline',  # NEW

    # 데이터 생성 함수들 (NEW)
    'generate_sample_data',
    'generate_machine_data',
    'create_electrical_features',
    'quick_generate_data',
    'create_realistic_dataset',

    # 설정
    'MODEL_REGISTRY',
    'DEFAULT_MODEL_CONFIG',
    'configure_models'
]