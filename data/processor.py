"""
스마트팩토리 에너지 관리 시스템 - 데이터 전처리 모듈

확장 가능한 데이터 전처리 파이프라인
- 스트리밍/배치 하이브리드 처리
- 모듈화된 전처리 단계
- 장애 허용성 및 데이터 품질 보장
- 실시간 데이터 정제 및 변환
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Core 모듈
from core.config import get_config
from core.logger import get_logger, log_performance
from core.exceptions import (
    DataProcessingError, DataValidationError,
    InsufficientDataError, safe_execute
)


class ProcessingStage(Enum):
    """처리 단계 열거형"""
    RAW = "raw"
    CLEANED = "cleaned"
    TRANSFORMED = "transformed"
    FEATURED = "featured"
    SCALED = "scaled"
    READY = "ready"


@dataclass
class ProcessingResult:
    """처리 결과 클래스"""
    data: pd.DataFrame
    stage: ProcessingStage
    metadata: Dict[str, Any]
    processing_time: float
    error_log: List[str]

    def is_success(self) -> bool:
        """처리 성공 여부"""
        return len(self.error_log) == 0


class BaseProcessor(ABC):
    """기본 전처리기 클래스"""

    def __init__(self, processor_name: str):
        self.processor_name = processor_name
        self.logger = get_logger(f"processor.{processor_name}")
        self.config = get_config()
        self.is_fitted = False
        self.processing_stats = {
            'total_processed': 0,
            'success_count': 0,
            'error_count': 0,
            'average_processing_time': 0.0
        }

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseProcessor':
        """전처리기 학습"""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 변환"""
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """학습 후 변환"""
        return self.fit(data).transform(data)

    @log_performance
    def process(self, data: pd.DataFrame) -> ProcessingResult:
        """안전한 데이터 처리"""
        start_time = datetime.now()
        error_log = []

        try:
            # 입력 검증
            self._validate_input(data)

            # 변환 수행
            processed_data = self.transform(data)

            # 결과 검증
            self._validate_output(processed_data)

            self.processing_stats['success_count'] += 1

        except Exception as e:
            error_log.append(str(e))
            processed_data = data.copy()  # 원본 데이터 반환
            self.processing_stats['error_count'] += 1
            self.logger.error(f"처리 오류 [{self.processor_name}]: {e}")

        finally:
            # 처리 통계 업데이트
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats['total_processed'] += 1
            self.processing_stats['average_processing_time'] = (
                (self.processing_stats['average_processing_time'] *
                 (self.processing_stats['total_processed'] - 1) + processing_time) /
                self.processing_stats['total_processed']
            )

        return ProcessingResult(
            data=processed_data,
            stage=self._get_stage(),
            metadata=self._get_metadata(),
            processing_time=processing_time,
            error_log=error_log
        )

    def _validate_input(self, data: pd.DataFrame):
        """입력 데이터 검증"""
        if data is None or data.empty:
            raise DataValidationError("입력 데이터가 비어있음")

        required_columns = self._get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"필수 컬럼 누락: {missing_columns}")

    def _validate_output(self, data: pd.DataFrame):
        """출력 데이터 검증"""
        if data is None or data.empty:
            raise DataProcessingError("처리 결과가 비어있음")

    @abstractmethod
    def _get_required_columns(self) -> List[str]:
        """필수 컬럼 목록"""
        pass

    @abstractmethod
    def _get_stage(self) -> ProcessingStage:
        """현재 처리 단계"""
        pass

    def _get_metadata(self) -> Dict[str, Any]:
        """메타데이터 반환"""
        return {
            'processor_name': self.processor_name,
            'is_fitted': self.is_fitted,
            'stats': self.processing_stats.copy()
        }


class DataCleaner(BaseProcessor):
    """데이터 정제 처리기"""

    def __init__(self):
        super().__init__("data_cleaner")
        self.outlier_detector = None
        self.imputer = None
        self.outlier_threshold = 3.0

    def fit(self, data: pd.DataFrame) -> 'DataCleaner':
        """정제기 학습"""
        self.logger.info("데이터 정제기 학습 시작")

        # 이상치 감지기 학습
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            self.outlier_threshold = self.config.data.outlier_threshold or 3.0

        # 결측값 대치기 학습
        if data.isnull().sum().sum() > 0:
            self.imputer = KNNImputer(n_neighbors=5)
            self.imputer.fit(data[numeric_columns])

        self.is_fitted = True
        self.logger.info("데이터 정제기 학습 완료")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정제"""
        if not self.is_fitted:
            raise DataProcessingError("정제기가 학습되지 않음")

        cleaned_data = data.copy()

        # 1. 중복 제거
        initial_shape = cleaned_data.shape
        cleaned_data = cleaned_data.drop_duplicates()
        if cleaned_data.shape[0] != initial_shape[0]:
            self.logger.info(f"중복 제거: {initial_shape[0] - cleaned_data.shape[0]} 행")

        # 2. 이상치 처리
        cleaned_data = self._handle_outliers(cleaned_data)

        # 3. 결측값 처리
        cleaned_data = self._handle_missing_values(cleaned_data)

        # 4. 데이터 타입 정규화
        cleaned_data = self._normalize_dtypes(cleaned_data)

        return cleaned_data

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """이상치 처리"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in ['timestamp', 'machine_id', 'sensor_id']:
                continue

            # Z-score 기반 이상치 감지
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outlier_mask = z_scores > self.outlier_threshold

            if outlier_mask.sum() > 0:
                # 이상치를 중앙값으로 대체
                median_value = data[col].median()
                data.loc[data[col].index[outlier_mask], col] = median_value
                self.logger.debug(f"이상치 처리 [{col}]: {outlier_mask.sum()} 개")

        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """결측값 처리"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if self.imputer and len(numeric_columns) > 0:
            # 수치형 컬럼 결측값 대치
            data[numeric_columns] = self.imputer.transform(data[numeric_columns])

        # 범주형 컬럼 결측값 처리
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else 'unknown'
                data[col].fillna(mode_value, inplace=True)

        return data

    def _normalize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 정규화"""
        # 타임스탬프 컬럼 처리
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])

        # ID 컬럼을 문자열로 변환
        for col in ['machine_id', 'sensor_id']:
            if col in data.columns:
                data[col] = data[col].astype(str)

        # 수치형 컬럼 타입 최적화
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].dtype == 'float64':
                data[col] = pd.to_numeric(data[col], downcast='float')
            elif data[col].dtype == 'int64':
                data[col] = pd.to_numeric(data[col], downcast='integer')

        return data

    def _get_required_columns(self) -> List[str]:
        """필수 컬럼 목록"""
        return []  # 정제기는 모든 컬럼을 처리할 수 있음

    def _get_stage(self) -> ProcessingStage:
        """현재 처리 단계"""
        return ProcessingStage.CLEANED


class FeatureEngineer(BaseProcessor):
    """특성 공학 처리기"""

    def __init__(self):
        super().__init__("feature_engineer")
        self.feature_configs = {}

    def fit(self, data: pd.DataFrame) -> 'FeatureEngineer':
        """특성 공학기 학습"""
        self.logger.info("특성 공학기 학습 시작")

        # 시계열 특성 설정
        if 'timestamp' in data.columns:
            self.feature_configs['temporal_features'] = True

        # 전력 관련 특성 설정
        power_columns = [col for col in data.columns if 'power' in col.lower()]
        if power_columns:
            self.feature_configs['power_features'] = power_columns

        # 통계 특성 설정
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        self.feature_configs['statistical_features'] = list(numeric_columns)

        self.is_fitted = True
        self.logger.info("특성 공학기 학습 완료")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """특성 생성"""
        if not self.is_fitted:
            raise DataProcessingError("특성 공학기가 학습되지 않음")

        featured_data = data.copy()

        # 1. 시간 기반 특성
        if self.feature_configs.get('temporal_features'):
            featured_data = self._create_temporal_features(featured_data)

        # 2. 전력 기반 특성
        if self.feature_configs.get('power_features'):
            featured_data = self._create_power_features(featured_data)

        # 3. 통계 기반 특성
        if self.feature_configs.get('statistical_features'):
            featured_data = self._create_statistical_features(featured_data)

        # 4. 도메인 특화 특성
        featured_data = self._create_domain_features(featured_data)

        return featured_data

    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """시간 기반 특성 생성"""
        if 'timestamp' not in data.columns:
            return data

        timestamp_col = pd.to_datetime(data['timestamp'])

        # 기본 시간 특성
        data['hour'] = timestamp_col.dt.hour
        data['day_of_week'] = timestamp_col.dt.dayofweek
        data['month'] = timestamp_col.dt.month
        data['is_weekend'] = (timestamp_col.dt.dayofweek >= 5).astype(int)

        # 시간대 구분 (피크/오프피크)
        data['is_peak_hour'] = ((timestamp_col.dt.hour >= 9) &
                               (timestamp_col.dt.hour <= 17)).astype(int)

        # 작업 시간 구분
        data['work_shift'] = pd.cut(timestamp_col.dt.hour,
                                   bins=[0, 8, 16, 24],
                                   labels=['night', 'day', 'evening'],
                                   include_lowest=True)

        return data

    def _create_power_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """전력 기반 특성 생성"""
        power_columns = self.feature_configs.get('power_features', [])

        for col in power_columns:
            if col in data.columns:
                # 전력 효율성 지표
                if 'voltage' in data.columns and 'current' in data.columns:
                    data['power_factor'] = data[col] / (data['voltage'] * data['current'] + 1e-8)
                    data['apparent_power'] = data['voltage'] * data['current']

                # 전력 변화율
                data[f'{col}_change'] = data[col].pct_change().fillna(0)

                # 이동 평균
                data[f'{col}_ma_5'] = data[col].rolling(window=5, min_periods=1).mean()
                data[f'{col}_ma_15'] = data[col].rolling(window=15, min_periods=1).mean()

                # 전력 레벨 분류
                power_quantiles = data[col].quantile([0.25, 0.5, 0.75])
                data[f'{col}_level'] = pd.cut(data[col],
                                             bins=[-float('inf')] + power_quantiles.tolist() + [float('inf')],
                                             labels=['low', 'medium', 'high', 'very_high'])

        return data

    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """통계 기반 특성 생성"""
        numeric_columns = self.feature_configs.get('statistical_features', [])

        # 롤링 윈도우 통계
        window_sizes = [5, 10, 15]

        for col in numeric_columns:
            if col in data.columns and col not in ['hour', 'day_of_week', 'month']:
                for window in window_sizes:
                    # 롤링 통계
                    data[f'{col}_std_{window}'] = data[col].rolling(window=window, min_periods=1).std()
                    data[f'{col}_min_{window}'] = data[col].rolling(window=window, min_periods=1).min()
                    data[f'{col}_max_{window}'] = data[col].rolling(window=window, min_periods=1).max()

                    # 롤링 변동 계수
                    rolling_mean = data[col].rolling(window=window, min_periods=1).mean()
                    rolling_std = data[col].rolling(window=window, min_periods=1).std()
                    data[f'{col}_cv_{window}'] = rolling_std / (rolling_mean + 1e-8)

        return data

    def _create_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """도메인 특화 특성 생성"""
        # 머신별 정규화된 전력 사용량
        if 'power_consumption' in data.columns and 'machine_id' in data.columns:
            machine_power_mean = data.groupby('machine_id')['power_consumption'].transform('mean')
            data['normalized_power'] = data['power_consumption'] / (machine_power_mean + 1e-8)

        # 온도-전력 상관관계
        if all(col in data.columns for col in ['temperature', 'power_consumption']):
            data['temp_power_ratio'] = data['temperature'] / (data['power_consumption'] + 1e-8)

        # 가동률 추정
        if 'power_consumption' in data.columns:
            # 전력 소비가 최소값의 150% 이상이면 가동 중으로 간주
            min_power = data['power_consumption'].min()
            threshold = min_power * 1.5
            data['is_operating'] = (data['power_consumption'] > threshold).astype(int)

        return data

    def _get_required_columns(self) -> List[str]:
        """필수 컬럼 목록"""
        return []  # 유연한 특성 생성

    def _get_stage(self) -> ProcessingStage:
        """현재 처리 단계"""
        return ProcessingStage.FEATURED


class DataScaler(BaseProcessor):
    """데이터 스케일링 처리기"""

    def __init__(self, scaling_method: str = "standard"):
        super().__init__("data_scaler")
        self.scaling_method = scaling_method
        self.scalers = {}
        self.feature_columns = []

    def fit(self, data: pd.DataFrame) -> 'DataScaler':
        """스케일러 학습"""
        self.logger.info(f"데이터 스케일러 학습 시작 (방법: {self.scaling_method})")

        # 스케일링할 컬럼 선택 (수치형 컬럼만)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour', 'is_operating']
        self.feature_columns = [col for col in numeric_columns if col not in exclude_columns]

        # 스케일러 생성 및 학습
        for col in self.feature_columns:
            if self.scaling_method == "standard":
                scaler = StandardScaler()
            elif self.scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif self.scaling_method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"지원하지 않는 스케일링 방법: {self.scaling_method}")

            scaler.fit(data[[col]])
            self.scalers[col] = scaler

        self.is_fitted = True
        self.logger.info(f"데이터 스케일러 학습 완료 ({len(self.feature_columns)} 컬럼)")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 스케일링"""
        if not self.is_fitted:
            raise DataProcessingError("스케일러가 학습되지 않음")

        scaled_data = data.copy()

        # 각 컬럼별 스케일링 적용
        for col in self.feature_columns:
            if col in scaled_data.columns:
                scaled_values = self.scalers[col].transform(scaled_data[[col]])
                scaled_data[col] = scaled_values.flatten()

        return scaled_data

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """스케일링 역변환"""
        if not self.is_fitted:
            raise DataProcessingError("스케일러가 학습되지 않음")

        original_data = data.copy()

        # 각 컬럼별 역변환 적용
        for col in self.feature_columns:
            if col in original_data.columns:
                original_values = self.scalers[col].inverse_transform(original_data[[col]])
                original_data[col] = original_values.flatten()

        return original_data

    def _get_required_columns(self) -> List[str]:
        """필수 컬럼 목록"""
        return self.feature_columns

    def _get_stage(self) -> ProcessingStage:
        """현재 처리 단계"""
        return ProcessingStage.SCALED


class DataProcessingPipeline:
    """데이터 처리 파이프라인"""

    def __init__(self, processors: List[BaseProcessor] = None):
        self.logger = get_logger("processing_pipeline")
        self.config = get_config()
        self.processors = processors or []
        self.is_fitted = False
        self.pipeline_stats = {
            'total_processed': 0,
            'success_rate': 0.0,
            'average_processing_time': 0.0
        }

    def add_processor(self, processor: BaseProcessor):
        """전처리기 추가"""
        self.processors.append(processor)
        self.logger.info(f"전처리기 추가: {processor.processor_name}")

    def fit(self, data: pd.DataFrame) -> 'DataProcessingPipeline':
        """파이프라인 학습"""
        self.logger.info("데이터 처리 파이프라인 학습 시작")

        current_data = data.copy()

        for processor in self.processors:
            self.logger.info(f"전처리기 학습: {processor.processor_name}")
            processor.fit(current_data)
            current_data = processor.transform(current_data)

        self.is_fitted = True
        self.logger.info("데이터 처리 파이프라인 학습 완료")
        return self

    @log_performance
    def transform(self, data: pd.DataFrame) -> ProcessingResult:
        """데이터 변환"""
        if not self.is_fitted:
            raise DataProcessingError("파이프라인이 학습되지 않음")

        start_time = datetime.now()
        current_data = data.copy()
        all_errors = []
        all_metadata = {}

        # 각 전처리기 순차 적용
        for processor in self.processors:
            result = processor.process(current_data)
            current_data = result.data
            all_errors.extend(result.error_log)
            all_metadata[processor.processor_name] = result.metadata

        # 파이프라인 통계 업데이트
        processing_time = (datetime.now() - start_time).total_seconds()
        self.pipeline_stats['total_processed'] += 1

        success = len(all_errors) == 0
        if success:
            self.pipeline_stats['success_rate'] = (
                (self.pipeline_stats['success_rate'] * (self.pipeline_stats['total_processed'] - 1) + 1) /
                self.pipeline_stats['total_processed']
            )

        self.pipeline_stats['average_processing_time'] = (
            (self.pipeline_stats['average_processing_time'] *
             (self.pipeline_stats['total_processed'] - 1) + processing_time) /
            self.pipeline_stats['total_processed']
        )

        return ProcessingResult(
            data=current_data,
            stage=ProcessingStage.READY,
            metadata={
                'pipeline_stats': self.pipeline_stats,
                'processors': all_metadata,
                'total_processors': len(self.processors)
            },
            processing_time=processing_time,
            error_log=all_errors
        )

    def fit_transform(self, data: pd.DataFrame) -> ProcessingResult:
        """학습 후 변환"""
        self.fit(data)
        return self.transform(data)

    def save_pipeline(self, filepath: str):
        """파이프라인 저장"""
        pipeline_data = {
            'processors': self.processors,
            'is_fitted': self.is_fitted,
            'pipeline_stats': self.pipeline_stats
        }
        joblib.dump(pipeline_data, filepath)
        self.logger.info(f"파이프라인 저장: {filepath}")

    def load_pipeline(self, filepath: str):
        """파이프라인 로드"""
        pipeline_data = joblib.load(filepath)
        self.processors = pipeline_data['processors']
        self.is_fitted = pipeline_data['is_fitted']
        self.pipeline_stats = pipeline_data['pipeline_stats']
        self.logger.info(f"파이프라인 로드: {filepath}")


class RealTimeProcessor:
    """실시간 데이터 처리기"""

    def __init__(self, pipeline: DataProcessingPipeline):
        self.logger = get_logger("realtime_processor")
        self.pipeline = pipeline
        self.buffer_size = 1000
        self.data_buffer = []
        self.processing_interval = 60  # seconds

    def add_data(self, data: Union[pd.DataFrame, Dict]):
        """실시간 데이터 추가"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        self.data_buffer.append(data)

        # 버퍼 크기 제한
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)

    def process_buffer(self) -> Optional[ProcessingResult]:
        """버퍼 데이터 처리"""
        if not self.data_buffer:
            return None

        try:
            # 버퍼 데이터 결합
            combined_data = pd.concat(self.data_buffer, ignore_index=True)

            # 파이프라인 적용
            result = self.pipeline.transform(combined_data)

            # 버퍼 클리어
            self.data_buffer.clear()

            return result

        except Exception as e:
            self.logger.error(f"실시간 처리 오류: {e}")
            return None


# 팩토리 함수들
def create_default_pipeline(scaling_method: str = "standard") -> DataProcessingPipeline:
    """기본 처리 파이프라인 생성"""
    pipeline = DataProcessingPipeline()

    # 처리 단계 순서대로 추가
    pipeline.add_processor(DataCleaner())
    pipeline.add_processor(FeatureEngineer())
    pipeline.add_processor(DataScaler(scaling_method))

    return pipeline


def create_custom_pipeline(processor_configs: List[Dict]) -> DataProcessingPipeline:
    """사용자 정의 파이프라인 생성"""
    pipeline = DataProcessingPipeline()

    for config in processor_configs:
        processor_type = config.get('type')
        processor_params = config.get('params', {})

        if processor_type == 'cleaner':
            processor = DataCleaner()
        elif processor_type == 'feature_engineer':
            processor = FeatureEngineer()
        elif processor_type == 'scaler':
            processor = DataScaler(**processor_params)
        else:
            raise ValueError(f"지원하지 않는 전처리기 타입: {processor_type}")

        pipeline.add_processor(processor)

    return pipeline


def create_realtime_processor(pipeline: DataProcessingPipeline = None) -> RealTimeProcessor:
    """실시간 처리기 생성"""
    if pipeline is None:
        pipeline = create_default_pipeline()

    return RealTimeProcessor(pipeline)


# 사용 예시
if __name__ == "__main__":
    # 기본 파이프라인 생성
    pipeline = create_default_pipeline()

    # 샘플 데이터로 테스트
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'machine_id': ['machine_1'] * 50 + ['machine_2'] * 50,
        'power_consumption': np.random.normal(100, 20, 100),
        'voltage': np.random.normal(220, 10, 100),
        'current': np.random.normal(0.5, 0.1, 100),
        'temperature': np.random.normal(25, 5, 100)
    })

    # 파이프라인 학습 및 변환
    result = pipeline.fit_transform(sample_data)

    print(f"처리 결과: {result.stage}")
    print(f"처리 시간: {result.processing_time:.3f}초")
    print(f"오류 수: {len(result.error_log)}")
    print(f"결과 데이터 형태: {result.data.shape}")