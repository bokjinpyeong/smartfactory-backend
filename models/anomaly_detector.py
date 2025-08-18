"""
스마트팩토리 에너지 관리 시스템 이상탐지 모델
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

from .base_model import ClassificationModel
from ..core.config import get_config
from ..core.logger import get_logger, log_performance, timer
from ..core.exceptions import ModelTrainingError, ModelPredictionError


class AnomalyDetector(ClassificationModel):
    """이상탐지 기본 클래스"""

    def __init__(self, model_name: str = "anomaly_detector", algorithm: str = "isolation_forest"):
        super().__init__(model_name, "anomaly_detection")
        self.algorithm = algorithm
        self.scaler = StandardScaler()
        self.threshold = 0.0
        self.contamination = self.config.model.anomaly_contamination

        # 알고리즘별 설정
        self.algorithm_configs = {
            'isolation_forest': {
                'contamination': self.contamination,
                'n_estimators': self.config.model.anomaly_n_estimators,
                'max_samples': self.config.model.anomaly_max_samples,
                'random_state': 42,
                'n_jobs': -1
            },
            'one_class_svm': {
                'nu': self.contamination,
                'kernel': 'rbf',
                'gamma': 'scale'
            }
        }

        self.logger.info(f"이상탐지 모델 초기화: {algorithm}")

    def _create_model(self) -> Any:
        """이상탐지 모델 생성"""
        config = self.algorithm_configs.get(self.algorithm, {})

        if self.algorithm == "isolation_forest":
            return IsolationForest(**config)
        elif self.algorithm == "one_class_svm":
            return OneClassSVM(**config)
        else:
            raise ValueError(f"지원하지 않는 알고리즘: {self.algorithm}")

    @log_performance("anomaly_training")
    def _train_model(
            self,
            X: pd.DataFrame,
            y: pd.Series = None,  # 비지도 학습이므로 사용하지 않음
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """이상탐지 모델 학습"""
        training_start = pd.Timestamp.now()

        # 데이터 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 모델 학습 (정상 데이터만 사용)
        self.model.fit(X_scaled)

        # 임계값 설정 (훈련 데이터의 anomaly score 분포 기반)
        anomaly_scores = self.model.decision_function(X_scaled)
        self.threshold = np.percentile(anomaly_scores, self.contamination * 100)

        training_time = (pd.Timestamp.now() - training_start).total_seconds()

        # 훈련 데이터에 대한 예측 수행
        train_predictions = self.model.predict(X_scaled)
        anomaly_count = (train_predictions == -1).sum()

        training_result = {
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'threshold': self.threshold,
            'training_samples': len(X),
            'detected_anomalies': anomaly_count,
            'anomaly_ratio': anomaly_count / len(X),
            'training_time': training_time,
            'config': self.algorithm_configs[self.algorithm]
        }

        self.logger.info(f"이상탐지 모델 학습 완료: {anomaly_count}개 이상치 탐지 ({anomaly_count / len(X) * 100:.2f}%)")

        return training_result

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """이상탐지 예측"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # -1(이상), 1(정상) → 1(이상), 0(정상)으로 변환
        return (predictions == -1).astype(int)

    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """이상점수 반환"""
        if not self.is_trained:
            raise ModelPredictionError("모델이 학습되지 않았습니다")

        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    def predict_with_scores(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """예측과 이상점수를 함께 반환"""
        scores = self.get_anomaly_scores(X)
        predictions = (scores < self.threshold).astype(int)
        return predictions, scores


class RealTimeAnomalyDetector:
    """실시간 이상탐지"""

    def __init__(self, contamination: float = 0.03):
        self.logger = get_logger("realtime_anomaly")
        self.detector = AnomalyDetector(contamination=contamination)
        self.is_initialized = False
        self.alert_threshold = 0.7
        self.history_buffer = []
        self.max_buffer_size = 1000

    def train(self, X_train: pd.DataFrame, feature_names: List[str]) -> None:
        """실시간 탐지기 학습"""
        self.logger.info("실시간 이상탐지 모델 학습 시작")

        with timer("realtime_anomaly_training"):
            self.detector.train(X_train, None)
            self.feature_names = feature_names
            self.is_initialized = True

        self.logger.info("실시간 이상탐지 모델 학습 완료")

    def detect_single(self, data_point: Dict[str, float]) -> Dict[str, Any]:
        """단일 데이터 포인트 이상탐지"""
        if not self.is_initialized:
            raise ModelPredictionError("탐지기가 초기화되지 않았습니다")

        # 데이터 포인트를 DataFrame으로 변환
        df = pd.DataFrame([data_point])

        # 필요한 피처만 선택
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                raise ModelPredictionError(f"필요한 피처가 누락되었습니다: {missing_features}")
            df = df[self.feature_names]

        # 이상탐지 수행
        prediction, score = self.detector.predict_with_scores(df)

        result = {
            'is_anomaly': bool(prediction[0]),
            'anomaly_score': float(score[0]),
            'confidence': abs(float(score[0])) / abs(self.detector.threshold) if self.detector.threshold != 0 else 1.0,
            'timestamp': pd.Timestamp.now().isoformat(),
            'alert_level': self._get_alert_level(score[0])
        }

        # 히스토리에 추가
        self._add_to_history(result)

        return result

    def predict_batch(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """배치 데이터 이상탐지"""
        if not self.is_initialized:
            raise ModelPredictionError("탐지기가 초기화되지 않았습니다")

        predictions, scores = self.detector.predict_with_scores(X)

        # 배치 결과를 히스토리에 추가
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            result = {
                'is_anomaly': bool(pred),
                'anomaly_score': float(score),
                'confidence': abs(float(score)) / abs(self.detector.threshold) if self.detector.threshold != 0 else 1.0,
                'timestamp': pd.Timestamp.now().isoformat(),
                'alert_level': self._get_alert_level(score)
            }
            self._add_to_history(result)

        return predictions, scores

    def _get_alert_level(self, score: float) -> str:
        """알림 레벨 결정"""
        confidence = abs(score) / abs(self.detector.threshold) if self.detector.threshold != 0 else 1.0

        if score < self.detector.threshold:  # 이상치인 경우
            if confidence > 2.0:
                return "HIGH"
            elif confidence > 1.5:
                return "MEDIUM"
            else:
                return "LOW"
        else:
            return "NORMAL"

    def _add_to_history(self, result: Dict[str, Any]) -> None:
        """히스토리에 결과 추가"""
        self.history_buffer.append(result)

        # 버퍼 크기 제한
        if len(self.history_buffer) > self.max_buffer_size:
            self.history_buffer.pop(0)

    def get_recent_anomalies(self, hours: int = 24) -> List[Dict[str, Any]]:
        """최근 이상치 반환"""
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)

        recent_anomalies = [
            result for result in self.history_buffer
            if (result['is_anomaly'] and
                pd.Timestamp(result['timestamp']) > cutoff_time)
        ]

        return recent_anomalies

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """최근 통계 반환"""
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)

        recent_results = [
            result for result in self.history_buffer
            if pd.Timestamp(result['timestamp']) > cutoff_time
        ]

        if not recent_results:
            return {'total_count': 0}

        total_count = len(recent_results)
        anomaly_count = sum(1 for r in recent_results if r['is_anomaly'])

        alert_levels = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'NORMAL': 0}
        for result in recent_results:
            alert_levels[result['alert_level']] += 1

        return {
            'total_count': total_count,
            'anomaly_count': anomaly_count,
            'anomaly_ratio': anomaly_count / total_count,
            'alert_levels': alert_levels,
            'period_hours': hours
        }


class PowerAnomalyDetector(AnomalyDetector):
    """전력 소비 이상탐지 전용 클래스"""

    def __init__(self, model_name: str = "power_anomaly_detector"):
        super().__init__(model_name, "isolation_forest")
        self.power_thresholds = {}
        self.machine_profiles = {}

    def train_with_power_context(
            self,
            X: pd.DataFrame,
            power_col: str = 'Power_Consumption_Realistic',
            machine_col: str = 'Machine_ID'
    ) -> Dict[str, Any]:
        """전력 컨텍스트를 고려한 학습"""

        # 기본 이상탐지 모델 학습
        training_result = self.train(X, None)

        # 기계별 전력 프로파일 생성
        if machine_col in X.columns:
            self._create_machine_profiles(X, power_col, machine_col)

        # 전력 임계값 설정
        self._set_power_thresholds(X, power_col)

        training_result.update({
            'power_thresholds': self.power_thresholds,
            'machine_profiles_count': len(self.machine_profiles)
        })

        return training_result

    def _create_machine_profiles(
            self,
            X: pd.DataFrame,
            power_col: str,
            machine_col: str
    ) -> None:
        """기계별 전력 프로파일 생성"""
        for machine_id in X[machine_col].unique():
            machine_data = X[X[machine_col] == machine_id]
            power_data = machine_data[power_col]

            self.machine_profiles[machine_id] = {
                'mean_power': power_data.mean(),
                'std_power': power_data.std(),
                'min_power': power_data.min(),
                'max_power': power_data.max(),
                'percentile_95': power_data.quantile(0.95),
                'percentile_99': power_data.quantile(0.99),
                'sample_count': len(power_data)
            }

    def _set_power_thresholds(self, X: pd.DataFrame, power_col: str) -> None:
        """전력 임계값 설정"""
        power_data = X[power_col]

        self.power_thresholds = {
            'low_threshold': power_data.quantile(0.05),
            'high_threshold': power_data.quantile(0.95),
            'extreme_high_threshold': power_data.quantile(0.99),
            'mean': power_data.mean(),
            'std': power_data.std()
        }

    def detect_power_anomaly(
            self,
            data_point: Dict[str, Any],
            power_col: str = 'Power_Consumption_Realistic',
            machine_col: str = 'Machine_ID'
    ) -> Dict[str, Any]:
        """전력 이상 탐지"""

        # 기본 이상탐지
        basic_result = self.detect_single(data_point)

        # 전력 기반 추가 분석
        power_value = data_point.get(power_col, 0)
        machine_id = data_point.get(machine_col, None)

        power_analysis = {
            'power_value': power_value,
            'is_power_anomaly': False,
            'power_anomaly_type': 'normal',
            'severity': 'normal'
        }

        # 전력 임계값 기반 판정
        if power_value > self.power_thresholds['extreme_high_threshold']:
            power_analysis.update({
                'is_power_anomaly': True,
                'power_anomaly_type': 'extreme_high_consumption',
                'severity': 'critical'
            })
        elif power_value > self.power_thresholds['high_threshold']:
            power_analysis.update({
                'is_power_anomaly': True,
                'power_anomaly_type': 'high_consumption',
                'severity': 'warning'
            })
        elif power_value < self.power_thresholds['low_threshold']:
            power_analysis.update({
                'is_power_anomaly': True,
                'power_anomaly_type': 'low_consumption',
                'severity': 'info'
            })

        # 기계별 프로파일 기반 판정
        if machine_id and machine_id in self.machine_profiles:
            profile = self.machine_profiles[machine_id]

            # Z-score 계산
            z_score = (power_value - profile['mean_power']) / profile['std_power']

            if abs(z_score) > 3:  # 3-sigma 규칙
                power_analysis.update({
                    'is_power_anomaly': True,
                    'power_anomaly_type': f'machine_specific_{"high" if z_score > 0 else "low"}',
                    'severity': 'warning',
                    'z_score': z_score
                })

        # 결과 통합
        result = {
            **basic_result,
            'power_analysis': power_analysis,
            'overall_anomaly': basic_result['is_anomaly'] or power_analysis['is_power_anomaly']
        }

        return result

    def get_machine_profile(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """특정 기계의 프로파일 반환"""
        return self.machine_profiles.get(machine_id)

    def get_power_thresholds(self) -> Dict[str, float]:
        """전력 임계값 반환"""
        return self.power_thresholds.copy()


def create_anomaly_detector(
        algorithm: str = "isolation_forest",
        contamination: float = 0.05
) -> AnomalyDetector:
    """이상탐지 모델 생성 함수"""
    detector = AnomalyDetector(algorithm=algorithm)
    detector.contamination = contamination
    return detector


def create_power_anomaly_detector() -> PowerAnomalyDetector:
    """전력 이상탐지 모델 생성 함수"""
    return PowerAnomalyDetector()


def create_realtime_detector(contamination: float = 0.03) -> RealTimeAnomalyDetector:
    """실시간 이상탐지 모델 생성 함수"""
    return RealTimeAnomalyDetector(contamination)