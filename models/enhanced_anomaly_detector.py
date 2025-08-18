"""
향상된 이상탐지 모델 - ipynb 통합 시스템 기반
지능형 이상치 라벨링, GPU 최적화, 다중 알고리즘 융합
"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

from .base_model import ClassificationModel
from .anomaly_detector import AnomalyDetector
from ..core.config import get_config
from ..core.logger import get_logger, log_performance, timer
from ..core.exceptions import ModelTrainingError, ModelPredictionError


class IntelligentAnomalyLabeler:
    """지능형 이상치 라벨링 - 다중 소스 융합"""

    def __init__(self, contamination_rate: float = 0.05):
        self.contamination_rate = contamination_rate
        self.logger = get_logger("intelligent_labeler")

    def create_intelligent_labels(
            self,
            X: pd.DataFrame,
            y: Optional[pd.Series] = None,
            return_details: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """지능형 이상치 라벨링 - 여러 기법 융합"""

        self.logger.info(f"지능형 이상치 탐지 (contamination: {self.contamination_rate:.1%})")

        anomaly_scores = np.zeros(len(X))
        anomaly_sources = {}

        # 1. 물리적 임계값 기반
        physical_score = self._calculate_physical_anomalies(X)
        anomaly_sources['physical'] = (physical_score > 0).sum()

        # 2. 통계적 이상치 (Z-score)
        statistical_score = self._calculate_statistical_anomalies(X, y)
        anomaly_sources['statistical'] = (statistical_score > 0).sum()

        # 3. 효율성 기반 이상치
        efficiency_score = self._calculate_efficiency_anomalies(X, y)
        anomaly_sources['efficiency'] = (efficiency_score > 0).sum()

        # 4. 머신러닝 기반 (Isolation Forest)
        ml_score = self._calculate_ml_anomalies(X)
        anomaly_sources['ml_isolation'] = (ml_score > 3).sum()

        # 5. 시계열 패턴 이상
        temporal_score = self._calculate_temporal_anomalies(X, y)
        anomaly_sources['temporal'] = (temporal_score > 0).sum()

        # 6. 종합 점수 계산 (가중 합)
        total_anomaly_scores = (
                physical_score * 1.5 +  # 물리적 이상 중시
                statistical_score * 1.2 +  # 통계적 이상
                efficiency_score * 1.3 +  # 효율성 이상 중시
                ml_score * 1.0 +  # ML 기반
                temporal_score * 0.8  # 시계열 패턴
        )

        # 7. 최종 이상치 선정
        target_anomalies = int(len(X) * self.contamination_rate)

        if target_anomalies > 0:
            threshold = np.percentile(total_anomaly_scores, (1 - self.contamination_rate) * 100)
            final_anomalies = (total_anomaly_scores >= threshold).astype(int)
        else:
            final_anomalies = np.zeros(len(X), dtype=int)

        final_rate = final_anomalies.mean()

        self.logger.info(f"이상치 소스별 개수:")
        for source, count in anomaly_sources.items():
            self.logger.info(f"  {source}: {count:,}개")
        self.logger.info(f"최종 이상치: {final_anomalies.sum():,}개 ({final_rate:.3%})")

        if return_details:
            return final_anomalies, {
                'scores': total_anomaly_scores,
                'sources': anomaly_sources,
                'threshold': threshold if target_anomalies > 0 else 0
            }

        return final_anomalies

    def _calculate_physical_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """물리적 임계값 기반 이상치"""
        physical_score = np.zeros(len(X))

        # 온도 이상
        temp_cols = [col for col in X.columns if 'temp' in col.lower()]
        for col in temp_cols:
            if col in X.columns:
                high_temp = X[col] > X[col].quantile(0.98)
                low_temp = X[col] < X[col].quantile(0.02)
                physical_score += (high_temp.astype(int) * 2 + low_temp.astype(int))

        # 진동 이상
        vib_cols = [col for col in X.columns if 'vib' in col.lower()]
        for col in vib_cols:
            if col in X.columns:
                high_vib = X[col] > X[col].quantile(0.97)
                physical_score += high_vib.astype(int) * 3

        return physical_score

    def _calculate_statistical_anomalies(self, X: pd.DataFrame, y: Optional[pd.Series]) -> np.ndarray:
        """통계적 이상치 (Z-score)"""
        statistical_score = np.zeros(len(X))

        # 전력 이상치
        if y is not None:
            y_zscore = np.abs(stats.zscore(y))
            statistical_score += (y_zscore > 3).astype(int) * 4

        # 주요 피처 이상치
        important_features = ['Total_Current', 'Load_Torque', 'RMS_Vibration', 'Motor_Temperature']
        for feature in important_features:
            if feature in X.columns:
                feature_zscore = np.abs(stats.zscore(X[feature]))
                statistical_score += (feature_zscore > 3.5).astype(int)

        return statistical_score

    def _calculate_efficiency_anomalies(self, X: pd.DataFrame, y: Optional[pd.Series]) -> np.ndarray:
        """효율성 기반 이상치"""
        efficiency_score = np.zeros(len(X))

        if y is not None:
            # 전력 효율성 = 출력 / 전류
            if 'Total_Current' in X.columns:
                efficiency = y / (X['Total_Current'] + 1e-6)
                eff_zscore = np.abs(stats.zscore(efficiency))
                efficiency_score += (eff_zscore > 3).astype(int) * 2

            # 작업 효율성 = 전력 / 작업부하
            if 'Workload_Percentage' in X.columns:
                work_efficiency = y / (X['Workload_Percentage'] + 1e-6)
                work_eff_zscore = np.abs(stats.zscore(work_efficiency))
                efficiency_score += (work_eff_zscore > 3).astype(int) * 2

        return efficiency_score

    def _calculate_ml_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """머신러닝 기반 이상치 (Isolation Forest)"""
        try:
            # 샘플링으로 성능 최적화
            sample_size = min(20000, len(X))
            if len(X) > sample_size:
                sample_idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_idx]
            else:
                X_sample = X

            iso_forest = IsolationForest(
                contamination=min(0.15, self.contamination_rate * 3),
                random_state=42,
                n_jobs=-1
            )
            iso_forest.fit(X_sample)

            # 이상 점수 계산
            anomaly_scores_iso = iso_forest.decision_function(X)
            score_range = np.ptp(anomaly_scores_iso) + 1e-8
            scores_normalized = (anomaly_scores_iso - anomaly_scores_iso.min()) / score_range
            ml_score = (1 - scores_normalized) * 5  # 이상할수록 높은 점수

            return ml_score

        except Exception as e:
            self.logger.warning(f"Isolation Forest 실패: {e}")
            return np.zeros(len(X))

    def _calculate_temporal_anomalies(self, X: pd.DataFrame, y: Optional[pd.Series]) -> np.ndarray:
        """시계열 패턴 이상"""
        temporal_score = np.zeros(len(X))

        if y is not None and len(y) > 10:
            # 전력 변화량 이상
            power_diff = np.abs(np.diff(y, prepend=y.iloc[0] if hasattr(y, 'iloc') else y[0]))
            power_diff_zscore = np.abs(stats.zscore(power_diff))
            temporal_score += (power_diff_zscore > 3).astype(int) * 2

        return temporal_score


class EnhancedAnomalyDetector(ClassificationModel):
    """향상된 이상탐지 모델 - GPU 최적화 및 다중 알고리즘 융합"""

    def __init__(self, model_name: str = "enhanced_anomaly_detector"):
        super().__init__(model_name, "enhanced_anomaly_detection")

        # GPU 설정
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # 모델 구성
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.labeler = IntelligentAnomalyLabeler()

        # 성능 추적
        self.performance_history = {}

        self.logger.info(f"향상된 이상탐지 모델 초기화 (GPU: {self.use_gpu})")

    def _create_model(self) -> Dict[str, Any]:
        """다중 모델 생성"""
        models = {}

        # 1) GPU 기반 모델들 (가능한 경우)
        if self.use_gpu:
            models.update(self._create_gpu_models())

        # 2) CPU 기반 안정적 모델들
        models.update(self._create_cpu_models())

        self.logger.info(f"생성된 모델 개수: {len(models)}")
        return models

    def _create_gpu_models(self) -> Dict[str, Any]:
        """GPU 기반 모델 생성"""
        gpu_models = {}

        try:
            # RAPIDS cuML (가능한 경우)
            try:
                from cuml.ensemble import RandomForestClassifier as cuRF
                gpu_models['cuml_rf'] = cuRF(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                )
                self.logger.info("cuML RandomForest 추가")
            except ImportError:
                self.logger.info("cuML 사용 불가")

            # XGBoost GPU
            try:
                import xgboost as xgb
                gpu_models['xgb_gpu'] = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    tree_method='gpu_hist',
                    random_state=42,
                    eval_metric='logloss',
                    verbosity=0
                )
                self.logger.info("XGBoost GPU 추가")
            except ImportError:
                self.logger.info("XGBoost 사용 불가")

        except Exception as e:
            self.logger.warning(f"GPU 모델 생성 실패: {e}")

        return gpu_models

    def _create_cpu_models(self) -> Dict[str, Any]:
        """CPU 기반 안정적 모델 생성"""
        return {
            'sklearn_rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=4
            ),
            'isolation_forest': IsolationForest(
                n_estimators=150,
                contamination=0.05,
                random_state=42,
                n_jobs=4
            )
        }

    @log_performance("enhanced_anomaly_training")
    def _train_model(
            self,
            X: pd.DataFrame,
            y: pd.Series = None,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """향상된 이상탐지 모델 학습"""
        training_start = pd.Timestamp.now()

        # 지능형 라벨링으로 이상치 생성
        if y is None:
            # 무감독 학습을 위한 지능형 라벨링
            y_anomaly, labeling_details = self.labeler.create_intelligent_labels(
                X, return_details=True
            )
        else:
            y_anomaly = y
            labeling_details = {}

        # 데이터 스케일링
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # 모델 생성
        self.models = self._create_model()

        # 각 모델 학습
        training_results = {}
        for name, model in self.models.items():
            try:
                start_time = pd.Timestamp.now()

                if name == 'cuml_rf' and self.use_gpu:
                    # cuML 모델 학습
                    import cudf
                    X_gpu = cudf.DataFrame(X_scaled.iloc[:50000] if len(X_scaled) > 50000 else X_scaled)
                    y_gpu = cudf.Series(y_anomaly[:50000] if len(y_anomaly) > 50000 else y_anomaly)
                    model.fit(X_gpu, y_gpu)

                elif name == 'isolation_forest':
                    # 비지도 학습
                    model.fit(X_scaled)

                else:
                    # 일반 지도 학습
                    model.fit(X_scaled, y_anomaly)

                train_time = (pd.Timestamp.now() - start_time).total_seconds()
                training_results[name] = {
                    'training_time': train_time,
                    'success': True
                }

                self.logger.info(f"{name} 학습 완료: {train_time:.1f}초")

                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"{name} 학습 실패: {str(e)}")
                training_results[name] = {
                    'training_time': 0,
                    'success': False,
                    'error': str(e)
                }
                # 실패한 모델 제거
                if name in self.models:
                    del self.models[name]

        # 성공한 모델들의 가중치 설정
        self._calculate_model_weights()

        total_training_time = (pd.Timestamp.now() - training_start).total_seconds()

        result = {
            'labeling_details': labeling_details,
            'model_results': training_results,
            'successful_models': len([r for r in training_results.values() if r['success']]),
            'total_training_time': total_training_time,
            'anomaly_ratio': y_anomaly.mean(),
            'gpu_used': self.use_gpu,
            'model_weights': self.weights
        }

        return result

    def _calculate_model_weights(self):
        """모델별 가중치 계산"""
        # 기본 가중치 (성능 기반으로 추후 조정 가능)
        default_weights = {
            'cuml_rf': 0.3,
            'xgb_gpu': 0.3,
            'sklearn_rf': 0.25,
            'isolation_forest': 0.15
        }

        # 실제 존재하는 모델에만 가중치 할당
        total_weight = 0
        for name in self.models.keys():
            if name in default_weights:
                self.weights[name] = default_weights[name]
                total_weight += default_weights[name]
            else:
                self.weights[name] = 0.2
                total_weight += 0.2

        # 가중치 정규화
        if total_weight > 0:
            for name in self.weights:
                self.weights[name] /= total_weight

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """앙상블 예측"""
        if not self.models:
            raise ModelPredictionError("학습된 모델이 없습니다")

        # 데이터 스케일링
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        # 각 모델의 예측
        predictions = {}
        probabilities = {}

        for name, model in self.models.items():
            try:
                if name == 'cuml_rf' and self.use_gpu:
                    import cudf
                    X_gpu = cudf.DataFrame(X_scaled)
                    pred = model.predict(X_gpu)
                    if hasattr(pred, 'to_pandas'):
                        pred = pred.to_pandas().values

                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_gpu)[:, 1]
                        if hasattr(prob, 'to_pandas'):
                            prob = prob.to_pandas().values
                    else:
                        prob = pred.astype(float)

                elif name == 'isolation_forest':
                    pred = model.predict(X_scaled)
                    pred = (pred == -1).astype(int)  # -1(이상) -> 1, 1(정상) -> 0
                    prob = -model.score_samples(X_scaled)
                    prob_range = np.ptp(prob) + 1e-8
                    prob = (prob - prob.min()) / prob_range

                else:
                    pred = model.predict(X_scaled)
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[:, 1]
                    else:
                        prob = pred.astype(float)

                predictions[name] = pred
                probabilities[name] = prob

            except Exception as e:
                self.logger.warning(f"{name} 예측 실패: {str(e)}")
                continue

        if not predictions:
            raise ModelPredictionError("모든 모델의 예측이 실패했습니다")

        # 가중 앙상블
        weighted_prob = np.zeros(len(X))
        total_weight = 0

        for name, prob in probabilities.items():
            weight = self.weights.get(name, 0.2)
            weighted_prob += weight * prob
            total_weight += weight

        if total_weight > 0:
            weighted_prob /= total_weight

        ensemble_pred = (weighted_prob >= 0.5).astype(int)

        return ensemble_pred

    def predict_with_details(self, X: pd.DataFrame) -> Dict[str, Any]:
        """상세 예측 결과 반환"""
        # 기본 예측
        ensemble_pred = self.predict(X)

        # 개별 모델 예측도 포함
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        individual_predictions = {}
        individual_probabilities = {}

        for name, model in self.models.items():
            try:
                if name == 'isolation_forest':
                    pred = model.predict(X_scaled)
                    pred = (pred == -1).astype(int)
                    prob = -model.score_samples(X_scaled)
                    prob_range = np.ptp(prob) + 1e-8
                    prob = (prob - prob.min()) / prob_range
                else:
                    pred = model.predict(X_scaled)
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[:, 1]
                    else:
                        prob = pred.astype(float)

                individual_predictions[name] = pred
                individual_probabilities[name] = prob

            except Exception as e:
                self.logger.warning(f"{name} 상세 예측 실패: {str(e)}")
                continue

        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': individual_predictions,
            'individual_probabilities': individual_probabilities,
            'model_weights': self.weights.copy()
        }

    def get_anomaly_scores(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """각 모델의 이상 점수 반환"""
        scores = {}
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        for name, model in self.models.items():
            try:
                if name == 'isolation_forest':
                    score = -model.score_samples(X_scaled)
                elif hasattr(model, 'decision_function'):
                    score = model.decision_function(X_scaled)
                elif hasattr(model, 'predict_proba'):
                    score = model.predict_proba(X_scaled)[:, 1]
                else:
                    score = model.predict(X_scaled).astype(float)

                scores[name] = score

            except Exception as e:
                self.logger.warning(f"{name} 점수 계산 실패: {str(e)}")

        return scores


class RealTimeEnhancedDetector:
    """실시간 향상된 이상탐지"""

    def __init__(self, contamination: float = 0.03):
        self.logger = get_logger("realtime_enhanced")
        self.detector = EnhancedAnomalyDetector()
        self.is_initialized = False
        self.history_buffer = []
        self.max_buffer_size = 1000
        self.alert_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }

    def initialize(self, X_train: pd.DataFrame, feature_names: List[str]) -> None:
        """실시간 탐지기 초기화"""
        self.logger.info("실시간 향상된 이상탐지 초기화")

        with timer("realtime_enhanced_training"):
            self.detector.train(X_train, None)
            self.feature_names = feature_names
            self.is_initialized = True

        self.logger.info("실시간 향상된 이상탐지 초기화 완료")

    def detect_single(self, data_point: Dict[str, float]) -> Dict[str, Any]:
        """단일 데이터 포인트 향상된 이상탐지"""
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

        # 상세 예측 수행
        detailed_results = self.detector.predict_with_details(df)

        # 앙상블 점수 계산
        anomaly_scores = self.detector.get_anomaly_scores(df)
        ensemble_score = np.mean([score[0] for score in anomaly_scores.values() if len(score) > 0])

        result = {
            'is_anomaly': bool(detailed_results['ensemble_prediction'][0]),
            'ensemble_score': float(ensemble_score),
            'individual_predictions': {k: bool(v[0]) for k, v in detailed_results['individual_predictions'].items()},
            'individual_scores': {k: float(v[0]) for k, v in anomaly_scores.items()},
            'confidence': self._calculate_confidence(detailed_results, anomaly_scores),
            'alert_level': self._get_alert_level(ensemble_score),
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_consensus': self._calculate_consensus(detailed_results['individual_predictions'])
        }

        # 히스토리에 추가
        self._add_to_history(result)

        return result

    def _calculate_confidence(self, detailed_results: Dict, scores: Dict) -> float:
        """예측 신뢰도 계산"""
        # 모델 간 합의도
        predictions = list(detailed_results['individual_predictions'].values())
        if not predictions:
            return 0.0

        consensus = np.mean([pred[0] for pred in predictions])
        consensus_confidence = abs(consensus - 0.5) * 2  # 0.5에서 멀수록 확신

        # 점수 일관성
        score_values = [score[0] for score in scores.values() if len(score) > 0]
        if score_values:
            score_std = np.std(score_values)
            score_confidence = 1 / (1 + score_std)  # 표준편차가 낮을수록 높은 신뢰도
        else:
            score_confidence = 0.5

        return (consensus_confidence + score_confidence) / 2

    def _calculate_consensus(self, individual_predictions: Dict) -> float:
        """모델 간 합의도 계산"""
        if not individual_predictions:
            return 0.0

        predictions = [pred[0] for pred in individual_predictions.values()]
        return np.mean(predictions)

    def _get_alert_level(self, score: float) -> str:
        """알림 레벨 결정"""
        if score >= self.alert_thresholds['high']:
            return "HIGH"
        elif score >= self.alert_thresholds['medium']:
            return "MEDIUM"
        elif score >= self.alert_thresholds['low']:
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

    def get_system_health(self) -> Dict[str, Any]:
        """시스템 건강도 평가"""
        if not self.history_buffer:
            return {'status': 'no_data'}

        recent_results = self.history_buffer[-100:]  # 최근 100개

        anomaly_rate = np.mean([r['is_anomaly'] for r in recent_results])
        avg_confidence = np.mean([r['confidence'] for r in recent_results])
        alert_distribution = {}

        for level in ['NORMAL', 'LOW', 'MEDIUM', 'HIGH']:
            alert_distribution[level] = sum(1 for r in recent_results if r['alert_level'] == level)

        # 시스템 상태 판정
        if anomaly_rate > 0.3:
            status = 'critical'
        elif anomaly_rate > 0.1:
            status = 'warning'
        elif avg_confidence < 0.5:
            status = 'uncertain'
        else:
            status = 'healthy'

        return {
            'status': status,
            'anomaly_rate': anomaly_rate,
            'average_confidence': avg_confidence,
            'alert_distribution': alert_distribution,
            'recent_samples': len(recent_results),
            'last_update': pd.Timestamp.now().isoformat()
        }


# 팩토리 함수들
def create_enhanced_anomaly_detector() -> EnhancedAnomalyDetector:
    """향상된 이상탐지 모델 생성"""
    return EnhancedAnomalyDetector()


def create_realtime_enhanced_detector(contamination: float = 0.03) -> RealTimeEnhancedDetector:
    """실시간 향상된 이상탐지 모델 생성"""
    return RealTimeEnhancedDetector(contamination)


def create_intelligent_labeler(contamination_rate: float = 0.05) -> IntelligentAnomalyLabeler:
    """지능형 이상치 라벨러 생성"""
    return IntelligentAnomalyLabeler(contamination_rate)