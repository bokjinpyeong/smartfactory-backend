"""
통합 예측 시스템 - 이상탐지 + 전력예측 통합
ipynb IntegratedPredictionSystem 기반 모듈화
"""
import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

from .enhanced_anomaly_detector import EnhancedAnomalyDetector, IntelligentAnomalyLabeler
from .power_predictor import PowerPredictionEnsemble
from .base_model import EnsembleModel
from ..core.config import get_config
from ..core.logger import get_logger, log_performance, timer
from ..core.exceptions import ModelTrainingError, ModelPredictionError


class IntegratedPredictionSystem(EnsembleModel):
    """통합 예측 시스템 - 이상탐지 + 전력예측"""

    def __init__(self, model_name: str = "integrated_prediction_system"):
        super().__init__(model_name, "integrated_prediction")

        # GPU/CPU 설정
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # 하위 시스템들
        self.anomaly_system = EnhancedAnomalyDetector("anomaly_subsystem")
        self.power_system = PowerPredictionEnsemble("power_subsystem", self.device)
        self.labeler = IntelligentAnomalyLabeler()

        # 데이터 전처리
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_name = ""

        # 성능 추적
        self.training_history = {}
        self.feature_importance = {}

        self.logger.info(f"통합 예측 시스템 초기화 (GPU: {self.use_gpu})")

    def _create_model(self) -> None:
        """통합 시스템은 별도 모델 생성 불필요"""
        return None

    @log_performance("integrated_training")
    def _train_model(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            sample_for_speed: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        """통합 시스템 학습"""
        training_start = pd.Timestamp.now()

        self.logger.info(f"통합 시스템 학습 시작 (데이터: {X.shape})")

        # 피처 이름 저장
        self.feature_names = list(X.columns)
        self.target_name = y.name or "target"

        # 대용량 데이터 안전한 샘플링
        X_sample, y_sample, y_anomaly_sample = self._safe_sampling(
            X, y, sample_for_speed
        )

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        training_times = {}

        # 1) 이상탐지 시스템 학습
        self.logger.info("이상탐지 시스템 학습 중...")
        anomaly_start = time.time()

        try:
            anomaly_result = self.anomaly_system.train(X_sample, None)
            training_times['anomaly_system'] = time.time() - anomaly_start
            self.logger.info(f"이상탐지 학습 완료: {training_times['anomaly_system']:.1f}초")
        except Exception as e:
            self.logger.error(f"이상탐지 학습 실패: {str(e)}")
            raise ModelTrainingError(f"이상탐지 시스템 학습 실패: {str(e)}")

        # 2) 전력예측 시스템 학습
        self.logger.info("전력예측 시스템 학습 중...")
        power_start = time.time()

        try:
            # 검증 데이터 샘플링 (있는 경우)
            X_val_sample, y_val_sample = None, None
            if X_val is not None and y_val is not None:
                val_sample_size = min(20000, len(X_val))
                if len(X_val) > val_sample_size:
                    val_idx = np.random.choice(len(X_val), val_sample_size, replace=False)
                    X_val_sample = X_val.iloc[val_idx]
                    y_val_sample = y_val.iloc[val_idx]
                else:
                    X_val_sample, y_val_sample = X_val, y_val

            power_result = self.power_system.train(
                X_sample, y_sample,
                X_val_sample, y_val_sample
            )
            training_times['power_system'] = time.time() - power_start
            self.logger.info(f"전력예측 학습 완료: {training_times['power_system']:.1f}초")
        except Exception as e:
            self.logger.error(f"전력예측 학습 실패: {str(e)}")
            raise ModelTrainingError(f"전력예측 시스템 학습 실패: {str(e)}")

        # 3) 피처 중요도 통합
        self._integrate_feature_importance()

        total_training_time = (pd.Timestamp.now() - training_start).total_seconds()

        # 학습 결과 정리
        training_result = {
            'anomaly_result': anomaly_result,
            'power_result': power_result,
            'training_times': training_times,
            'total_training_time': total_training_time,
            'sample_sizes': {
                'original': len(X),
                'used': len(X_sample)
            },
            'feature_importance': self.feature_importance,
            'gpu_used': self.use_gpu,
            'device': str(self.device)
        }

        self.training_history = training_result

        self.logger.info(f"통합 시스템 학습 완료: {total_training_time:.1f}초")
        return training_result

    def _safe_sampling(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            sample_for_speed: bool
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """안전한 대용량 데이터 샘플링"""

        if sample_for_speed and len(X) > 100000:
            sample_size = min(100000, len(X) // 3)
            self.logger.info(f"안전한 랜덤 샘플링: {len(X):,} → {sample_size:,}")

            try:
                sample_idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_idx].copy()
                y_sample = y.iloc[sample_idx].copy()

                # 지능형 이상치 라벨링
                y_anomaly_sample = self.labeler.create_intelligent_labels(X_sample, y_sample)

                self.logger.info(f"샘플링 완료: {len(X_sample):,}개")
                self.logger.info(f"샘플 이상치 비율: {y_anomaly_sample.mean():.3%}")

                return X_sample, y_sample, y_anomaly_sample

            except Exception as e:
                self.logger.error(f"샘플링 실패: {str(e)}")
                self.logger.info("전체 데이터 사용 (메모리 주의)")

        # 전체 데이터 사용
        y_anomaly = self.labeler.create_intelligent_labels(X, y)
        return X, y, y_anomaly

    def _integrate_feature_importance(self):
        """하위 시스템들의 피처 중요도 통합"""
        self.feature_importance = {}

        # 이상탐지 시스템 피처 중요도
        if hasattr(self.anomaly_system, 'feature_importance'):
            anomaly_importance = getattr(self.anomaly_system, 'feature_importance', {})
            for model_name, importance in anomaly_importance.items():
                self.feature_importance[f"anomaly_{model_name}"] = importance

        # 전력예측 시스템 피처 중요도
        power_importance = self.power_system.get_feature_importance()
        for model_name, importance in power_importance.items():
            self.feature_importance[f"power_{model_name}"] = importance

    def _predict_model(self, X: pd.DataFrame) -> Dict[str, Any]:
        """통합 예측 (이상탐지 + 전력예측)"""
        # 이상탐지 예측
        anomaly_results = self.anomaly_system.predict_with_details(X)

        # 전력예측
        power_results = self.power_system.predict_ensemble(X)

        return {
            'anomaly_results': anomaly_results,
            'power_results': power_results
        }

    def predict_integrated(self, X: pd.DataFrame) -> Dict[str, Any]:
        """통합 예측 결과 반환"""
        if not self.is_trained:
            raise ModelPredictionError("시스템이 학습되지 않았습니다")

        # 피처 검증
        self._validate_prediction_data(X)

        # 통합 예측 수행
        results = self._predict_model(X)

        # 결과 후처리 및 해석
        interpreted_results = self._interpret_results(results, X)

        return interpreted_results

    def _interpret_results(self, results: Dict[str, Any], X: pd.DataFrame) -> Dict[str, Any]:
        """예측 결과 해석 및 통합"""
        anomaly_results = results['anomaly_results']
        power_results = results['power_results']

        # 기본 예측 결과
        anomaly_predictions = anomaly_results['ensemble_prediction']
        power_predictions = power_results['ensemble_prediction']

        # 신뢰도 및 위험도 계산
        risk_assessment = self._assess_risk(anomaly_results, power_results, X)

        # 권고사항 생성
        recommendations = self._generate_recommendations(anomaly_results, power_results, risk_assessment)

        return {
            'anomaly_detection': {
                'predictions': anomaly_predictions,
                'probabilities': anomaly_results.get('individual_probabilities', {}),
                'individual_results': anomaly_results.get('individual_predictions', {}),
                'anomaly_count': int(anomaly_predictions.sum()),
                'anomaly_ratio': float(anomaly_predictions.mean())
            },
            'power_prediction': {
                'predictions': power_predictions,
                'individual_results': power_results.get('individual_predictions', {}),
                'mean_prediction': float(power_predictions.mean()),
                'std_prediction': float(power_predictions.std()),
                'min_prediction': float(power_predictions.min()),
                'max_prediction': float(power_predictions.max())
            },
            'risk_assessment': risk_assessment,
            'recommendations': recommendations,
            'system_info': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'sample_count': len(X),
                'gpu_used': self.use_gpu
            }
        }

    def _assess_risk(
            self,
            anomaly_results: Dict[str, Any],
            power_results: Dict[str, Any],
            X: pd.DataFrame
    ) -> Dict[str, Any]:
        """위험도 평가"""

        anomaly_predictions = anomaly_results['ensemble_prediction']
        power_predictions = power_results['ensemble_prediction']

        # 이상치 기반 위험도
        anomaly_risk = anomaly_predictions.mean()

        # 전력 소비 패턴 기반 위험도
        power_mean = power_predictions.mean()
        power_std = power_predictions.std()
        power_cv = power_std / (power_mean + 1e-6)  # 변동계수

        # 전력 예측 불확실성
        individual_predictions = power_results.get('individual_predictions', {})
        if len(individual_predictions) > 1:
            pred_array = np.array(list(individual_predictions.values()))
            prediction_uncertainty = np.std(pred_array, axis=0).mean()
        else:
            prediction_uncertainty = 0.0

        # 종합 위험도 계산
        risk_factors = {
            'anomaly_rate': anomaly_risk,
            'power_variability': min(power_cv, 1.0),  # 1로 클리핑
            'prediction_uncertainty': min(prediction_uncertainty / power_mean, 1.0) if power_mean > 0 else 0.0
        }

        # 가중 평균으로 종합 위험도 계산
        weights = {'anomaly_rate': 0.4, 'power_variability': 0.3, 'prediction_uncertainty': 0.3}
        overall_risk = sum(risk_factors[k] * weights[k] for k in weights.keys())

        # 위험 레벨 분류
        if overall_risk > 0.7:
            risk_level = "HIGH"
        elif overall_risk > 0.4:
            risk_level = "MEDIUM"
        elif overall_risk > 0.2:
            risk_level = "LOW"
        else:
            risk_level = "NORMAL"

        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'details': {
                'anomaly_count': int(anomaly_predictions.sum()),
                'high_power_samples': int((power_predictions > power_mean + 2 * power_std).sum()),
                'low_power_samples': int((power_predictions < power_mean - 2 * power_std).sum())
            }
        }

    def _generate_recommendations(
            self,
            anomaly_results: Dict[str, Any],
            power_results: Dict[str, Any],
            risk_assessment: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """권고사항 생성"""
        recommendations = []

        risk_level = risk_assessment['risk_level']
        anomaly_count = risk_assessment['details']['anomaly_count']
        power_predictions = power_results['ensemble_prediction']

        # 위험 레벨별 기본 권고
        if risk_level == "HIGH":
            recommendations.append({
                'type': 'alert',
                'priority': 'high',
                'message': '높은 위험도 감지 - 즉시 점검 필요',
                'action': '전문가 점검 및 예방 조치 수행'
            })

        # 이상치 관련 권고
        if anomaly_count > 0:
            recommendations.append({
                'type': 'anomaly',
                'priority': 'medium' if anomaly_count < len(power_predictions) * 0.1 else 'high',
                'message': f'{anomaly_count}개 이상치 감지',
                'action': '해당 설비/시점 상세 점검 권장'
            })

        # 전력 소비 관련 권고
        power_mean = power_predictions.mean()
        high_power_count = risk_assessment['details']['high_power_samples']
        low_power_count = risk_assessment['details']['low_power_samples']

        if high_power_count > 0:
            recommendations.append({
                'type': 'power',
                'priority': 'medium',
                'message': f'{high_power_count}개 샘플에서 높은 전력 소비 예측',
                'action': '전력 사용 최적화 검토 필요'
            })

        if low_power_count > 0:
            recommendations.append({
                'type': 'power',
                'priority': 'low',
                'message': f'{low_power_count}개 샘플에서 낮은 전력 소비 예측',
                'action': '설비 가동률 점검 권장'
            })

        # 예측 불확실성 관련 권고
        uncertainty = risk_assessment['risk_factors']['prediction_uncertainty']
        if uncertainty > 0.3:
            recommendations.append({
                'type': 'uncertainty',
                'priority': 'low',
                'message': '높은 예측 불확실성 감지',
                'action': '추가 데이터 수집 또는 모델 재학습 고려'
            })

        # 정상 상태 메시지
        if not recommendations:
            recommendations.append({
                'type': 'normal',
                'priority': 'info',
                'message': '시스템 상태 정상',
                'action': '정기 모니터링 지속'
            })

        return recommendations

    def evaluate_integrated(
            self,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            y_anomaly_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """통합 시스템 평가"""

        # 예측 수행
        integrated_results = self.predict_integrated(X_test)

        # 전력예측 평가
        power_predictions = integrated_results['power_prediction']['predictions']
        power_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, power_predictions)),
            'mae': mean_absolute_error(y_test, power_predictions),
            'r2_score': r2_score(y_test, power_predictions),
            'mape': np.mean(np.abs((y_test - power_predictions) / (y_test + 1e-6))) * 100
        }

        # 이상탐지 평가 (라벨이 있는 경우)
        anomaly_metrics = {}
        if y_anomaly_test is not None:
            anomaly_predictions = integrated_results['anomaly_detection']['predictions']
            anomaly_metrics = {
                'f1_score': f1_score(y_anomaly_test, anomaly_predictions),
                'precision': precision_score(y_anomaly_test, anomaly_predictions, zero_division=0),
                'recall': recall_score(y_anomaly_test, anomaly_predictions, zero_division=0),
                'accuracy': (y_anomaly_test == anomaly_predictions).mean()
            }

        # 통합 성능 지표
        overall_performance = {
            'power_prediction': power_metrics,
            'anomaly_detection': anomaly_metrics,
            'risk_assessment_accuracy': self._evaluate_risk_assessment(integrated_results, y_test, y_anomaly_test),
            'system_latency': self._measure_prediction_latency(X_test.head(100)),  # 100개 샘플로 지연시간 측정
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }

        return overall_performance

    def _evaluate_risk_assessment(
            self,
            results: Dict[str, Any],
            y_test: pd.Series,
            y_anomaly_test: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """위험도 평가 정확도 측정"""
        risk_info = results['risk_assessment']

        # 실제 위험 상황 정의 (이상치 또는 극단적 전력 소비)
        actual_risk_indicators = []

        if y_anomaly_test is not None:
            actual_risk_indicators.append(y_anomaly_test)

        # 전력 기반 위험 지표 (극단값)
        power_threshold = y_test.quantile(0.95)
        power_risk = (y_test > power_threshold).astype(int)
        actual_risk_indicators.append(power_risk)

        if not actual_risk_indicators:
            return {'risk_correlation': 0.0}

        # 실제 위험도 (여러 지표 중 하나라도 위험하면 위험)
        actual_risk = np.logical_or.reduce(actual_risk_indicators).astype(float)
        predicted_risk = np.full(len(actual_risk), risk_info['overall_risk'])

        # 상관관계 계산
        correlation = np.corrcoef(actual_risk, predicted_risk)[0, 1]

        return {
            'risk_correlation': correlation if not np.isnan(correlation) else 0.0,
            'risk_level_accuracy': (risk_info['risk_level'] != 'NORMAL') == (actual_risk.mean() > 0.1)
        }

    def _measure_prediction_latency(self, X_sample: pd.DataFrame) -> Dict[str, float]:
        """예측 지연시간 측정"""
        try:
            start_time = time.time()
            _ = self.predict_integrated(X_sample)
            total_time = time.time() - start_time

            return {
                'total_latency_seconds': total_time,
                'per_sample_ms': (total_time / len(X_sample)) * 1000,
                'samples_per_second': len(X_sample) / total_time
            }
        except Exception as e:
            self.logger.warning(f"지연시간 측정 실패: {str(e)}")
            return {'total_latency_seconds': -1, 'per_sample_ms': -1, 'samples_per_second': -1}

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        return {
            'system_name': self.model_name,
            'is_trained': self.is_trained,
            'gpu_available': self.use_gpu,
            'device': str(self.device),
            'subsystems': {
                'anomaly_detector': {
                    'is_trained': self.anomaly_system.is_trained,
                    'models_count': len(getattr(self.anomaly_system, 'models', {}))
                },
                'power_predictor': {
                    'is_trained': self.power_system.is_trained,
                    'models_count': len(getattr(self.power_system, 'base_models', {}))
                }
            },
            'feature_count': len(self.feature_names),
            'target_name': self.target_name,
            'training_history': bool(self.training_history),
            'last_training': self.training_history.get('timestamp', 'Never') if self.training_history else 'Never'
        }

    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """피처 중요도 요약"""
        if not self.feature_importance:
            return {}

        # 모든 모델의 피처 중요도 통합
        combined_importance = {}

        for model_name, importance_dict in self.feature_importance.items():
            for feature, importance in importance_dict.items():
                if feature not in combined_importance:
                    combined_importance[feature] = []
                combined_importance[feature].append(importance)

        # 평균 중요도 계산
        avg_importance = {
            feature: np.mean(importances)
            for feature, importances in combined_importance.items()
        }

        # 상위 피처들
        top_features = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15])

        return {
            'top_features': top_features,
            'total_features': len(avg_importance),
            'model_count': len(self.feature_importance),
            'detailed_importance': self.feature_importance
        }

    def export_predictions(
            self,
            X: pd.DataFrame,
            output_format: str = 'dict'
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """예측 결과 내보내기"""

        results = self.predict_integrated(X)

        if output_format == 'dataframe':
            # DataFrame 형태로 변환
            df_results = pd.DataFrame(index=X.index)

            # 이상탐지 결과
            df_results['is_anomaly'] = results['anomaly_detection']['predictions']
            df_results['anomaly_ratio'] = results['anomaly_detection']['anomaly_ratio']

            # 전력예측 결과
            df_results['predicted_power'] = results['power_prediction']['predictions']
            df_results['power_mean'] = results['power_prediction']['mean_prediction']
            df_results['power_std'] = results['power_prediction']['std_prediction']

            # 위험도 평가
            df_results['risk_score'] = results['risk_assessment']['overall_risk']
            df_results['risk_level'] = results['risk_assessment']['risk_level']

            # 시스템 정보
            df_results['prediction_timestamp'] = results['system_info']['timestamp']

            return df_results

        return results

    def save_system(self, filepath: str) -> None:
        """통합 시스템 저장"""
        try:
            # 하위 시스템들 개별 저장
            anomaly_path = filepath.replace('.pkl', '_anomaly.pkl')
            power_path = filepath.replace('.pkl', '_power.pkl')

            self.anomaly_system.save_model(anomaly_path)
            self.power_system.save_model(power_path)

            # 메인 시스템 정보 저장
            system_data = {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'training_history': self.training_history,
                'feature_importance': self.feature_importance,
                'use_gpu': self.use_gpu,
                'device': str(self.device),
                'is_trained': self.is_trained,
                'subsystem_paths': {
                    'anomaly': anomaly_path,
                    'power': power_path
                }
            }

            import joblib
            joblib.dump(system_data, filepath)

            self.logger.info(f"통합 시스템 저장 완료: {filepath}")

        except Exception as e:
            self.logger.error(f"시스템 저장 실패: {str(e)}")
            raise

    def load_system(self, filepath: str) -> None:
        """통합 시스템 로드"""
        try:
            import joblib
            system_data = joblib.load(filepath)

            # 메인 시스템 정보 복원
            self.model_name = system_data['model_name']
            self.model_type = system_data['model_type']
            self.feature_names = system_data['feature_names']
            self.target_name = system_data['target_name']
            self.training_history = system_data['training_history']
            self.feature_importance = system_data['feature_importance']
            self.use_gpu = system_data['use_gpu']
            self.device = torch.device(system_data['device'])
            self.is_trained = system_data['is_trained']

            # 하위 시스템들 로드
            subsystem_paths = system_data['subsystem_paths']

            self.anomaly_system.load_model(subsystem_paths['anomaly'])
            self.power_system.load_model(subsystem_paths['power'])

            self.logger.info(f"통합 시스템 로드 완료: {filepath}")

        except Exception as e:
            self.logger.error(f"시스템 로드 실패: {str(e)}")
            raise


class BatchPredictionProcessor:
    """배치 예측 처리기"""

    def __init__(self, integrated_system: IntegratedPredictionSystem):
        self.system = integrated_system
        self.logger = get_logger("batch_processor")

    def process_batch(
            self,
            X: pd.DataFrame,
            batch_size: int = 1000,
            progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """대용량 데이터 배치 처리"""

        if not self.system.is_trained:
            raise ModelPredictionError("시스템이 학습되지 않았습니다")

        total_batches = (len(X) + batch_size - 1) // batch_size
        results_list = []

        self.logger.info(f"배치 처리 시작: {len(X):,}개 샘플, {total_batches}개 배치")

        for i in range(0, len(X), batch_size):
            batch_start = i
            batch_end = min(i + batch_size, len(X))
            batch_data = X.iloc[batch_start:batch_end]

            try:
                # 배치 예측
                batch_results = self.system.export_predictions(batch_data, 'dataframe')
                results_list.append(batch_results)

                # 진행률 콜백
                if progress_callback:
                    progress = (i // batch_size + 1) / total_batches
                    progress_callback(progress, f"처리 중: {batch_end:,}/{len(X):,}")

                if (i // batch_size + 1) % 10 == 0:
                    self.logger.info(f"배치 진행률: {batch_end:,}/{len(X):,} ({(batch_end / len(X) * 100):.1f}%)")

            except Exception as e:
                self.logger.error(f"배치 {i // batch_size + 1} 처리 실패: {str(e)}")
                # 오류 발생 시 빈 결과로 대체
                empty_batch = pd.DataFrame(index=batch_data.index)
                for col in ['is_anomaly', 'predicted_power', 'risk_score', 'risk_level']:
                    empty_batch[col] = np.nan
                results_list.append(empty_batch)

        # 결과 병합
        final_results = pd.concat(results_list, ignore_index=False)

        self.logger.info(f"배치 처리 완료: {len(final_results):,}개 결과")
        return final_results


# 팩토리 함수들
def create_integrated_system(device=None) -> IntegratedPredictionSystem:
    """통합 예측 시스템 생성"""
    return IntegratedPredictionSystem()


def create_batch_processor(integrated_system: IntegratedPredictionSystem) -> BatchPredictionProcessor:
    """배치 처리기 생성"""
    return BatchPredictionProcessor(integrated_system)


def run_complete_pipeline(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        validation_split: float = 0.2,
        sample_for_speed: bool = True
) -> Dict[str, Any]:
    """완전한 파이프라인 실행"""

    logger = get_logger("complete_pipeline")
    logger.info("완전한 통합 파이프라인 실행 시작")

    # 검증 데이터 분할
    if validation_split > 0:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
    else:
        X_train_split, X_val, y_train_split, y_val = X_train, None, y_train, None

    # 통합 시스템 생성 및 학습
    system = create_integrated_system()

    with timer("complete_pipeline_training"):
        training_result = system.train(
            X_train_split, y_train_split,
            X_val, y_val,
            sample_for_speed=sample_for_speed
        )

    # 평가 수행
    with timer("complete_pipeline_evaluation"):
        evaluation_result = system.evaluate_integrated(X_test, y_test)

    # 피처 중요도 요약
    feature_importance = system.get_feature_importance_summary()

    # 시스템 상태
    system_status = system.get_system_status()

    logger.info("완전한 통합 파이프라인 실행 완료")

    return {
        'system': system,
        'training_result': training_result,
        'evaluation_result': evaluation_result,
        'feature_importance': feature_importance,
        'system_status': system_status,
        'pipeline_info': {
            'training_samples': len(X_train_split),
            'validation_samples': len(X_val) if X_val is not None else 0,
            'test_samples': len(X_test),
            'validation_split': validation_split,
            'sampling_used': sample_for_speed
        }
    }