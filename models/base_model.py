"""
스마트팩토리 에너지 관리 시스템 기본 모델 클래스
"""
import os
import abc
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json

from ..core.config import get_config
from ..core.logger import get_logger, log_performance
from ..core.exceptions import (
    ModelException, ModelLoadError, ModelSaveError,
    ModelTrainingError, ModelPredictionError, ModelValidationError
)


class BaseModel(abc.ABC):
    """기본 모델 추상 클래스"""

    def __init__(self, model_name: str, model_type: str = "base"):
        self.model_name = model_name
        self.model_type = model_type
        self.config = get_config()
        self.logger = get_logger(f"model_{model_name}")

        # 모델 상태
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.feature_names = []
        self.target_name = ""

        # 메타데이터
        self.metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'framework': 'scikit-learn',  # 기본값
            'training_config': {},
            'performance_metrics': {}
        }

        self.logger.info(f"{model_type} 모델 '{model_name}' 초기화 완료")

    @abc.abstractmethod
    def _create_model(self) -> Any:
        """모델 생성 (하위 클래스에서 구현)"""
        pass

    @abc.abstractmethod
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """모델 학습 (하위 클래스에서 구현)"""
        pass

    @abc.abstractmethod
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """모델 예측 (하위 클래스에서 구현)"""
        pass

    @log_performance("model_training")
    def train(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """모델 학습"""
        self.logger.info(f"모델 학습 시작: {self.model_name}")

        try:
            # 입력 검증
            self._validate_training_data(X_train, y_train)

            # 피처 이름 저장
            self.feature_names = list(X_train.columns)
            self.target_name = y_train.name or "target"

            # 모델 생성
            if self.model is None:
                self.model = self._create_model()

            # 학습 수행
            training_result = self._train_model(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                **kwargs
            )

            # 학습 완료 처리
            self.is_trained = True
            self.training_history = training_result
            self._update_metadata(X_train, y_train, training_result)

            self.logger.info(f"모델 학습 완료: {self.model_name}")
            return training_result

        except Exception as e:
            self.logger.error(f"모델 학습 실패: {str(e)}")
            raise ModelTrainingError(
                f"모델 '{self.model_name}' 학습 중 오류 발생",
                details={'error': str(e), 'model_type': self.model_type}
            ) from e

    @log_performance("model_prediction")
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        try:
            # 모델 학습 상태 확인
            if not self.is_trained or self.model is None:
                raise ModelPredictionError(
                    f"모델 '{self.model_name}'이 학습되지 않았습니다"
                )

            # 입력 검증
            self._validate_prediction_data(X)

            # 예측 수행
            predictions = self._predict_model(X)

            self.logger.debug(f"예측 완료: {len(predictions)}개 샘플")
            return predictions

        except Exception as e:
            if isinstance(e, ModelPredictionError):
                raise

            self.logger.error(f"예측 실패: {str(e)}")
            raise ModelPredictionError(
                f"모델 '{self.model_name}' 예측 중 오류 발생",
                details={'error': str(e), 'input_shape': X.shape}
            ) from e

    def evaluate(
            self,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """모델 평가"""
        self.logger.info(f"모델 평가 시작: {self.model_name}")

        try:
            # 예측 수행
            y_pred = self.predict(X_test)

            # 평가 메트릭 계산
            evaluation_metrics = self._calculate_metrics(y_test, y_pred, metrics)

            # 메타데이터 업데이트
            self.metadata['performance_metrics'].update(evaluation_metrics)

            self.logger.info(f"모델 평가 완료: {self.model_name}")
            return evaluation_metrics

        except Exception as e:
            self.logger.error(f"모델 평가 실패: {str(e)}")
            raise ModelValidationError(
                f"모델 '{self.model_name}' 평가 중 오류 발생",
                details={'error': str(e)}
            ) from e

    def save_model(self, file_path: str) -> None:
        """모델 저장"""
        self.logger.info(f"모델 저장 시작: {file_path}")

        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 저장할 데이터 준비
            model_data = {
                'model': self.model,
                'metadata': self.metadata,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'model_name': self.model_name,
                'model_type': self.model_type
            }

            # 모델 저장
            joblib.dump(model_data, file_path)

            # 메타데이터 별도 저장
            metadata_path = file_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

            self.logger.info(f"모델 저장 완료: {file_path}")

        except Exception as e:
            self.logger.error(f"모델 저장 실패: {str(e)}")
            raise ModelSaveError(
                f"모델 '{self.model_name}' 저장 중 오류 발생",
                details={'file_path': file_path, 'error': str(e)}
            ) from e

    def load_model(self, file_path: str) -> None:
        """모델 로드"""
        self.logger.info(f"모델 로드 시작: {file_path}")

        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                raise ModelLoadError(f"모델 파일을 찾을 수 없습니다: {file_path}")

            # 모델 로드
            model_data = joblib.load(file_path)

            # 데이터 복원
            self.model = model_data['model']
            self.metadata = model_data['metadata']
            self.feature_names = model_data['feature_names']
            self.target_name = model_data['target_name']
            self.is_trained = model_data['is_trained']
            self.training_history = model_data['training_history']

            # 이름 확인
            if 'model_name' in model_data:
                self.model_name = model_data['model_name']
            if 'model_type' in model_data:
                self.model_type = model_data['model_type']

            self.logger.info(f"모델 로드 완료: {self.model_name}")

        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise

            self.logger.error(f"모델 로드 실패: {str(e)}")
            raise ModelLoadError(
                f"모델 로드 중 오류 발생",
                details={'file_path': file_path, 'error': str(e)}
            ) from e

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """피처 중요도 반환"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None

        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'target_name': self.target_name,
            'metadata': self.metadata.copy()
        }

    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """학습 데이터 검증"""
        if X.empty:
            raise ModelValidationError("학습 데이터가 비어있습니다")

        if len(X) != len(y):
            raise ModelValidationError(
                f"피처와 타겟의 길이가 다릅니다: {len(X)} vs {len(y)}"
            )

        if X.isnull().any().any():
            null_columns = X.columns[X.isnull().any()].tolist()
            raise ModelValidationError(
                f"학습 데이터에 결측치가 있습니다: {null_columns}"
            )

        if y.isnull().any():
            raise ModelValidationError("타겟 데이터에 결측치가 있습니다")

    def _validate_prediction_data(self, X: pd.DataFrame) -> None:
        """예측 데이터 검증"""
        if X.empty:
            raise ModelValidationError("예측 데이터가 비어있습니다")

        # 피처 이름 확인
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ModelValidationError(
                    f"필요한 피처가 누락되었습니다: {missing_features}"
                )

            # 피처 순서 맞추기
            X = X[self.feature_names]

        if X.isnull().any().any():
            null_columns = X.columns[X.isnull().any()].tolist()
            raise ModelValidationError(
                f"예측 데이터에 결측치가 있습니다: {null_columns}"
            )

    def _calculate_metrics(
            self,
            y_true: pd.Series,
            y_pred: np.ndarray,
            metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """평가 메트릭 계산"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # 기본 메트릭
        default_metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred)
        }

        # MAPE 계산 (0으로 나누기 방지)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            default_metrics['mape'] = mape

        # 사용자 지정 메트릭이 있으면 필터링
        if metrics:
            return {k: v for k, v in default_metrics.items() if k in metrics}

        return default_metrics

    def _update_metadata(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            training_result: Dict[str, Any]
    ) -> None:
        """메타데이터 업데이트"""
        self.metadata.update({
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'feature_count': len(X_train.columns),
            'target_name': self.target_name,
            'feature_names': self.feature_names,
            'training_config': training_result.get('config', {}),
            'training_time_seconds': training_result.get('training_time', 0)
        })


class RegressionModel(BaseModel):
    """회귀 모델 기본 클래스"""

    def __init__(self, model_name: str, model_type: str = "regression"):
        super().__init__(model_name, model_type)

    def _calculate_metrics(
            self,
            y_true: pd.Series,
            y_pred: np.ndarray,
            metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """회귀 모델용 메트릭 계산"""
        metrics_dict = super()._calculate_metrics(y_true, y_pred, metrics)

        # 회귀 전용 메트릭 추가
        from sklearn.metrics import explained_variance_score, max_error

        additional_metrics = {
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred)
        }

        metrics_dict.update(additional_metrics)
        return metrics_dict


class ClassificationModel(BaseModel):
    """분류 모델 기본 클래스"""

    def __init__(self, model_name: str, model_type: str = "classification"):
        super().__init__(model_name, model_type)

    def _calculate_metrics(
            self,
            y_true: pd.Series,
            y_pred: np.ndarray,
            metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """분류 모델용 메트릭 계산"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        metrics_dict = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        # ROC-AUC (이진 분류 또는 확률 예측이 가능한 경우)
        try:
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(y_true.index)
                if y_proba.shape[1] == 2:  # 이진 분류
                    metrics_dict['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # 다중 클래스
                    metrics_dict['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except:
            pass  # ROC-AUC 계산 실패 시 무시

        # 사용자 지정 메트릭이 있으면 필터링
        if metrics:
            return {k: v for k, v in metrics_dict.items() if k in metrics}

        return metrics_dict

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """확률 예측"""
        if not self.is_trained or self.model is None:
            raise ModelPredictionError(
                f"모델 '{self.model_name}'이 학습되지 않았습니다"
            )

        if not hasattr(self.model, 'predict_proba'):
            raise ModelPredictionError(
                f"모델 '{self.model_name}'은 확률 예측을 지원하지 않습니다"
            )

        try:
            self._validate_prediction_data(X)
            probabilities = self.model.predict_proba(X)
            return probabilities
        except Exception as e:
            raise ModelPredictionError(
                f"확률 예측 중 오류 발생: {str(e)}"
            ) from e


class EnsembleModel(BaseModel):
    """앙상블 모델 기본 클래스"""

    def __init__(self, model_name: str, model_type: str = "ensemble"):
        super().__init__(model_name, model_type)
        self.base_models = {}
        self.ensemble_method = "averaging"  # averaging, voting, stacking

    def add_base_model(self, name: str, model: BaseModel) -> None:
        """기본 모델 추가"""
        self.base_models[name] = model
        self.logger.info(f"기본 모델 추가: {name}")

    def _train_model(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """앙상블 모델 학습"""
        training_results = {}

        for name, model in self.base_models.items():
            self.logger.info(f"기본 모델 학습 시작: {name}")

            try:
                result = model.train(X, y, X_val, y_val, **kwargs)
                training_results[name] = result
                self.logger.info(f"기본 모델 학습 완료: {name}")
            except Exception as e:
                self.logger.error(f"기본 모델 학습 실패: {name} - {str(e)}")
                raise ModelTrainingError(
                    f"앙상블 기본 모델 '{name}' 학습 실패",
                    details={'error': str(e)}
                ) from e

        return {
            'ensemble_method': self.ensemble_method,
            'base_models_results': training_results,
            'training_time': sum(r.get('training_time', 0) for r in training_results.values())
        }

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """앙상블 예측"""
        if not self.base_models:
            raise ModelPredictionError("기본 모델이 없습니다")

        predictions = []
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"기본 모델 '{name}' 예측 실패: {str(e)}")
                continue

        if not predictions:
            raise ModelPredictionError("모든 기본 모델의 예측이 실패했습니다")

        # 앙상블 방법에 따른 결합
        predictions_array = np.array(predictions)

        if self.ensemble_method == "averaging":
            return np.mean(predictions_array, axis=0)
        elif self.ensemble_method == "median":
            return np.median(predictions_array, axis=0)
        elif self.ensemble_method == "voting":
            # 분류용 - 다수결 투표
            from scipy.stats import mode
            return mode(predictions_array, axis=0)[0].flatten()
        else:
            return np.mean(predictions_array, axis=0)  # 기본값

    def _create_model(self) -> None:
        """앙상블은 별도 모델 생성 불필요"""
        return None

    def get_base_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """각 기본 모델의 예측 결과 반환"""
        predictions = {}
        for name, model in self.base_models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                self.logger.warning(f"기본 모델 '{name}' 예측 실패: {str(e)}")
        return predictions


class ModelManager:
    """모델 관리자"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.logger = get_logger("model_manager")
        self.registered_models = {}

        # 모델 디렉토리 생성
        os.makedirs(models_dir, exist_ok=True)

    def register_model(self, model: BaseModel) -> None:
        """모델 등록"""
        self.registered_models[model.model_name] = model
        self.logger.info(f"모델 등록: {model.model_name}")

    def get_model(self, model_name: str) -> BaseModel:
        """등록된 모델 반환"""
        if model_name not in self.registered_models:
            raise ModelException(f"등록되지 않은 모델: {model_name}")
        return self.registered_models[model_name]

    def list_models(self) -> List[str]:
        """등록된 모델 목록 반환"""
        return list(self.registered_models.keys())

    def save_all_models(self) -> None:
        """모든 등록된 모델 저장"""
        for name, model in self.registered_models.items():
            if model.is_trained:
                file_path = os.path.join(self.models_dir, f"{name}.pkl")
                model.save_model(file_path)
                self.logger.info(f"모델 저장: {name}")

    def load_model_from_file(self, model_name: str, file_path: str, model_class: type) -> BaseModel:
        """파일에서 모델 로드"""
        model = model_class(model_name)
        model.load_model(file_path)
        self.register_model(model)
        return model

    def get_model_summary(self) -> Dict[str, Dict[str, Any]]:
        """모든 모델의 요약 정보 반환"""
        summary = {}
        for name, model in self.registered_models.items():
            summary[name] = model.get_model_info()
        return summary

    def compare_models(
            self,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """모델 성능 비교"""
        if model_names is None:
            model_names = self.list_models()

        comparison_results = []

        for name in model_names:
            if name not in self.registered_models:
                self.logger.warning(f"모델을 찾을 수 없습니다: {name}")
                continue

            model = self.registered_models[name]
            if not model.is_trained:
                self.logger.warning(f"모델이 학습되지 않았습니다: {name}")
                continue

            try:
                metrics = model.evaluate(X_test, y_test)
                metrics['model_name'] = name
                metrics['model_type'] = model.model_type
                comparison_results.append(metrics)
            except Exception as e:
                self.logger.error(f"모델 평가 실패: {name} - {str(e)}")

        if comparison_results:
            return pd.DataFrame(comparison_results)
        else:
            return pd.DataFrame()


# 전역 모델 매니저
_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """전역 모델 매니저 반환"""
    return _model_manager


def register_model(model: BaseModel) -> None:
    """모델 등록 (편의 함수)"""
    _model_manager.register_model(model)


def get_model(model_name: str) -> BaseModel:
    """모델 조회 (편의 함수)"""
    return _model_manager.get_model(model_name)