"""
스마트팩토리 에너지 관리 시스템 전력 예측 모델
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

warnings.filterwarnings('ignore')

from .base_model import RegressionModel, EnsembleModel
from ..core.config import get_config
from ..core.logger import get_logger, log_performance, timer
from ..core.exceptions import ModelTrainingError, ModelPredictionError


class LSTMPowerPredictor(RegressionModel):
    """LSTM 기반 전력 예측 모델"""

    def __init__(self, model_name: str = "lstm_power_predictor"):
        super().__init__(model_name, "lstm_regression")
        self.sequence_length = self.config.model.lstm_sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model = None
        self.scaler_X = None
        self.scaler_y = None

        self.logger.info(f"LSTM 모델 초기화 - 디바이스: {self.device}")

    def _create_model(self) -> nn.Module:
        """LSTM 모델 생성"""

        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
                super(LSTMNet, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=dropout
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                # LSTM 출력
                lstm_out, _ = self.lstm(x)

                # 마지막 시퀀스의 출력만 사용
                last_output = lstm_out[:, -1, :]

                # 드롭아웃 및 완전연결층
                out = self.dropout(last_output)
                out = self.fc(out)
                return out

        return LSTMNet

    def _prepare_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 데이터 준비"""
        X, y = [], []

        for i in range(len(data) - self.sequence_length + 1):
            # 시퀀스 데이터 (피처들)
            sequence = data.iloc[i:i + self.sequence_length].drop(columns=[target_col]).values
            # 타겟 (마지막 시점의 전력 소비)
            target = data.iloc[i + self.sequence_length - 1][target_col]

            X.append(sequence)
            y.append(target)

        return np.array(X), np.array(y)

    @log_performance("lstm_training")
    def _train_model(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """LSTM 모델 학습"""
        from sklearn.preprocessing import StandardScaler

        training_start = pd.Timestamp.now()

        # 데이터 준비 - X와 y를 결합하여 시퀀스 생성
        data_combined = X.copy()
        data_combined[y.name] = y

        # 시퀀스 데이터 생성
        X_seq, y_seq = self._prepare_sequences(data_combined, y.name)

        if len(X_seq) == 0:
            raise ModelTrainingError("시퀀스 데이터가 충분하지 않습니다")

        # 데이터 스케일링
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        # X 스케일링 (각 시퀀스의 피처들을 스케일링)
        n_samples, seq_len, n_features = X_seq.shape
        X_seq_reshaped = X_seq.reshape(-1, n_features)
        X_seq_scaled = self.scaler_X.fit_transform(X_seq_reshaped)
        X_seq_scaled = X_seq_scaled.reshape(n_samples, seq_len, n_features)

        # y 스케일링
        y_seq_scaled = self.scaler_y.fit_transform(y_seq.reshape(-1, 1)).flatten()

        # PyTorch 텐서 변환
        X_tensor = torch.FloatTensor(X_seq_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_seq_scaled).to(self.device)

        # 데이터 로더 생성
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.model.lstm_batch_size,
            shuffle=True
        )

        # 모델 생성
        model_class = self._create_model()
        self.lstm_model = model_class(
            input_size=n_features,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        ).to(self.device)

        # 손실 함수 및 옵티마이저
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        # 학습
        train_losses = []
        val_losses = []

        for epoch in range(self.config.model.lstm_epochs):
            # 학습 모드
            self.lstm_model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_train_loss)

            # 검증 (있는 경우)
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate_lstm(X_val, y_val, criterion)
                val_losses.append(val_loss)
                scheduler.step(val_loss)
            else:
                scheduler.step(avg_train_loss)

            if epoch % 20 == 0:
                self.logger.debug(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}")

        training_time = (pd.Timestamp.now() - training_start).total_seconds()

        # 학습 결과
        training_result = {
            'epochs': self.config.model.lstm_epochs,
            'final_train_loss': train_losses[-1],
            'train_losses': train_losses,
            'sequence_length': self.sequence_length,
            'device': str(self.device),
            'training_samples': len(X_seq),
            'training_time': training_time
        }

        if val_losses:
            training_result.update({
                'final_val_loss': val_losses[-1],
                'val_losses': val_losses
            })

        self.model = self.lstm_model  # base_model 호환성

        return training_result

    def _evaluate_lstm(self, X_val: pd.DataFrame, y_val: pd.Series, criterion) -> float:
        """LSTM 모델 검증"""
        # 검증 데이터 시퀀스 생성
        data_val = X_val.copy()
        data_val[y_val.name] = y_val
        X_val_seq, y_val_seq = self._prepare_sequences(data_val, y_val.name)

        if len(X_val_seq) == 0:
            return float('inf')

        # 스케일링
        n_samples, seq_len, n_features = X_val_seq.shape
        X_val_reshaped = X_val_seq.reshape(-1, n_features)
        X_val_scaled = self.scaler_X.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(n_samples, seq_len, n_features)
        y_val_scaled = self.scaler_y.transform(y_val_seq.reshape(-1, 1)).flatten()

        # 텐서 변환
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)

        # 평가 모드
        self.lstm_model.eval()
        with torch.no_grad():
            val_outputs = self.lstm_model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor).item()

        return val_loss

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """LSTM 예측"""
        if self.lstm_model is None:
            raise ModelPredictionError("LSTM 모델이 학습되지 않았습니다")

        # 임시 타겟 컬럼 추가 (시퀀스 생성용)
        X_temp = X.copy()
        X_temp['temp_target'] = 0  # 더미 값

        # 시퀀스 생성
        X_seq, _ = self._prepare_sequences(X_temp, 'temp_target')

        if len(X_seq) == 0:
            # 시퀀스가 생성되지 않는 경우 마지막 행 반복
            X_seq = np.array([X.iloc[-self.sequence_length:].values])

        # 스케일링
        n_samples, seq_len, n_features = X_seq.shape
        X_seq_reshaped = X_seq.reshape(-1, n_features)
        X_seq_scaled = self.scaler_X.transform(X_seq_reshaped)
        X_seq_scaled = X_seq_scaled.reshape(n_samples, seq_len, n_features)

        # 텐서 변환
        X_tensor = torch.FloatTensor(X_seq_scaled).to(self.device)

        # 예측
        self.lstm_model.eval()
        with torch.no_grad():
            predictions_scaled = self.lstm_model(X_tensor).squeeze()
            predictions = self.scaler_y.inverse_transform(
                predictions_scaled.cpu().numpy().reshape(-1, 1)
            ).flatten()

        return predictions


class XGBoostPowerPredictor(RegressionModel):
    """XGBoost 기반 전력 예측 모델"""

    def __init__(self, model_name: str = "xgboost_power_predictor"):
        super().__init__(model_name, "xgboost_regression")

    def _create_model(self) -> xgb.XGBRegressor:
        """XGBoost 모델 생성"""
        return xgb.XGBRegressor(
            n_estimators=self.config.model.xgb_n_estimators,
            learning_rate=self.config.model.xgb_learning_rate,
            max_depth=self.config.model.xgb_max_depth,
            random_state=42,
            n_jobs=-1
        )

    @log_performance("xgboost_training")
    def _train_model(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """XGBoost 모델 학습"""
        training_start = pd.Timestamp.now()

        # 검증 데이터 설정
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]

        # 모델 학습
        self.model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric='rmse',
            early_stopping_rounds=20,
            verbose=False
        )

        training_time = (pd.Timestamp.now() - training_start).total_seconds()

        # 피처 중요도 가져오기
        feature_importance = dict(zip(X.columns, self.model.feature_importances_))

        training_result = {
            'best_iteration': getattr(self.model, 'best_iteration', self.config.model.xgb_n_estimators),
            'feature_importance': feature_importance,
            'training_samples': len(X),
            'training_time': training_time,
            'config': {
                'n_estimators': self.config.model.xgb_n_estimators,
                'learning_rate': self.config.model.xgb_learning_rate,
                'max_depth': self.config.model.xgb_max_depth
            }
        }

        return training_result

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """XGBoost 예측"""
        return self.model.predict(X)


class RandomForestPowerPredictor(RegressionModel):
    """Random Forest 기반 전력 예측 모델"""

    def __init__(self, model_name: str = "rf_power_predictor"):
        super().__init__(model_name, "random_forest_regression")

    def _create_model(self) -> RandomForestRegressor:
        """Random Forest 모델 생성"""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )

    @log_performance("rf_training")
    def _train_model(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            **kwargs
    ) -> Dict[str, Any]:
        """Random Forest 모델 학습"""
        training_start = pd.Timestamp.now()

        # 모델 학습
        self.model.fit(X, y)

        training_time = (pd.Timestamp.now() - training_start).total_seconds()

        # 피처 중요도
        feature_importance = dict(zip(X.columns, self.model.feature_importances_))

        return {
            'feature_importance': feature_importance,
            'training_samples': len(X),
            'training_time': training_time,
            'oob_score': getattr(self.model, 'oob_score_', None)
        }

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Random Forest 예측"""
        return self.model.predict(X)


class MLPPowerPredictor(RegressionModel):
    """MLP 기반 전력 예측 모델"""

    def __init__(self, model_name: str = "mlp_power_predictor"):
        super().__init__(model_name, "mlp_regression")

    def _create_model(self) -> MLPRegressor:
        """MLP 모델 생성"""
        return MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )

    @log_performance("mlp_training")
    def _train_model(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            **kwargs
    ) -> Dict[str, Any]:
        """MLP 모델 학습"""
        training_start = pd.Timestamp.now()

        # 데이터 스케일링
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 모델 학습
        self.model.fit(X_scaled, y)

        training_time = (pd.Timestamp.now() - training_start).total_seconds()

        return {
            'n_iter': self.model.n_iter_,
            'loss': self.model.loss_,
            'training_samples': len(X),
            'training_time': training_time
        }

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """MLP 예측"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class PowerPredictionEnsemble(EnsembleModel):
    """전력 예측 앙상블 모델"""

    def __init__(self, model_name: str = "power_prediction_ensemble", device=None):
        super().__init__(model_name, "power_prediction_ensemble")
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._sample_data = None

        # 기본 모델들 추가
        self.add_base_model("xgboost", XGBoostPowerPredictor())
        self.add_base_model("random_forest", RandomForestPowerPredictor())
        self.add_base_model("mlp", MLPPowerPredictor())

        # GPU가 사용 가능한 경우에만 LSTM 추가
        if torch.cuda.is_available():
            self.add_base_model("lstm", LSTMPowerPredictor())
            self.logger.info("GPU 사용 가능 - LSTM 모델 포함")
        else:
            self.logger.info("GPU 사용 불가 - LSTM 모델 제외")

    def train_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """모든 기본 모델 학습"""
        return self.train(X_train, y_train, X_val, y_val)

    def predict_ensemble(self, X: pd.DataFrame) -> Dict[str, Any]:
        """앙상블 예측 및 개별 모델 예측 반환"""
        individual_predictions = self.get_base_model_predictions(X)
        ensemble_prediction = self.predict(X)

        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': individual_predictions
        }

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """각 모델의 피처 중요도 반환"""
        importance_dict = {}

        for name, model in self.base_models.items():
            if hasattr(model, 'get_feature_importance') and model.is_trained:
                importance = model.get_feature_importance()
                if importance:
                    importance_dict[name] = importance

        return importance_dict

    def plot_training_history(self) -> None:
        """학습 히스토리 플롯"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            plot_idx = 0
            for name, model in self.base_models.items():
                if plot_idx >= 4:
                    break

                if hasattr(model, 'training_history') and model.training_history:
                    history = model.training_history

                    if name == "lstm" and 'train_losses' in history:
                        # LSTM 손실 그래프
                        axes[plot_idx].plot(history['train_losses'], label='Train Loss')
                        if 'val_losses' in history:
                            axes[plot_idx].plot(history['val_losses'], label='Val Loss')
                        axes[plot_idx].set_title(f'{name.upper()} Training Loss')
                        axes[plot_idx].set_xlabel('Epoch')
                        axes[plot_idx].set_ylabel('Loss')
                        axes[plot_idx].legend()

                    elif 'feature_importance' in history:
                        # 피처 중요도 그래프
                        importance = history['feature_importance']
                        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

                        axes[plot_idx].barh(list(top_features.keys()), list(top_features.values()))
                        axes[plot_idx].set_title(f'{name.upper()} Feature Importance (Top 10)')
                        axes[plot_idx].set_xlabel('Importance')

                    plot_idx += 1

            # 남은 subplot 숨기기
            for i in range(plot_idx, 4):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.show()

        except ImportError:
            self.logger.warning("matplotlib을 사용할 수 없어 그래프를 표시할 수 없습니다")
        except Exception as e:
            self.logger.error(f"그래프 표시 중 오류: {str(e)}")


# 편의 함수들
def create_lstm_predictor() -> LSTMPowerPredictor:
    """LSTM 예측 모델 생성"""
    return LSTMPowerPredictor()


def create_xgboost_predictor() -> XGBoostPowerPredictor:
    """XGBoost 예측 모델 생성"""
    return XGBoostPowerPredictor()


def create_rf_predictor() -> RandomForestPowerPredictor:
    """Random Forest 예측 모델 생성"""
    return RandomForestPowerPredictor()


def create_mlp_predictor() -> MLPPowerPredictor:
    """MLP 예측 모델 생성"""
    return MLPPowerPredictor()


def create_ensemble_predictor(device=None) -> PowerPredictionEnsemble:
    """앙상블 예측 모델 생성"""
    return PowerPredictionEnsemble(device=device)