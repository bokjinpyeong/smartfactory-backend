"""
스마트팩토리 에너지 관리 시스템 설정 모듈
"""
import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """AI 모델 설정"""
    # 이상탐지 모델 설정
    anomaly_contamination: float = 0.05
    anomaly_n_estimators: int = 100
    anomaly_max_samples: int = 1000

    # 예측 모델 설정
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_sequence_length: int = 24

    # XGBoost 설정
    xgb_n_estimators: int = 100
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6


@dataclass
class DataConfig:
    """데이터 처리 설정"""
    # 센서 데이터
    sensor_data_path: str = "data/sensor/"
    processed_data_path: str = "data/processed/"
    model_data_path: str = "data/models/"

    # 데이터 처리
    batch_size: int = 50000
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 스케일링
    scaling_method: str = "standard"  # standard, minmax

    # 데이터 검증
    max_missing_ratio: float = 0.1
    outlier_threshold: float = 3.0


@dataclass
class SystemConfig:
    """시스템 설정"""
    # 실시간 처리
    realtime_batch_size: int = 100
    processing_interval: int = 60  # seconds
    alert_threshold: float = 0.7

    # 데이터베이스
    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "smartfactory_energy"
    db_user: str = "root"
    db_password: str = ""

    # MQTT 설정
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "factory/energy"

    # API 설정
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4


@dataclass
class PowerConfig:
    """전력 관리 설정"""
    # TOU 요금제 (시간대별 요금)
    tou_peak_hours: List[int] = None
    tou_off_peak_hours: List[int] = None
    tou_peak_rate: float = 1.5
    tou_off_peak_rate: float = 0.8
    tou_normal_rate: float = 1.0

    # 피크 전력 제약
    peak_power_limit: float = 1000.0  # kW
    safety_margin: float = 0.1  # 10% 안전 여유

    # 스케줄링
    scheduling_horizon: int = 24  # hours
    min_processing_time: int = 10  # minutes
    max_processing_time: int = 480  # minutes

    def __post_init__(self):
        if self.tou_peak_hours is None:
            self.tou_peak_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17]
        if self.tou_off_peak_hours is None:
            self.tou_off_peak_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]


class ConfigManager:
    """설정 관리자"""

    def __init__(self, config_path: Optional[str] = None, env: str = "development"):
        self.env = env
        self.config_path = config_path or self._get_default_config_path()

        # 기본 설정 초기화
        self.model = ModelConfig()
        self.data = DataConfig()
        self.system = SystemConfig()
        self.power = PowerConfig()

        # 환경별 설정 로드
        self._load_config()

    def _get_default_config_path(self) -> str:
        """기본 설정 파일 경로 반환"""
        base_dir = Path(__file__).parent.parent
        return str(base_dir / "config" / f"{self.env}.yaml")

    def _load_config(self):
        """설정 파일 로드"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                self._update_from_dict(config_data)
                print(f"✅ 설정 파일 로드 완료: {self.config_path}")

            except Exception as e:
                print(f"⚠️ 설정 파일 로드 실패: {e}")
                print("기본 설정을 사용합니다.")
        else:
            print(f"⚠️ 설정 파일이 없습니다: {self.config_path}")
            print("기본 설정을 사용합니다.")

    def _update_from_dict(self, config_data: Dict):
        """딕셔너리에서 설정 업데이트"""
        if 'model' in config_data:
            self._update_dataclass(self.model, config_data['model'])

        if 'data' in config_data:
            self._update_dataclass(self.data, config_data['data'])

        if 'system' in config_data:
            self._update_dataclass(self.system, config_data['system'])

        if 'power' in config_data:
            self._update_dataclass(self.power, config_data['power'])

    def _update_dataclass(self, obj, data: Dict):
        """데이터클래스 객체 업데이트"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    def get_db_url(self) -> str:
        """데이터베이스 URL 생성"""
        return (f"mysql+pymysql://{self.system.db_user}:{self.system.db_password}"
                f"@{self.system.db_host}:{self.system.db_port}/{self.system.db_name}")

    def get_mqtt_config(self) -> Dict[str, Union[str, int]]:
        """MQTT 설정 반환"""
        return {
            'host': self.system.mqtt_broker,
            'port': self.system.mqtt_port,
            'topic_prefix': self.system.mqtt_topic_prefix
        }

    def get_tou_schedule(self) -> Dict[str, List[int]]:
        """TOU 요금제 스케줄 반환"""
        return {
            'peak': self.power.tou_peak_hours,
            'off_peak': self.power.tou_off_peak_hours,
            'normal': [h for h in range(24)
                       if h not in self.power.tou_peak_hours + self.power.tou_off_peak_hours]
        }

    def get_power_constraints(self) -> Dict[str, float]:
        """전력 제약 조건 반환"""
        effective_limit = self.power.peak_power_limit * (1 - self.power.safety_margin)
        return {
            'peak_limit': self.power.peak_power_limit,
            'effective_limit': effective_limit,
            'safety_margin': self.power.safety_margin
        }

    def save_config(self, output_path: Optional[str] = None):
        """현재 설정을 파일로 저장"""
        if output_path is None:
            output_path = self.config_path

        config_data = {
            'model': {
                'anomaly_contamination': self.model.anomaly_contamination,
                'anomaly_n_estimators': self.model.anomaly_n_estimators,
                'anomaly_max_samples': self.model.anomaly_max_samples,
                'lstm_epochs': self.model.lstm_epochs,
                'lstm_batch_size': self.model.lstm_batch_size,
                'lstm_sequence_length': self.model.lstm_sequence_length,
                'xgb_n_estimators': self.model.xgb_n_estimators,
                'xgb_learning_rate': self.model.xgb_learning_rate,
                'xgb_max_depth': self.model.xgb_max_depth
            },
            'data': {
                'sensor_data_path': self.data.sensor_data_path,
                'processed_data_path': self.data.processed_data_path,
                'model_data_path': self.data.model_data_path,
                'batch_size': self.data.batch_size,
                'train_ratio': self.data.train_ratio,
                'val_ratio': self.data.val_ratio,
                'test_ratio': self.data.test_ratio,
                'scaling_method': self.data.scaling_method,
                'max_missing_ratio': self.data.max_missing_ratio,
                'outlier_threshold': self.data.outlier_threshold
            },
            'system': {
                'realtime_batch_size': self.system.realtime_batch_size,
                'processing_interval': self.system.processing_interval,
                'alert_threshold': self.system.alert_threshold,
                'db_host': self.system.db_host,
                'db_port': self.system.db_port,
                'db_name': self.system.db_name,
                'mqtt_broker': self.system.mqtt_broker,
                'mqtt_port': self.system.mqtt_port,
                'mqtt_topic_prefix': self.system.mqtt_topic_prefix,
                'api_host': self.system.api_host,
                'api_port': self.system.api_port,
                'api_workers': self.system.api_workers
            },
            'power': {
                'tou_peak_hours': self.power.tou_peak_hours,
                'tou_off_peak_hours': self.power.tou_off_peak_hours,
                'tou_peak_rate': self.power.tou_peak_rate,
                'tou_off_peak_rate': self.power.tou_off_peak_rate,
                'tou_normal_rate': self.power.tou_normal_rate,
                'peak_power_limit': self.power.peak_power_limit,
                'safety_margin': self.power.safety_margin,
                'scheduling_horizon': self.power.scheduling_horizon,
                'min_processing_time': self.power.min_processing_time,
                'max_processing_time': self.power.max_processing_time
            }
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        print(f"✅ 설정 파일 저장 완료: {output_path}")


# 전역 설정 인스턴스
config = ConfigManager()


def get_config() -> ConfigManager:
    """전역 설정 인스턴스 반환"""
    return config


def update_config(env: str = None, config_path: str = None) -> ConfigManager:
    """설정 재로드"""
    global config
    if env or config_path:
        config = ConfigManager(config_path=config_path, env=env or config.env)
    else:
        config._load_config()
    return config