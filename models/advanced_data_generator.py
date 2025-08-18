"""
고급 데이터 생성기 - 물리학 기반 현실적 공장 데이터 생성
ipynb의 create_enhanced_industrial_data 기반 모듈화
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from ..core.config import get_config
from ..core.logger import get_logger


class MachineType(Enum):
    """기계 타입 정의"""
    CNC = "CNC"
    LATHE = "Lathe"
    MILL = "Mill"
    DRILL = "Drill"
    GRINDER = "Grinder"


class OperationalMode(Enum):
    """운영 모드 정의"""
    AUTO = "Auto"
    MANUAL = "Manual"
    MAINTENANCE = "Maintenance"


class Shift(Enum):
    """교대 근무 정의"""
    MORNING = "Morning"
    AFTERNOON = "Afternoon"
    NIGHT = "Night"


class OperatorSkill(Enum):
    """운영자 숙련도 정의"""
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    EXPERT = "Expert"


@dataclass
class MachineProfile:
    """기계 프로필 정의"""
    machine_type: MachineType
    base_power: float  # 기본 전력 (W)
    power_range: Tuple[float, float]  # 전력 범위
    efficiency_factor: float  # 효율성 인수
    maintenance_cycle: int  # 유지보수 주기 (일)
    vibration_baseline: float  # 기본 진동 수준
    temperature_baseline: float  # 기본 온도


@dataclass
class EnvironmentalConditions:
    """환경 조건 정의"""
    ambient_temp_range: Tuple[float, float] = (20, 30)
    humidity_range: Tuple[float, float] = (40, 80)
    pressure_range: Tuple[float, float] = (1000, 1025)
    seasonal_factor: float = 1.0


class AdvancedDataGenerator:
    """고급 산업 데이터 생성기"""

    def __init__(self, random_seed: int = 42):
        self.logger = get_logger("data_generator")
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # 기계 프로필 초기화
        self.machine_profiles = self._initialize_machine_profiles()

        # 물리 상수
        self.physics_constants = {
            'torque_to_power_factor': 9549,  # RPM to rad/s conversion factor
            'voltage_nominal': 220,  # V
            'current_efficiency': 0.85,  # 전류 효율
            'cooling_factor': 3.0,  # 냉각 전력 인수
            'ambient_effect': 1.5  # 환경 영향 인수
        }

        # 교대별 승수
        self.shift_multipliers = {
            Shift.MORNING: 1.0,
            Shift.AFTERNOON: 0.95,
            Shift.NIGHT: 0.9
        }

        # 숙련도별 승수
        self.skill_multipliers = {
            OperatorSkill.BEGINNER: 1.15,
            OperatorSkill.INTERMEDIATE: 1.0,
            OperatorSkill.EXPERT: 0.92
        }

        self.logger.info("고급 데이터 생성기 초기화 완료")

    def _initialize_machine_profiles(self) -> Dict[MachineType, MachineProfile]:
        """기계 프로필 초기화"""
        return {
            MachineType.CNC: MachineProfile(
                MachineType.CNC, 650, (400, 900), 0.88, 30, 2.5, 85
            ),
            MachineType.LATHE: MachineProfile(
                MachineType.LATHE, 420, (250, 600), 0.85, 25, 2.0, 75
            ),
            MachineType.MILL: MachineProfile(
                MachineType.MILL, 550, (350, 750), 0.82, 28, 3.0, 80
            ),
            MachineType.DRILL: MachineProfile(
                MachineType.DRILL, 280, (180, 400), 0.80, 20, 1.5, 70
            ),
            MachineType.GRINDER: MachineProfile(
                MachineType.GRINDER, 380, (200, 550), 0.78, 22, 4.0, 90
            )
        }

    def generate_enhanced_industrial_data(
        self,
        n_samples: int = 100000,
        anomaly_rate: float = 0.05,
        environmental_conditions: Optional[EnvironmentalConditions] = None,
        include_3phase: bool = True
    ) -> pd.DataFrame:
        """향상된 산업용 데이터 생성"""

        self.logger.info(f"산업용 데이터 생성 중... ({n_samples:,}개 샘플)")

        if environmental_conditions is None:
            environmental_conditions = EnvironmentalConditions()

        # 기본 운영 조건 생성
        data = self._generate_base_operational_data(n_samples, environmental_conditions)

        # 3상 전력 시스템 데이터 추가
        if include_3phase:
            data = self._add_3phase_electrical_data(data)

        # 물리학 기반 전력 계산
        data = self._calculate_physics_based_power(data)

        # 이상 상황 시뮬레이션
        data = self._simulate_anomalous_conditions(data, anomaly_rate)

        # 물리적 제약 적용
        data = self._apply_physical_constraints(data)

        # 시계열 특성 추가
        data = self._add_temporal_features(data)

        # 최종 검증 및 정리
        data = self._finalize_data(data)

        self.logger.info(f"데이터 생성 완료: {data.shape}")
        return data

    def _generate_base_operational_data(
        self,
        n_samples: int,
        env_conditions: EnvironmentalConditions
    ) -> pd.DataFrame:
        """기본 운영 데이터 생성"""

        # 범주형 변수 생성
        machine_types = np.random.choice(list(MachineType), n_samples)
        operational_modes = np.random.choice(list(OperationalMode), n_samples, p=[0.7, 0.25, 0.05])
        shifts = np.random.choice(list(Shift), n_samples)
        operator_skills = np.random.choice(list(OperatorSkill), n_samples, p=[0.2, 0.6, 0.2])

        # 기계별 ID 생성
        machine_ids = [f"{mt.value}_{i%10:03d}" for i, mt in enumerate(machine_types)]

        data = pd.DataFrame({
            'Machine_ID': machine_ids,
            'Machine_Type': [mt.value for mt in machine_types],
            'Operational_Mode': [om.value for om in operational_modes],
            'Shift': [s.value for s in shifts],
            'Operator_Skill': [os.value for os in operator_skills]
        })

        # 온도 센서들 (기계별 특성 반영)
        for i, machine_type in enumerate(machine_types):
            profile = self.machine_profiles[machine_type]

            # 모터 온도 - 기계별 기본값 + 변동
            base_temp = profile.temperature_baseline
            data.loc[i, 'Motor_Temperature'] = np.random.normal(base_temp, 12)

            # 베어링 온도 - 모터보다 낮음
            data.loc[i, 'Bearing_Temperature'] = np.random.normal(base_temp - 40, 8)

            # 오일 온도 - 중간값
            data.loc[i, 'Oil_Temperature'] = np.random.normal(base_temp - 25, 10)

        # 환경 온도 (계절성 반영)
        ambient_temp_base = np.mean(env_conditions.ambient_temp_range)
        ambient_temp_var = (env_conditions.ambient_temp_range[1] - env_conditions.ambient_temp_range[0]) / 4
        data['Ambient_Temperature'] = np.random.normal(
            ambient_temp_base * env_conditions.seasonal_factor,
            ambient_temp_var
        )

        # 진동 및 기계적 요소
        for i, machine_type in enumerate(machine_types):
            profile = self.machine_profiles[machine_type]

            # RMS 진동 - 기계별 기본값
            data.loc[i, 'RMS_Vibration'] = np.random.exponential(profile.vibration_baseline)

            # 피크 진동 - RMS의 1.5-2배
            data.loc[i, 'Peak_Vibration'] = data.loc[i, 'RMS_Vibration'] * np.random.uniform(1.5, 2.0)

        # 토크 및 RPM (상관관계 있음)
        data['Shaft_Speed_RPM'] = np.random.normal(1800, 180, n_samples)

        # 토크는 RPM과 반비례 관계 (일정한 전력 유지)
        base_torque = 120
        speed_factor = 1800 / data['Shaft_Speed_RPM']
        data['Load_Torque'] = base_torque * speed_factor * np.random.normal(1.0, 0.2, n_samples)

        # 전기적 요소 (기본값)
        data['Total_Current'] = np.random.normal(55, 12, n_samples)
        data['Avg_Voltage'] = np.random.normal(self.physics_constants['voltage_nominal'], 8, n_samples)
        data['Power_Factor'] = np.random.uniform(0.75, 0.95, n_samples)

        # 운영 요소
        data['Workload_Percentage'] = np.random.uniform(15, 95, n_samples)
        data['Production_Rate'] = np.random.normal(100, 15, n_samples)
        data['Efficiency_Index'] = np.random.normal(0.85, 0.1, n_samples)

        # 환경 요소
        data['Humidity'] = np.random.uniform(*env_conditions.humidity_range, n_samples)
        data['Pressure'] = np.random.normal(np.mean(env_conditions.pressure_range), 15, n_samples)

        return data

    def _add_3phase_electrical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """3상 전력 시스템 데이터 추가"""

        # 총 전류를 3상으로 분배 (약간의 불균형 포함)
        total_current = data['Total_Current']

        # 기본적으로 균등 분배하되 약간의 불균형 추가
        current_base = total_current / 3
        imbalance_factor = np.random.normal(0, 0.05, len(data))

        data['Current_Phase_A'] = current_base * (1 + imbalance_factor)
        data['Current_Phase_B'] = current_base * (1 - imbalance_factor * 0.5)
        data['Current_Phase_C'] = total_current - data['Current_Phase_A'] - data['Current_Phase_B']

        # 전압도 3상으로 분배 (더 작은 불균형)
        voltage_base = data['Avg_Voltage']
        voltage_imbalance = np.random.normal(0, 0.02, len(data))

        data['Voltage_Phase_A'] = voltage_base * (1 + voltage_imbalance)
        data['Voltage_Phase_B'] = voltage_base * (1 - voltage_imbalance * 0.3)
        data['Voltage_Phase_C'] = voltage_base * (1 - voltage_imbalance * 0.7)

        # 평균값 재계산
        data['Avg_Voltage'] = (data['Voltage_Phase_A'] + data['Voltage_Phase_B'] + data['Voltage_Phase_C']) / 3
        data['Total_Current'] = data['Current_Phase_A'] + data['Current_Phase_B'] + data['Current_Phase_C']

        return data

    def _calculate_physics_based_power(self, data: pd.DataFrame) -> pd.DataFrame:
        """물리학 기반 전력 계산"""

        # 기계별 기본 전력 매핑
        machine_base_power = data['Machine_Type'].map({
            'CNC': 650, 'Lathe': 420, 'Mill': 550, 'Drill': 280, 'Grinder': 380
        })

        # 교대 및 숙련도 승수 적용
        shift_mult = data['Shift'].map({
            'Morning': 1.0, 'Afternoon': 0.95, 'Night': 0.9
        })
        skill_mult = data['Operator_Skill'].map({
            'Beginner': 1.15, 'Intermediate': 1.0, 'Expert': 0.92
        })

        # 복합 전력 계산 공식
        power_components = []

        # 1. 기본 전력 (기계 타입 + 교대 + 숙련도)
        base_power = machine_base_power * shift_mult * skill_mult
        power_components.append(base_power)

        # 2. 전기적 전력 (P = √3 × V × I × PF)
        electrical_power = (
            np.sqrt(3) * data['Avg_Voltage'] * data['Total_Current'] *
            data['Power_Factor'] * self.physics_constants['current_efficiency'] * 0.001
        )
        power_components.append(electrical_power)

        # 3. 기계적 전력 (P = T × ω / 9549)
        mechanical_power = (
            data['Load_Torque'] * data['Shaft_Speed_RPM'] /
            self.physics_constants['torque_to_power_factor'] * 0.8
        )
        power_components