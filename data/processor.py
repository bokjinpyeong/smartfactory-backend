"""
스마트팩토리 에너지 관리 시스템 데이터 전처리 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

from ..core.config import get_config
from ..core.logger import get_logger, log_performance
from ..core.exceptions import (
    DataProcessingError, DataValidationError,
    InsufficientDataError, DataCorruptionError
)


class DataProcessor:
    """데이터 전처리 클래스"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("data_processor")
        self.scalers = {}
        self.encoders = {}
        self._feature_names = []

    @log_performance("data_cleaning")
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정리"""
        self.logger.info("데이터 정리 시작",
                         original_shape=data.shape)

        try:
            # 1. 기본 정리
            data_clean = self._basic_cleaning(data)

            # 2. 이상치 제거
            data_clean = self._remove_outliers(data_clean)

            # 3. 결측치 처리
            data_clean = self._handle_missing_values(data_clean)

            # 4. 데이터 타입 최적화
            data_clean = self._optimize_dtypes(data_clean)

            self.logger.info("데이터 정리 완료",
                             final_shape=data_clean.shape,
                             removed_rows=len(data) - len(data_clean))

            return data_clean

        except Exception as e:
            raise DataProcessingError(
                "데이터 정리 중 오류 발생",
                details={'error': str(e), 'data_shape': data.shape}
            ) from e

    def _basic_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """기본 데이터 정리"""
        data_clean = data.copy()

        # 음수 전력 소비 제거
        if 'Power_Consumption' in data_clean.columns:
            negative_power = (data_clean['Power_Consumption'] < 0).sum()
            if negative_power > 0:
                self.logger.warning(f"음수 전력 소비 제거: {negative_power}개")
                data_clean = data_clean[data_clean['Power_Consumption'] >= 0]

        # 비현실적인 온도 제거
        temp_columns = [col for col in data_clean.columns if 'Temperature' in col]
        for col in temp_columns:
            before_count = len(data_clean)
            data_clean = data_clean[
                (data_clean[col] >= -50) & (data_clean[col] <= 200)
                ]
            removed = before_count - len(data_clean)
            if removed > 0:
                self.logger.warning(f"비현실적인 온도 제거 ({col}): {removed}개")

        # 중복 제거
        duplicates = data_clean.duplicated().sum()
        if duplicates > 0:
            data_clean = data_clean.drop_duplicates()
            self.logger.info(f"중복 데이터 제거: {duplicates}개")

        return data_clean

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """이상치 제거"""
        data_clean = data.copy()
        numeric_columns = data_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in ['Machine_ID', 'Event_Sequence_Number']:  # ID 컬럼 제외
                continue

            Q1 = data_clean[col].quantile(0.25)
            Q3 = data_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            before_count = len(data_clean)
            data_clean = data_clean[
                (data_clean[col] >= lower_bound) & (data_clean[col] <= upper_bound)
                ]
            removed = before_count - len(data_clean)

            if removed > 0:
                self.logger.debug(f"이상치 제거 ({col}): {removed}개")

        return data_clean

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        data_clean = data.copy()

        # 결측치 비율 확인
        missing_ratio = data_clean.isnull().sum() / len(data_clean)
        high_missing_cols = missing_ratio[missing_ratio > self.config.data.max_missing_ratio].index

        if len(high_missing_cols) > 0:
            self.logger.warning(f"결측치 비율이 높은 컬럼 제거: {list(high_missing_cols)}")
            data_clean = data_clean.drop(columns=high_missing_cols)

        # 수치형 컬럼: 평균값으로 대체
        numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data_clean[col].isnull().sum() > 0:
                mean_val = data_clean[col].mean()
                data_clean[col].fillna(mean_val, inplace=True)
                self.logger.debug(f"결측치 대체 ({col}): 평균값 {mean_val:.2f}")

        # 범주형 컬럼: 최빈값으로 대체
        categorical_columns = data_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if data_clean[col].isnull().sum() > 0:
                mode_val = data_clean[col].mode().iloc[0] if not data_clean[col].mode().empty else 'Unknown'
                data_clean[col].fillna(mode_val, inplace=True)
                self.logger.debug(f"결측치 대체 ({col}): 최빈값 {mode_val}")

        return data_clean

    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 최적화"""
        data_optimized = data.copy()

        # Float64 → Float32
        float_columns = data_optimized.select_dtypes(include=['float64']).columns
        if len(float_columns) > 0:
            data_optimized[float_columns] = data_optimized[float_columns].astype('float32')
            self.logger.debug(f"Float32로 변환: {len(float_columns)}개 컬럼")

        # 정수형 최적화
        int_columns = ['Shaft_Alignment_Status', 'Failure_Occurrence', 'Event_Sequence_Number']
        for col in int_columns:
            if col in data_optimized.columns:
                data_optimized[col] = data_optimized[col].astype('int16')

        # 카테고리형 변환
        categorical_columns = [
            'Machine_ID', 'Machine_Type', 'Production_Line_ID',
            'Operational_Mode', 'Job_Code', 'Shift_Code',
            'Operator_ID', 'Machine_Location_Zone'
        ]
        for col in categorical_columns:
            if col in data_optimized.columns:
                data_optimized[col] = data_optimized[col].astype('category')

        # 메모리 사용량 로깅
        memory_usage = data_optimized.memory_usage(deep=True).sum() / 1024 ** 2
        self.logger.info(f"최적화 후 메모리 사용량: {memory_usage:.2f} MB")

        return data_optimized

    @log_performance("realistic_power_model")
    def create_realistic_power_model(self, data: pd.DataFrame) -> pd.DataFrame:
        """물리 법칙을 반영한 현실적인 전력 소비 계산"""
        self.logger.info("현실적인 전력 소비 모델 생성")

        data_with_realistic = data.copy()

        def calculate_realistic_power(row):
            # 기본 소비 전력 (기계 타입별)
            base_power = {
                'CNC': 500, 'Lathe': 300, 'Mill': 400, 'Drill': 200
            }

            machine_type = row.get('Machine_Type', 'CNC')
            base = base_power.get(machine_type, 400)

            # 부하에 따른 전력 (토크 × RPM = 기계적 파워)
            mechanical_power = (row['Load_Torque'] * row['Shaft_Speed_RPM']) / 1000000
            load_power = mechanical_power * 1000 * 0.8  # 80% 효율

            # 온도에 따른 추가 소모 (냉각 시스템)
            temp_excess = max(0, row['Motor_Temperature'] - 80)
            cooling_power = temp_excess * 2

            # 진동에 따른 비효율성
            vibration_penalty = row['RMS_Vibration'] * 10

            # 역률에 따른 효율성
            power_factor = max(0.5, row['Power_Factor'])
            efficiency_factor = power_factor

            # 작업 부하율 반영
            workload_factor = row['Workload_Percentage'] / 100

            # 최종 계산
            total_power = (base + load_power + cooling_power + vibration_penalty) * workload_factor / efficiency_factor

            # 랜덤 노이즈 추가 (±5%)
            noise = np.random.normal(1, 0.05)
            total_power *= noise

            return max(50, total_power)  # 최소 50W

        # 새로운 현실적인 전력 소비 계산
        data_with_realistic['Power_Consumption_Realistic'] = data_with_realistic.apply(
            calculate_realistic_power, axis=1
        )

        # 통계 정보 로깅
        original_mean = data_with_realistic['Power_Consumption'].mean()
        realistic_mean = data_with_realistic['Power_Consumption_Realistic'].mean()

        self.logger.info("현실적인 전력 모델 생성 완료",
                         original_mean=original_mean,
                         realistic_mean=realistic_mean,
                         difference_percent=(realistic_mean - original_mean) / original_mean * 100)

        return data_with_realistic

    @log_performance("feature_engineering")
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """피처 엔지니어링"""
        self.logger.info("피처 엔지니어링 시작")

        data_engineered = data.copy()

        try:
            # 1. 시계열 피처
            data_engineered = self._create_time_features(data_engineered)

            # 2. 파생 변수
            data_engineered = self._create_derived_features(data_engineered)

            # 3. 상호작용 피처
            data_engineered = self._create_interaction_features(data_engineered)

            # 4. 카테고리 인코딩
            data_engineered = self._encode_categorical_features(data_engineered)

            self.logger.info("피처 엔지니어링 완료",
                             original_features=data.shape[1],
                             final_features=data_engineered.shape[1])

            return data_engineered

        except Exception as e:
            raise DataProcessingError(
                "피처 엔지니어링 중 오류 발생",
                details={'error': str(e)}
            ) from e

    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """시계열 피처 생성"""
        if 'Timestamp' not in data.columns:
            return data

        data_with_time = data.copy()
        data_with_time['Timestamp'] = pd.to_datetime(data_with_time['Timestamp'])

        # 시간 관련 피처
        data_with_time['Hour'] = data_with_time['Timestamp'].dt.hour
        data_with_time['Day_of_Week'] = data_with_time['Timestamp'].dt.dayofweek
        data_with_time['Month'] = data_with_time['Timestamp'].dt.month
        data_with_time['Is_Weekend'] = (data_with_time['Day_of_Week'] >= 5).astype(int)

        # TOU 요금제 시간대
        def get_time_price_factor(hour):
            if hour in self.config.power.tou_peak_hours:
                return self.config.power.tou_peak_rate
            elif hour in self.config.power.tou_off_peak_hours:
                return self.config.power.tou_off_peak_rate
            else:
                return self.config.power.tou_normal_rate

        data_with_time['Time_Price_Factor'] = data_with_time['Hour'].apply(get_time_price_factor)

        self.logger.debug("시계열 피처 생성 완료")
        return data_with_time

    def _create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 생성"""
        data_derived = data.copy()

        # 전력 효율성 지표
        if 'Power_Consumption_Realistic' in data_derived.columns and 'Mechanical_Power' in data_derived.columns:
            data_derived['Power_Efficiency'] = data_derived['Power_Consumption_Realistic'] / (
                        data_derived['Mechanical_Power'] + 1)

        # 온도 이상 플래그
        if 'Motor_Temperature' in data_derived.columns:
            data_derived['High_Temperature_Flag'] = (data_derived['Motor_Temperature'] > 130).astype(int)

        # 부하 수준 카테고리
        if 'Workload_Percentage' in data_derived.columns:
            data_derived['Load_Category'] = pd.cut(
                data_derived['Workload_Percentage'],
                bins=[0, 30, 70, 100],
                labels=['Low', 'Medium', 'High']
            )

        # 종합 전력 지표
        if all(col in data_derived.columns for col in ['Current_Phase_A', 'Current_Phase_B', 'Current_Phase_C']):
            data_derived['Total_Current'] = (
                    data_derived['Current_Phase_A'] +
                    data_derived['Current_Phase_B'] +
                    data_derived['Current_Phase_C']
            )

        if all(col in data_derived.columns for col in ['Voltage_Phase_A', 'Voltage_Phase_B', 'Voltage_Phase_C']):
            data_derived['Average_Voltage'] = (
                                                      data_derived['Voltage_Phase_A'] +
                                                      data_derived['Voltage_Phase_B'] +
                                                      data_derived['Voltage_Phase_C']
                                              ) / 3

        # 기계적 파워
        if 'Load_Torque' in data_derived.columns and 'Shaft_Speed_RPM' in data_derived.columns:
            data_derived['Mechanical_Power'] = data_derived['Load_Torque'] * data_derived['Shaft_Speed_RPM'] / 1000

        self.logger.debug("파생 변수 생성 완료")
        return data_derived

    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """상호작용 피처 생성"""
        data_interaction = data.copy()

        # 온도와 부하의 상호작용
        if 'Motor_Temperature' in data_interaction.columns and 'Workload_Percentage' in data_interaction.columns:
            data_interaction['Temp_Load_Interaction'] = (
                    data_interaction['Motor_Temperature'] * data_interaction['Workload_Percentage'] / 100
            )

        # 진동과 속도의 상호작용
        if 'RMS_Vibration' in data_interaction.columns and 'Shaft_Speed_RPM' in data_interaction.columns:
            data_interaction['Vibration_Speed_Interaction'] = (
                    data_interaction['RMS_Vibration'] * data_interaction['Shaft_Speed_RPM'] / 1000
            )

        self.logger.debug("상호작용 피처 생성 완료")
        return data_interaction

    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """카테고리 피처 인코딩"""
        data_encoded = data.copy()

        categorical_features = ['Machine_ID', 'Machine_Type', 'Operational_Mode', 'Load_Category']

        for col in categorical_features:
            if col in data_encoded.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    data_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(data_encoded[col].astype(str))
                else:
                    # 기존 인코더 사용 (새로운 값 처리)
                    try:
                        data_encoded[f'{col}_encoded'] = self.encoders[col].transform(data_encoded[col].astype(str))
                    except ValueError:
                        # 새로운 카테고리 값이 있는 경우 처리
                        known_classes = set(self.encoders[col].classes_)
                        data_values = set(data_encoded[col].astype(str).unique())
                        new_values = data_values - known_classes

                        if new_values:
                            self.logger.warning(f"새로운 카테고리 값 발견 ({col}): {new_values}")
                            # 인코더 재학습
                            self.encoders[col] = LabelEncoder()
                            data_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(
                                data_encoded[col].astype(str))

        self.logger.debug("카테고리 인코딩 완료")
        return data_encoded

    @log_performance("data_scaling")
    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """피처 스케일링"""
        self.logger.info("피처 스케일링 시작")

        data_scaled = data.copy()

        # 스케일링할 수치형 컬럼 선택
        numeric_columns = data_scaled.select_dtypes(include=[np.number]).columns
        exclude_columns = ['Machine_ID', 'Event_Sequence_Number'] + [col for col in numeric_columns if
                                                                     col.endswith('_encoded')]

        scale_columns = [col for col in numeric_columns if col not in exclude_columns]

        if len(scale_columns) == 0:
            self.logger.warning("스케일링할 컬럼이 없습니다")
            return data_scaled

        # 스케일러 선택
        if self.config.data.scaling_method == 'standard':
            scaler_class = StandardScaler
        elif self.config.data.scaling_method == 'minmax':
            scaler_class = MinMaxScaler
        else:
            raise DataValidationError(f"지원하지 않는 스케일링 방법: {self.config.data.scaling_method}")

        # 스케일링 수행
        if fit:
            self.scalers['feature_scaler'] = scaler_class()
            data_scaled[scale_columns] = self.scalers['feature_scaler'].fit_transform(data_scaled[scale_columns])
            self.logger.info(f"스케일러 학습 및 변환 완료 ({self.config.data.scaling_method})")
        else:
            if 'feature_scaler' not in self.scalers:
                raise DataProcessingError("스케일러가 학습되지 않았습니다. fit=True로 먼저 실행하세요.")
            data_scaled[scale_columns] = self.scalers['feature_scaler'].transform(data_scaled[scale_columns])
            self.logger.info("기존 스케일러로 변환 완료")

        return data_scaled

    @log_performance("data_splitting")
    def split_data(
            self,
            data: pd.DataFrame,
            target_col: str = 'Power_Consumption_Realistic',
            features: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """데이터 분할"""
        self.logger.info("데이터 분할 시작",
                         target_column=target_col,
                         total_samples=len(data))

        if target_col not in data.columns:
            raise DataValidationError(f"타겟 컬럼을 찾을 수 없습니다: {target_col}")

        # 피처 선택
        if features is None:
            features = self._select_final_features(data, target_col)

        # 존재하는 피처만 필터링
        available_features = [f for f in features if f in data.columns]
        if len(available_features) != len(features):
            missing_features = set(features) - set(available_features)
            self.logger.warning(f"일부 피처를 찾을 수 없습니다: {missing_features}")

        if len(available_features) == 0:
            raise DataProcessingError("사용할 수 있는 피처가 없습니다")

        # 피처와 타겟 분리
        X = data[available_features].copy()
        y = data[target_col].copy()

        # 결측치 제거
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            raise InsufficientDataError("결측치 제거 후 데이터가 없습니다")

        # 데이터 분할
        try:
            # Train/Test 분할
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_clean, y_clean,
                test_size=self.config.data.test_ratio,
                random_state=42,
                stratify=None  # 회귀 문제이므로 stratify 사용 안함
            )

            # Train/Validation 분할
            val_ratio_adjusted = self.config.data.val_ratio / (1 - self.config.data.test_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio_adjusted,
                random_state=42
            )

            # 피처 이름 저장
            self._feature_names = list(X_train.columns)

            self.logger.info("데이터 분할 완료",
                             train_samples=len(X_train),
                             val_samples=len(X_val),
                             test_samples=len(X_test),
                             features_count=len(self._feature_names))

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            raise DataProcessingError(
                "데이터 분할 중 오류 발생",
                details={'error': str(e), 'data_shape': X_clean.shape}
            ) from e

    def _select_final_features(self, data: pd.DataFrame, target_col: str) -> List[str]:
        """최종 피처 선택"""
        # 강한 상관관계 피처들
        core_features = [
            'Mechanical_Power', 'Workload_Percentage', 'Load_Torque',
            'Power_Factor', 'Shaft_Speed_RPM', 'Motor_Temperature'
        ]

        # 시계열 피처
        time_features = ['Hour', 'Day_of_Week', 'Month', 'Is_Weekend', 'Time_Price_Factor']

        # 파생 피처
        derived_features = ['Power_Efficiency', 'High_Temperature_Flag']

        # 인코딩된 피처
        encoded_features = [col for col in data.columns if col.endswith('_encoded')]

        # 모든 피처 결합
        all_features = core_features + time_features + derived_features + encoded_features

        # 실제 존재하는 피처만 선택
        final_features = [f for f in all_features if f in data.columns and f != target_col]

        self.logger.info(f"선택된 피처 수: {len(final_features)}")
        return final_features

    def get_feature_names(self) -> List[str]:
        """피처 이름 반환"""
        return self._feature_names.copy()

    def get_scalers(self) -> Dict:
        """스케일러 반환"""
        return self.scalers.copy()

    def get_encoders(self) -> Dict:
        """인코더 반환"""
        return self.encoders.copy()

    @log_performance("data_validation")
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 검증"""
        self.logger.info("데이터 검증 시작")

        validation_report = {
            'is_valid': True,
            'issues': [],
            'statistics': {},
            'warnings': []
        }

        try:
            # 1. 기본 통계
            validation_report['statistics'] = {
                'total_rows': len(data),
                'total_columns': data.shape[1],
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 ** 2,
                'missing_values': data.isnull().sum().sum(),
                'duplicate_rows': data.duplicated().sum()
            }

            # 2. 필수 컬럼 확인
            required_columns = ['Power_Consumption', 'Machine_ID', 'Timestamp']
            missing_required = [col for col in required_columns if col not in data.columns]
            if missing_required:
                validation_report['issues'].append({
                    'type': 'missing_required_columns',
                    'details': missing_required
                })
                validation_report['is_valid'] = False

            # 3. 데이터 타입 확인
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                validation_report['issues'].append({
                    'type': 'no_numeric_columns',
                    'details': 'numeric columns not found'
                })
                validation_report['is_valid'] = False

            # 4. 값 범위 확인
            for col in numeric_columns:
                if 'Power' in col:
                    negative_count = (data[col] < 0).sum()
                    if negative_count > 0:
                        validation_report['warnings'].append({
                            'type': 'negative_power_values',
                            'column': col,
                            'count': negative_count
                        })

                if 'Temperature' in col:
                    extreme_count = ((data[col] < -50) | (data[col] > 200)).sum()
                    if extreme_count > 0:
                        validation_report['warnings'].append({
                            'type': 'extreme_temperature_values',
                            'column': col,
                            'count': extreme_count
                        })

            # 5. 결측치 비율 확인
            missing_ratios = data.isnull().sum() / len(data)
            high_missing = missing_ratios[missing_ratios > self.config.data.max_missing_ratio]
            if len(high_missing) > 0:
                validation_report['warnings'].append({
                    'type': 'high_missing_ratio',
                    'columns': high_missing.to_dict()
                })

            # 6. 데이터 충분성 확인
            if len(data) < 1000:
                validation_report['warnings'].append({
                    'type': 'insufficient_data',
                    'current_size': len(data),
                    'recommended_minimum': 1000
                })

            self.logger.info("데이터 검증 완료",
                             is_valid=validation_report['is_valid'],
                             issues_count=len(validation_report['issues']),
                             warnings_count=len(validation_report['warnings']))

            return validation_report

        except Exception as e:
            validation_report['is_valid'] = False
            validation_report['issues'].append({
                'type': 'validation_error',
                'details': str(e)
            })
            return validation_report

    def create_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 요약 정보 생성"""
        summary = {
            'basic_info': {
                'shape': data.shape,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 ** 2,
                'dtypes': data.dtypes.value_counts().to_dict()
            },
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }

        # 수치형 컬럼 요약
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            summary['numeric_summary'][col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median()),
                'missing_count': int(data[col].isnull().sum())
            }

        # 범주형 컬럼 요약
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            value_counts = data[col].value_counts()
            summary['categorical_summary'][col] = {
                'unique_count': int(data[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing_count': int(data[col].isnull().sum())
            }

        return summary

    def save_preprocessing_artifacts(self, output_dir: str = "artifacts/preprocessing"):
        """전처리 아티팩트 저장"""
        import os
        import joblib

        os.makedirs(output_dir, exist_ok=True)

        # 스케일러 저장
        if self.scalers:
            scaler_path = os.path.join(output_dir, "scalers.pkl")
            joblib.dump(self.scalers, scaler_path)
            self.logger.info(f"스케일러 저장: {scaler_path}")

        # 인코더 저장
        if self.encoders:
            encoder_path = os.path.join(output_dir, "encoders.pkl")
            joblib.dump(self.encoders, encoder_path)
            self.logger.info(f"인코더 저장: {encoder_path}")

        # 피처 이름 저장
        if self._feature_names:
            feature_path = os.path.join(output_dir, "feature_names.pkl")
            joblib.dump(self._feature_names, feature_path)
            self.logger.info(f"피처 이름 저장: {feature_path}")

    def load_preprocessing_artifacts(self, input_dir: str = "artifacts/preprocessing"):
        """전처리 아티팩트 로드"""
        import os
        import joblib

        # 스케일러 로드
        scaler_path = os.path.join(input_dir, "scalers.pkl")
        if os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
            self.logger.info(f"스케일러 로드: {scaler_path}")

        # 인코더 로드
        encoder_path = os.path.join(input_dir, "encoders.pkl")
        if os.path.exists(encoder_path):
            self.encoders = joblib.load(encoder_path)
            self.logger.info(f"인코더 로드: {encoder_path}")

        # 피처 이름 로드
        feature_path = os.path.join(input_dir, "feature_names.pkl")
        if os.path.exists(feature_path):
            self._feature_names = joblib.load(feature_path)
            self.logger.info(f"피처 이름 로드: {feature_path}")


def create_processor() -> DataProcessor:
    """데이터 프로세서 인스턴스 생성"""
    return DataProcessor()


# 편의 함수들
def quick_clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """빠른 데이터 정리"""
    processor = create_processor()
    return processor.clean_data(data)


def quick_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """빠른 피처 엔지니어링"""
    processor = create_processor()
    data_with_realistic = processor.create_realistic_power_model(data)
    return processor.engineer_features(data_with_realistic)


def quick_data_preparation(
        data: pd.DataFrame,
        target_col: str = 'Power_Consumption_Realistic'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """전체 데이터 준비 파이프라인"""
    processor = create_processor()

    # 1. 데이터 정리
    data_clean = processor.clean_data(data)

    # 2. 현실적인 전력 모델 생성
    data_with_realistic = processor.create_realistic_power_model(data_clean)

    # 3. 피처 엔지니어링
    data_engineered = processor.engineer_features(data_with_realistic)

    # 4. 스케일링
    data_scaled = processor.scale_features(data_engineered, fit=True)

    # 5. 데이터 분할
    return processor.split_data(data_scaled, target_col)