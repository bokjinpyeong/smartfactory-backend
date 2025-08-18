"""
스마트팩토리 에너지 관리 시스템 - 데이터 검증 모듈

포괄적 데이터 품질 검증 시스템
- 실시간 데이터 품질 모니터링
- 다층 검증 프로세스
- 자동화된 품질 보고서
- 장애 허용성을 위한 검증 정책
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import json

# Core 모듈
from core.config import get_config
from core.logger import get_logger, log_performance
from core.exceptions import (
    DataValidationError, InsufficientDataError,
    SensorDataError, safe_execute
)


class ValidationLevel(Enum):
    """검증 레벨"""
    BASIC = "basic"  # 기본 검증
    STANDARD = "standard"  # 표준 검증
    STRICT = "strict"  # 엄격한 검증
    CUSTOM = "custom"  # 사용자 정의


class ValidationResult(Enum):
    """검증 결과"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """검증 이슈"""
    level: ValidationResult
    rule_name: str
    message: str
    affected_rows: List[int] = field(default_factory=list)
    affected_columns: List[str] = field(default_factory=list)
    severity_score: float = 0.0
    suggested_action: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            'level': self.level.value,
            'rule_name': self.rule_name,
            'message': self.message,
            'affected_rows_count': len(self.affected_rows),
            'affected_columns': self.affected_columns,
            'severity_score': self.severity_score,
            'suggested_action': self.suggested_action,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ValidationReport:
    """검증 보고서"""
    data_shape: Tuple[int, int]
    validation_level: ValidationLevel
    total_issues: int
    issues_by_level: Dict[str, int]
    issues: List[ValidationIssue]
    overall_score: float
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)

    def is_valid(self) -> bool:
        """데이터 유효성 판단"""
        return self.issues_by_level.get('fail', 0) == 0 and self.issues_by_level.get('error', 0) == 0

    def get_summary(self) -> Dict:
        """요약 정보"""
        return {
            'data_shape': self.data_shape,
            'validation_level': self.validation_level.value,
            'is_valid': self.is_valid(),
            'overall_score': self.overall_score,
            'total_issues': self.total_issues,
            'issues_by_level': self.issues_by_level,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat()
        }

    def to_json(self) -> str:
        """JSON 형태로 변환"""
        report_data = {
            'summary': self.get_summary(),
            'issues': [issue.to_dict() for issue in self.issues]
        }
        return json.dumps(report_data, indent=2, ensure_ascii=False)


class BaseValidationRule(ABC):
    """기본 검증 규칙 클래스"""

    def __init__(self, rule_name: str, severity_weight: float = 1.0):
        self.rule_name = rule_name
        self.severity_weight = severity_weight
        self.logger = get_logger(f"validator.{rule_name}")

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """검증 수행"""
        pass

    def _create_issue(self,
                      level: ValidationResult,
                      message: str,
                      affected_rows: List[int] = None,
                      affected_columns: List[str] = None,
                      suggested_action: str = "") -> ValidationIssue:
        """검증 이슈 생성"""
        severity_score = self._calculate_severity_score(level, affected_rows or [])

        return ValidationIssue(
            level=level,
            rule_name=self.rule_name,
            message=message,
            affected_rows=affected_rows or [],
            affected_columns=affected_columns or [],
            severity_score=severity_score,
            suggested_action=suggested_action
        )

    def _calculate_severity_score(self, level: ValidationResult, affected_rows: List[int]) -> float:
        """심각도 점수 계산"""
        base_scores = {
            ValidationResult.PASS: 0.0,
            ValidationResult.WARNING: 0.3,
            ValidationResult.FAIL: 0.7,
            ValidationResult.ERROR: 1.0
        }

        base_score = base_scores.get(level, 0.5)
        impact_factor = min(len(affected_rows) / 1000, 1.0)  # 영향받는 행 수 고려

        return (base_score + impact_factor * 0.3) * self.severity_weight


class SchemaValidationRule(BaseValidationRule):
    """스키마 검증 규칙"""

    def __init__(self, required_columns: List[str], optional_columns: List[str] = None):
        super().__init__("schema_validation")
        self.required_columns = required_columns
        self.optional_columns = optional_columns or []
        self.expected_dtypes = {
            'timestamp': ['datetime64[ns]', 'object'],
            'machine_id': ['object', 'string'],
            'sensor_id': ['object', 'string'],
            'power_consumption': ['float64', 'float32', 'int64', 'int32'],
            'voltage': ['float64', 'float32', 'int64', 'int32'],
            'current': ['float64', 'float32', 'int64', 'int32'],
            'temperature': ['float64', 'float32', 'int64', 'int32'],
            'humidity': ['float64', 'float32', 'int64', 'int32']
        }

    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """스키마 검증"""
        issues = []

        # 1. 필수 컬럼 존재 확인
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            issues.append(self._create_issue(
                ValidationResult.FAIL,
                f"필수 컬럼 누락: {missing_columns}",
                affected_columns=missing_columns,
                suggested_action="누락된 컬럼을 추가하거나 데이터 소스 확인"
            ))

        # 2. 데이터 타입 검증
        for col in data.columns:
            if col in self.expected_dtypes:
                expected_types = self.expected_dtypes[col]
                actual_type = str(data[col].dtype)

                if actual_type not in expected_types:
                    issues.append(self._create_issue(
                        ValidationResult.WARNING,
                        f"컬럼 '{col}' 데이터 타입 불일치: 예상 {expected_types}, 실제 {actual_type}",
                        affected_columns=[col],
                        suggested_action=f"컬럼 '{col}'을 적절한 타입으로 변환"
                    ))

        # 3. 빈 DataFrame 확인
        if data.empty:
            issues.append(self._create_issue(
                ValidationResult.FAIL,
                "데이터가 비어있음",
                suggested_action="데이터 소스 확인 및 데이터 수집 점검"
            ))

        return issues


class DataQualityRule(BaseValidationRule):
    """데이터 품질 검증 규칙"""

    def __init__(self, missing_threshold: float = 0.1, duplicate_threshold: float = 0.05):
        super().__init__("data_quality")
        self.missing_threshold = missing_threshold
        self.duplicate_threshold = duplicate_threshold

    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """데이터 품질 검증"""
        issues = []

        if data.empty:
            return issues

        # 1. 결측값 비율 확인
        for col in data.columns:
            missing_ratio = data[col].isnull().sum() / len(data)

            if missing_ratio > self.missing_threshold:
                missing_rows = data[data[col].isnull()].index.tolist()

                level = ValidationResult.FAIL if missing_ratio > 0.5 else ValidationResult.WARNING
                issues.append(self._create_issue(
                    level,
                    f"컬럼 '{col}' 결측값 비율 높음: {missing_ratio:.2%}",
                    affected_rows=missing_rows,
                    affected_columns=[col],
                    suggested_action="결측값 처리 (보간, 제거, 또는 기본값 대체)"
                ))

        # 2. 중복 행 확인
        duplicate_rows = data.duplicated()
        duplicate_ratio = duplicate_rows.sum() / len(data)

        if duplicate_ratio > self.duplicate_threshold:
            duplicate_indices = data[duplicate_rows].index.tolist()

            level = ValidationResult.FAIL if duplicate_ratio > 0.2 else ValidationResult.WARNING
            issues.append(self._create_issue(
                level,
                f"중복 행 비율 높음: {duplicate_ratio:.2%}",
                affected_rows=duplicate_indices,
                suggested_action="중복 행 제거 또는 데이터 수집 로직 점검"
            ))

        # 3. 컬럼별 고유값 분포 확인
        for col in data.select_dtypes(include=['object']).columns:
            unique_ratio = data[col].nunique() / len(data)

            if unique_ratio < 0.01 and data[col].nunique() > 1:  # 너무 적은 고유값
                issues.append(self._create_issue(
                    ValidationResult.WARNING,
                    f"컬럼 '{col}' 고유값 비율 낮음: {unique_ratio:.2%}",
                    affected_columns=[col],
                    suggested_action="데이터 다양성 부족 - 수집 범위 확장 고려"
                ))

        return issues


class BusinessLogicRule(BaseValidationRule):
    """비즈니스 로직 검증 규칙"""

    def __init__(self):
        super().__init__("business_logic", severity_weight=1.5)
        self.power_limits = {'min': 0, 'max': 10000}  # kW
        self.voltage_limits = {'min': 0, 'max': 500}  # V
        self.current_limits = {'min': 0, 'max': 100}  # A
        self.temp_limits = {'min': -50, 'max': 100}  # C
        self.humidity_limits = {'min': 0, 'max': 100}  # %

    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """비즈니스 로직 검증"""
        issues = []

        if data.empty:
            return issues

        # 1. 전력 소비량 범위 검증
        if 'power_consumption' in data.columns:
            issues.extend(self._validate_range(
                data, 'power_consumption', self.power_limits,
                "전력 소비량이 정상 범위를 벗어남"
            ))

        # 2. 전압 범위 검증
        if 'voltage' in data.columns:
            issues.extend(self._validate_range(
                data, 'voltage', self.voltage_limits,
                "전압이 정상 범위를 벗어남"
            ))

        # 3. 전류 범위 검증
        if 'current' in data.columns:
            issues.extend(self._validate_range(
                data, 'current', self.current_limits,
                "전류가 정상 범위를 벗어남"
            ))

        # 4. 온도 범위 검증
        if 'temperature' in data.columns:
            issues.extend(self._validate_range(
                data, 'temperature', self.temp_limits,
                "온도가 정상 범위를 벗어남"
            ))

        # 5. 습도 범위 검증
        if 'humidity' in data.columns:
            issues.extend(self._validate_range(
                data, 'humidity', self.humidity_limits,
                "습도가 정상 범위를 벗어남"
            ))

        # 6. 전력 법칙 검증 (P = V * I)
        if all(col in data.columns for col in ['power_consumption', 'voltage', 'current']):
            issues.extend(self._validate_power_law(data))

        # 7. 시간 순서 검증
        if 'timestamp' in data.columns:
            issues.extend(self._validate_temporal_order(data))

        return issues

    def _validate_range(self, data: pd.DataFrame, column: str, limits: Dict, message: str) -> List[ValidationIssue]:
        """범위 검증"""
        issues = []

        # 최소값 검증
        min_violations = data[data[column] < limits['min']].index.tolist()
        if min_violations:
            issues.append(self._create_issue(
                ValidationResult.FAIL,
                f"{message} - 최소값({limits['min']}) 미만: {len(min_violations)}개",
                affected_rows=min_violations,
                affected_columns=[column],
                suggested_action=f"컬럼 '{column}' 값을 {limits['min']} 이상으로 수정"
            ))

        # 최대값 검증
        max_violations = data[data[column] > limits['max']].index.tolist()
        if max_violations:
            issues.append(self._create_issue(
                ValidationResult.FAIL,
                f"{message} - 최대값({limits['max']}) 초과: {len(max_violations)}개",
                affected_rows=max_violations,
                affected_columns=[column],
                suggested_action=f"컬럼 '{column}' 값을 {limits['max']} 이하로 수정"
            ))

        return issues

    def _validate_power_law(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """전력 법칙 검증 (P = V * I)"""
        issues = []

        # 계산된 전력과 측정된 전력 비교
        calculated_power = data['voltage'] * data['current']
        measured_power = data['power_consumption']

        # 10% 이상 차이나는 경우
        power_diff_ratio = abs(calculated_power - measured_power) / (measured_power + 1e-8)
        violations = data[power_diff_ratio > 0.1].index.tolist()

        if violations:
            issues.append(self._create_issue(
                ValidationResult.WARNING,
                f"전력 법칙 위배: 계산값과 측정값 차이 > 10%: {len(violations)}개",
                affected_rows=violations,
                affected_columns=['power_consumption', 'voltage', 'current'],
                suggested_action="센서 교정 또는 계산 로직 점검"
            ))

        return issues

    def _validate_temporal_order(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """시간 순서 검증"""
        issues = []

        # 타임스탬프를 datetime으로 변환
        try:
            timestamps = pd.to_datetime(data['timestamp'])
        except:
            issues.append(self._create_issue(
                ValidationResult.ERROR,
                "타임스탬프 형식 오류",
                affected_columns=['timestamp'],
                suggested_action="타임스탬프를 올바른 datetime 형식으로 변환"
            ))
            return issues

        # 시간 순서 확인
        if not timestamps.is_monotonic_increasing:
            # 순서가 잘못된 인덱스 찾기
            disordered_indices = []
            for i in range(1, len(timestamps)):
                if timestamps.iloc[i] < timestamps.iloc[i - 1]:
                    disordered_indices.extend([i - 1, i])

            issues.append(self._create_issue(
                ValidationResult.WARNING,
                f"시간 순서 오류: {len(set(disordered_indices))}개 레코드",
                affected_rows=list(set(disordered_indices)),
                affected_columns=['timestamp'],
                suggested_action="타임스탬프 순서로 데이터 정렬"
            ))

        # 미래 시간 확인
        future_timestamps = timestamps[timestamps > pd.Timestamp.now()].index.tolist()
        if future_timestamps:
            issues.append(self._create_issue(
                ValidationResult.WARNING,
                f"미래 타임스탬프: {len(future_timestamps)}개",
                affected_rows=future_timestamps,
                affected_columns=['timestamp'],
                suggested_action="타임스탬프 값 확인 및 수정"
            ))

        return issues


class StatisticalAnomalyRule(BaseValidationRule):
    """통계적 이상 검증 규칙"""

    def __init__(self, z_threshold: float = 3.0):
        super().__init__("statistical_anomaly")
        self.z_threshold = z_threshold

    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """통계적 이상 검증"""
        issues = []

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in ['hour', 'day_of_week', 'month']:  # 범주형 특성 제외
                continue

            # Z-score 계산
            z_scores = np.abs((data[col] - data[col].mean()) / (data[col].std() + 1e-8))
            anomaly_indices = data[z_scores > self.z_threshold].index.tolist()

            if anomaly_indices:
                anomaly_ratio = len(anomaly_indices) / len(data)

                level = ValidationResult.FAIL if anomaly_ratio > 0.1 else ValidationResult.WARNING
                issues.append(self._create_issue(
                    level,
                    f"컬럼 '{col}' 통계적 이상값: {len(anomaly_indices)}개 (Z-score > {self.z_threshold})",
                    affected_rows=anomaly_indices,
                    affected_columns=[col],
                    suggested_action="이상값 검토 및 필요시 제거 또는 보정"
                ))

        return issues


class DataValidator:
    """데이터 검증기"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.logger = get_logger("data_validator")
        self.config = get_config()
        self.validation_level = validation_level
        self.rules: List[BaseValidationRule] = []
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'average_processing_time': 0.0
        }

        # 기본 규칙 설정
        self._setup_default_rules()

    def _setup_default_rules(self):
        """기본 검증 규칙 설정"""
        if self.validation_level == ValidationLevel.BASIC:
            self.add_rule(SchemaValidationRule(['timestamp', 'machine_id']))

        elif self.validation_level == ValidationLevel.STANDARD:
            self.add_rule(SchemaValidationRule([
                'timestamp', 'machine_id', 'power_consumption'
            ]))
            self.add_rule(DataQualityRule())
            self.add_rule(BusinessLogicRule())

        elif self.validation_level == ValidationLevel.STRICT:
            self.add_rule(SchemaValidationRule([
                'timestamp', 'machine_id', 'sensor_id',
                'power_consumption', 'voltage', 'current'
            ]))
            self.add_rule(DataQualityRule(missing_threshold=0.05, duplicate_threshold=0.02))
            self.add_rule(BusinessLogicRule())
            self.add_rule(StatisticalAnomalyRule(z_threshold=2.5))

    def add_rule(self, rule: BaseValidationRule):
        """검증 규칙 추가"""
        self.rules.append(rule)
        self.logger.info(f"검증 규칙 추가: {rule.rule_name}")

    def remove_rule(self, rule_name: str):
        """검증 규칙 제거"""
        self.rules = [rule for rule in self.rules if rule.rule_name != rule_name]
        self.logger.info(f"검증 규칙 제거: {rule_name}")

    @log_performance
    def validate(self, data: pd.DataFrame) -> ValidationReport:
        """데이터 검증 수행"""
        start_time = datetime.now()
        all_issues = []

        # 각 규칙 적용
        for rule in self.rules:
            try:
                rule_issues = rule.validate(data)
                all_issues.extend(rule_issues)

            except Exception as e:
                self.logger.error(f"검증 규칙 오류 [{rule.rule_name}]: {e}")
                all_issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    rule_name=rule.rule_name,
                    message=f"검증 규칙 실행 오류: {str(e)}",
                    suggested_action="검증 규칙 설정 확인"
                ))

        # 통계 계산
        processing_time = (datetime.now() - start_time).total_seconds()
        issues_by_level = self._count_issues_by_level(all_issues)
        overall_score = self._calculate_overall_score(all_issues, len(data))

        # 통계 업데이트
        self.validation_stats['total_validations'] += 1
        if issues_by_level.get('fail', 0) == 0 and issues_by_level.get('error', 0) == 0:
            self.validation_stats['passed_validations'] += 1

        self.validation_stats['average_processing_time'] = (
                (self.validation_stats['average_processing_time'] *
                 (self.validation_stats['total_validations'] - 1) + processing_time) /
                self.validation_stats['total_validations']
        )

        # 보고서 생성
        report = ValidationReport(
            data_shape=data.shape,
            validation_level=self.validation_level,
            total_issues=len(all_issues),
            issues_by_level=issues_by_level,
            issues=all_issues,
            overall_score=overall_score,
            processing_time=processing_time
        )

        self.logger.info(f"데이터 검증 완료: {report.get_summary()}")
        return report

    def _count_issues_by_level(self, issues: List[ValidationIssue]) -> Dict[str, int]:
        """레벨별 이슈 수 계산"""
        counts = {level.value: 0 for level in ValidationResult}

        for issue in issues:
            counts[issue.level.value] += 1

        return counts

    def _calculate_overall_score(self, issues: List[ValidationIssue], data_size: int) -> float:
        """전체 품질 점수 계산 (0-100)"""
        if not issues:
            return 100.0

        total_severity = sum(issue.severity_score for issue in issues)
        max_possible_severity = data_size * 1.0  # 최대 심각도

        # 점수 계산 (100 - 심각도 비율 * 100)
        score = max(0, 100 - (total_severity / max_possible_severity) * 100)
        return round(score, 2)

    def get_validation_statistics(self) -> Dict:
        """검증 통계 반환"""
        success_rate = (
                self.validation_stats['passed_validations'] /
                max(self.validation_stats['total_validations'], 1) * 100
        )

        return {
            **self.validation_stats,
            'success_rate': round(success_rate, 2),
            'total_rules': len(self.rules),
            'validation_level': self.validation_level.value
        }


class RealTimeValidator:
    """실시간 데이터 검증기"""

    def __init__(self, validator: DataValidator, buffer_size: int = 1000):
        self.logger = get_logger("realtime_validator")
        self.validator = validator
        self.buffer_size = buffer_size
        self.validation_buffer = []
        self.alert_threshold = 0.7  # 심각도 임계값
        self.callbacks: List[Callable] = []

    def add_alert_callback(self, callback: Callable[[ValidationReport], None]):
        """알림 콜백 추가"""
        self.callbacks.append(callback)

    def validate_stream(self, data: Union[pd.DataFrame, Dict]) -> Optional[ValidationReport]:
        """스트림 데이터 검증"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # 버퍼에 추가
        self.validation_buffer.append(data)

        # 버퍼 크기 제한
        if len(self.validation_buffer) > self.buffer_size:
            self.validation_buffer.pop(0)

        # 주기적 검증 (10개 데이터마다)
        if len(self.validation_buffer) % 10 == 0:
            combined_data = pd.concat(self.validation_buffer, ignore_index=True)
            report = self.validator.validate(combined_data)

            # 심각한 이슈 발생시 알림
            if report.overall_score < (1 - self.alert_threshold) * 100:
                self._trigger_alerts(report)

            return report

        return None

    def _trigger_alerts(self, report: ValidationReport):
        """알림 트리거"""
        self.logger.warning(f"데이터 품질 경고: 점수 {report.overall_score}")

        for callback in self.callbacks:
            try:
                callback(report)
            except Exception as e:
                self.logger.error(f"알림 콜백 오류: {e}")


# 팩토리 함수들
def create_validator(level: ValidationLevel = ValidationLevel.STANDARD) -> DataValidator:
    """검증기 생성"""
    return DataValidator(level)


def create_custom_validator(rules: List[BaseValidationRule]) -> DataValidator:
    """사용자 정의 검증기 생성"""
    validator = DataValidator(ValidationLevel.CUSTOM)
    validator.rules.clear()  # 기본 규칙 제거

    for rule in rules:
        validator.add_rule(rule)

    return validator


def create_realtime_validator(level: ValidationLevel = ValidationLevel.STANDARD) -> RealTimeValidator:
    """실시간 검증기 생성"""
    validator = create_validator(level)
    return RealTimeValidator(validator)


# 사용 예시
if __name__ == "__main__":
    # 검증기 생성
    validator = create_validator(ValidationLevel.STANDARD)

    # 샘플 데이터
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'machine_id': ['machine_1'] * 50 + ['machine_2'] * 50,
        'power_consumption': np.random.normal(100, 20, 100),
        'voltage': np.random.normal(220, 10, 100),
        'current': np.random.normal(0.5, 0.1, 100)
    })

    # 검증 수행
    report = validator.validate(sample_data)

    print("=== 검증 보고서 ===")
    print(f"데이터 형태: {report.data_shape}")
    print(f"유효성: {'통과' if report.is_valid() else '실패'}")
    print(f"전체 점수: {report.overall_score}/100")
    print(f"총 이슈: {report.total_issues}")
    print(f"레벨별 이슈: {report.issues_by_level}")

    # JSON 보고서 출력
    print("\n=== JSON 보고서 ===")
    print(report.to_json())