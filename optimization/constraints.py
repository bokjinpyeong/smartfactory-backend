"""
스마트팩토리 에너지 관리 시스템 - 제약 조건 관리 모듈

다양한 제약 조건 관리 및 검증
- 전력 제약 (피크 전력, 총 전력)
- 시간 제약 (납기일, 가용 시간)
- 자원 제약 (기계 용량, 작업자)
- 사용자 정의 제약 조건
- 제약 위반 감지 및 해결책 제안
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import math

# Core 모듈
from core.config import get_config
from core.logger import get_logger
from core.exceptions import (
    ConstraintViolationError, PeakPowerExceededError,
    OptimizationException, safe_execute
)


class ConstraintType(Enum):
    """제약 조건 타입"""
    POWER = "power"           # 전력 제약
    TIME = "time"             # 시간 제약
    RESOURCE = "resource"     # 자원 제약
    CAPACITY = "capacity"     # 용량 제약
    PRECEDENCE = "precedence" # 선후행 제약
    CUSTOM = "custom"         # 사용자 정의


class ViolationSeverity(Enum):
    """위반 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConstraintViolation:
    """제약 조건 위반 정보"""
    constraint_name: str
    constraint_type: ConstraintType
    severity: ViolationSeverity
    violation_value: float
    limit_value: float
    violation_ratio: float
    affected_entities: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            'constraint_name': self.constraint_name,
            'constraint_type': self.constraint_type.value,
            'severity': self.severity.value,
            'violation_value': self.violation_value,
            'limit_value': self.limit_value,
            'violation_ratio': self.violation_ratio,
            'affected_entities': self.affected_entities,
            'suggested_actions': self.suggested_actions,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ConstraintCheckResult:
    """제약 조건 검사 결과"""
    is_feasible: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)
    total_violations: int = 0
    max_violation_ratio: float = 0.0

    def __post_init__(self):
        self.total_violations = len(self.violations)
        self.violated_constraints = [v.constraint_name for v in self.violations]
        self.max_violation_ratio = max([v.violation_ratio for v in self.violations], default=0.0)

    def get_critical_violations(self) -> List[ConstraintViolation]:
        """치명적 위반 목록"""
        return [v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]

    def get_summary(self) -> Dict:
        """요약 정보"""
        severity_counts = {}
        for severity in ViolationSeverity:
            severity_counts[severity.value] = sum(
                1 for v in self.violations if v.severity == severity
            )

        return {
            'is_feasible': self.is_feasible,
            'total_violations': self.total_violations,
            'max_violation_ratio': self.max_violation_ratio,
            'severity_breakdown': severity_counts,
            'violated_constraints': self.violated_constraints
        }


class BaseConstraint(ABC):
    """기본 제약 조건 클래스"""

    def __init__(self,
                 constraint_name: str,
                 constraint_type: ConstraintType,
                 is_hard: bool = True,
                 penalty_weight: float = 1.0):
        self.constraint_name = constraint_name
        self.constraint_type = constraint_type
        self.is_hard = is_hard  # Hard constraint vs Soft constraint
        self.penalty_weight = penalty_weight
        self.logger = get_logger(f"constraint.{constraint_name}")

        # 통계
        self.stats = {
            'total_checks': 0,
            'violations': 0,
            'violation_rate': 0.0
        }

    @abstractmethod
    def check(self, *args, **kwargs) -> ConstraintCheckResult:
        """제약 조건 확인"""
        pass

    @abstractmethod
    def get_penalty(self, violation_ratio: float) -> float:
        """위반 패널티 계산"""
        pass

    def _create_violation(self,
                         violation_value: float,
                         limit_value: float,
                         affected_entities: List[str] = None,
                         suggested_actions: List[str] = None) -> ConstraintViolation:
        """제약 위반 객체 생성"""
        violation_ratio = abs(violation_value - limit_value) / max(abs(limit_value), 1e-8)

        # 심각도 결정
        if violation_ratio > 0.5:
            severity = ViolationSeverity.CRITICAL
        elif violation_ratio > 0.2:
            severity = ViolationSeverity.HIGH
        elif violation_ratio > 0.1:
            severity = ViolationSeverity.MEDIUM
        else:
            severity = ViolationSeverity.LOW

        return ConstraintViolation(
            constraint_name=self.constraint_name,
            constraint_type=self.constraint_type,
            severity=severity,
            violation_value=violation_value,
            limit_value=limit_value,
            violation_ratio=violation_ratio,
            affected_entities=affected_entities or [],
            suggested_actions=suggested_actions or []
        )

    def _update_stats(self, has_violation: bool):
        """통계 업데이트"""
        self.stats['total_checks'] += 1
        if has_violation:
            self.stats['violations'] += 1

        self.stats['violation_rate'] = (
            self.stats['violations'] / self.stats['total_checks'] * 100
        )


class PowerConstraint(BaseConstraint):
    """전력 제약 조건"""

    def __init__(self,
                 constraint_name: str = "power_limit",
                 max_power: float = 1000.0,
                 safety_margin: float = 0.1):
        super().__init__(constraint_name, ConstraintType.POWER)
        self.max_power = max_power
        self.safety_margin = safety_margin
        self.effective_limit = max_power * (1 - safety_margin)

    def check(self, power_schedule: Dict[datetime, float]) -> ConstraintCheckResult:
        """전력 스케줄 검사"""
        violations = []

        for timestamp, power_usage in power_schedule.items():
            if power_usage > self.effective_limit:
                violation = self._create_violation(
                    violation_value=power_usage,
                    limit_value=self.effective_limit,
                    affected_entities=[f"time_{timestamp}"],
                    suggested_actions=[
                        "전력 사용량이 높은 작업을 다른 시간대로 이동",
                        "동시 작업 수 제한",
                        "저전력 모드 사용 고려"
                    ]
                )
                violations.append(violation)

        is_feasible = len(violations) == 0
        self._update_stats(not is_feasible)

        return ConstraintCheckResult(
            is_feasible=is_feasible,
            violations=violations
        )

    def check_job_power(self, job, start_time: datetime, existing_schedule: Dict = None) -> ConstraintCheckResult:
        """단일 작업의 전력 제약 확인"""
        if existing_schedule is None:
            existing_schedule = {}

        # 작업 실행 시간 동안의 전력 사용량 계산
        job_end_time = start_time + timedelta(minutes=job.processing_time)
        current_time = start_time
        violations = []

        while current_time < job_end_time:
            # 해당 시간의 기존 전력 사용량
            existing_power = existing_schedule.get(current_time, 0.0)
            total_power = existing_power + job.power_requirement

            if total_power > self.effective_limit:
                violation = self._create_violation(
                    violation_value=total_power,
                    limit_value=self.effective_limit,
                    affected_entities=[job.job_id],
                    suggested_actions=[
                        f"작업 {job.job_id}을 다른 시간대로 이동",
                        "전력 요구량이 낮은 시간대 선택",
                        "기존 작업과의 중복 최소화"
                    ]
                )
                violations.append(violation)
                break

            current_time += timedelta(hours=1)

        is_feasible = len(violations) == 0
        self._update_stats(not is_feasible)

        return ConstraintCheckResult(
            is_feasible=is_feasible,
            violations=violations
        )

    def get_penalty(self, violation_ratio: float) -> float:
        """전력 위반 패널티"""
        # 제곱 패널티 (위반이 클수록 급격히 증가)
        return self.penalty_weight * (violation_ratio ** 2) * 1000


class TimeConstraint(BaseConstraint):
    """시간 제약 조건"""

    def __init__(self, constraint_name: str = "time_constraint"):
        super().__init__(constraint_name, ConstraintType.TIME)

    def check(self, job, start_time: datetime, finish_time: datetime) -> ConstraintCheckResult:
        """시간 제약 확인"""
        violations = []

        # 납기일 확인
        if finish_time > job.due_date:
            delay_hours = (finish_time - job.due_date).total_seconds() / 3600
            violation = self._create_violation(
                violation_value=finish_time.timestamp(),
                limit_value=job.due_date.timestamp(),
                affected_entities=[job.job_id],
                suggested_actions=[
                    "더 빠른 시작 시간 선택",
                    "작업 우선순위 조정",
                    "병렬 처리 가능성 검토"
                ]
            )
            violations.append(violation)

        # 최조 시작 시간 확인
        if start_time < job.earliest_start:
            early_hours = (job.earliest_start - start_time).total_seconds() / 3600
            violation = self._create_violation(
                violation_value=start_time.timestamp(),
                limit_value=job.earliest_start.timestamp(),
                affected_entities=[job.job_id],
                suggested_actions=[
                    f"시작 시간을 {job.earliest_start} 이후로 조정"
                ]
            )
            violations.append(violation)

        # 최대 완료 시간 확인 (latest_finish가 있는 경우)
        if hasattr(job, 'latest_finish') and job.latest_finish:
            if finish_time > job.latest_finish:
                violation = self._create_violation(
                    violation_value=finish_time.timestamp(),
                    limit_value=job.latest_finish.timestamp(),
                    affected_entities=[job.job_id],
                    suggested_actions=[
                        "더 빠른 시작 시간 필요",
                        "처리 시간 단축 방안 검토"
                    ]
                )
                violations.append(violation)

        is_feasible = len(violations) == 0
        self._update_stats(not is_feasible)

        return ConstraintCheckResult(
            is_feasible=is_feasible,
            violations=violations
        )

    def get_penalty(self, violation_ratio: float) -> float:
        """시간 위반 패널티"""
        # 지수 패널티 (지연이 클수록 매우 높은 비용)
        return self.penalty_weight * math.exp(violation_ratio) * 500


class ResourceConstraint(BaseConstraint):
    """자원 제약 조건"""

    def __init__(self,
                 constraint_name: str = "resource_constraint",
                 machine_capacities: Dict[str, int] = None):
        super().__init__(constraint_name, ConstraintType.RESOURCE)
        self.machine_capacities = machine_capacities or {}

    def check(self, machine_schedule: Dict[str, List]) -> ConstraintCheckResult:
        """기계별 자원 제약 확인"""
        violations = []

        for machine_id, jobs in machine_schedule.items():
            max_capacity = self.machine_capacities.get(machine_id, 1)

            # 시간별 동시 작업 수 확인
            time_slots = {}

            for job in jobs:
                if hasattr(job, 'actual_start') and job.actual_start:
                    start = job.actual_start
                    end = job.actual_finish or (start + timedelta(minutes=job.processing_time))

                    # 시간 슬롯별 작업 수 카운트
                    current = start
                    while current < end:
                        hour_key = current.replace(minute=0, second=0, microsecond=0)
                        if hour_key not in time_slots:
                            time_slots[hour_key] = 0
                        time_slots[hour_key] += 1
                        current += timedelta(hours=1)

            # 용량 초과 확인
            for time_slot, job_count in time_slots.items():
                if job_count > max_capacity:
                    violation = self._create_violation(
                        violation_value=job_count,
                        limit_value=max_capacity,
                        affected_entities=[machine_id, str(time_slot)],
                        suggested_actions=[
                            f"기계 {machine_id}의 동시 작업 수 제한",
                            "작업을 다른 시간대로 분산",
                            "추가 기계 자원 확보 고려"
                        ]
                    )
                    violations.append(violation)

        is_feasible = len(violations) == 0
        self._update_stats(not is_feasible)

        return ConstraintCheckResult(
            is_feasible=is_feasible,
            violations=violations
        )

    def get_penalty(self, violation_ratio: float) -> float:
        """자원 위반 패널티"""
        return self.penalty_weight * violation_ratio * 300


class PrecedenceConstraint(BaseConstraint):
    """선후행 제약 조건"""

    def __init__(self,
                 constraint_name: str = "precedence_constraint",
                 precedence_rules: List[Tuple[str, str]] = None):
        super().__init__(constraint_name, ConstraintType.PRECEDENCE)
        self.precedence_rules = precedence_rules or []  # (predecessor, successor) 쌍

    def check(self, job_schedule: Dict[str, dict]) -> ConstraintCheckResult:
        """선후행 관계 확인"""
        violations = []

        for predecessor_id, successor_id in self.precedence_rules:
            if predecessor_id in job_schedule and successor_id in job_schedule:
                pred_job = job_schedule[predecessor_id]
                succ_job = job_schedule[successor_id]

                # 선행 작업의 완료 시간과 후행 작업의 시작 시간 비교
                if (pred_job.get('finish_time') and succ_job.get('start_time')):
                    pred_finish = pred_job['finish_time']
                    succ_start = succ_job['start_time']

                    if pred_finish > succ_start:
                        violation = self._create_violation(
                            violation_value=succ_start.timestamp(),
                            limit_value=pred_finish.timestamp(),
                            affected_entities=[predecessor_id, successor_id],
                            suggested_actions=[
                                f"작업 {successor_id}의 시작을 {pred_finish} 이후로 조정",
                                f"작업 {predecessor_id}의 완료 시간 단축",
                                "선후행 관계 재검토"
                            ]
                        )
                        violations.append(violation)

        is_feasible = len(violations) == 0
        self._update_stats(not is_feasible)

        return ConstraintCheckResult(
            is_feasible=is_feasible,
            violations=violations
        )

    def get_penalty(self, violation_ratio: float) -> float:
        """선후행 위반 패널티"""
        return self.penalty_weight * violation_ratio * 800


class CustomConstraint(BaseConstraint):
    """사용자 정의 제약 조건"""

    def __init__(self,
                 constraint_name: str,
                 check_function: Callable,
                 penalty_function: Callable = None):
        super().__init__(constraint_name, ConstraintType.CUSTOM)
        self.check_function = check_function
        self.penalty_function = penalty_function or self._default_penalty

    def check(self, *args, **kwargs) -> ConstraintCheckResult:
        """사용자 정의 검사 함수 실행"""
        try:
            result = self.check_function(*args, **kwargs)

            # 결과가 ConstraintCheckResult가 아닌 경우 변환
            if not isinstance(result, ConstraintCheckResult):
                if isinstance(result, bool):
                    result = ConstraintCheckResult(is_feasible=result)
                else:
                    result = ConstraintCheckResult(is_feasible=False)

            self._update_stats(not result.is_feasible)
            return result

        except Exception as e:
            self.logger.error(f"사용자 정의 제약 조건 실행 오류: {e}")
            return ConstraintCheckResult(
                is_feasible=False,
                violations=[
                    ConstraintViolation(
                        constraint_name=self.constraint_name,
                        constraint_type=self.constraint_type,
                        severity=ViolationSeverity.HIGH,
                        violation_value=1.0,
                        limit_value=0.0,
                        violation_ratio=1.0,
                        suggested_actions=["제약 조건 함수 오류 수정 필요"]
                    )
                ]
            )

    def get_penalty(self, violation_ratio: float) -> float:
        """사용자 정의 패널티 함수"""
        return self.penalty_function(violation_ratio)

    def _default_penalty(self, violation_ratio: float) -> float:
        """기본 패널티 함수"""
        return self.penalty_weight * violation_ratio * 100


class ConstraintManager:
    """제약 조건 관리자"""

    def __init__(self):
        self.logger = get_logger("constraint_manager")
        self.config = get_config()
        self.constraints: Dict[str, BaseConstraint] = {}

        # 기본 제약 조건 설정
        self._setup_default_constraints()

        # 통계
        self.global_stats = {
            'total_constraint_checks': 0,
            'total_violations': 0,
            'violation_rate': 0.0
        }

    def _setup_default_constraints(self):
        """기본 제약 조건 설정"""
        # 전력 제약
        max_power = self.config.power.peak_power_limit
        safety_margin = self.config.power.safety_margin

        power_constraint = PowerConstraint(
            constraint_name="peak_power_limit",
            max_power=max_power,
            safety_margin=safety_margin
        )
        self.add_constraint(power_constraint)

        # 시간 제약
        time_constraint = TimeConstraint("delivery_time")
        self.add_constraint(time_constraint)

        # 자원 제약 (기본 기계 용량: 1)
        resource_constraint = ResourceConstraint(
            "machine_capacity",
            machine_capacities={"default": 1}
        )
        self.add_constraint(resource_constraint)

    def add_constraint(self, constraint: BaseConstraint):
        """제약 조건 추가"""
        self.constraints[constraint.constraint_name] = constraint
        self.logger.info(f"제약 조건 추가: {constraint.constraint_name}")

    def remove_constraint(self, constraint_name: str):
        """제약 조건 제거"""
        if constraint_name in self.constraints:
            del self.constraints[constraint_name]
            self.logger.info(f"제약 조건 제거: {constraint_name}")

    def check_constraints(self, *args, **kwargs) -> ConstraintCheckResult:
        """모든 제약 조건 확인"""
        all_violations = []
        total_checks = 0

        for constraint_name, constraint in self.constraints.items():
            try:
                # 제약 조건별로 적절한 인자 전달 (간단한 예시)
                if isinstance(constraint, PowerConstraint):
                    result = self._check_power_constraint(constraint, *args, **kwargs)
                elif isinstance(constraint, TimeConstraint):
                    result = self._check_time_constraint(constraint, *args, **kwargs)
                elif isinstance(constraint, ResourceConstraint):
                    result = self._check_resource_constraint(constraint, *args, **kwargs)
                else:
                    result = constraint.check(*args, **kwargs)

                all_violations.extend(result.violations)
                total_checks += 1

            except Exception as e:
                self.logger.error(f"제약 조건 '{constraint_name}' 확인 오류: {e}")
                # 오류가 발생한 제약 조건은 위반으로 처리
                error_violation = ConstraintViolation(
                    constraint_name=constraint_name,
                    constraint_type=constraint.constraint_type,
                    severity=ViolationSeverity.HIGH,
                    violation_value=1.0,
                    limit_value=0.0,
                    violation_ratio=1.0,
                    suggested_actions=[f"제약 조건 '{constraint_name}' 설정 확인"]
                )
                all_violations.append(error_violation)

        # 글로벌 통계 업데이트
        self.global_stats['total_constraint_checks'] += total_checks
        self.global_stats['total_violations'] += len(all_violations)

        if self.global_stats['total_constraint_checks'] > 0:
            self.global_stats['violation_rate'] = (
                self.global_stats['total_violations'] /
                self.global_stats['total_constraint_checks'] * 100
            )

        return ConstraintCheckResult(
            is_feasible=len(all_violations) == 0,
            violations=all_violations
        )

    def _check_power_constraint(self, constraint: PowerConstraint, *args, **kwargs) -> ConstraintCheckResult:
        """전력 제약 확인"""
        # 작업과 시작 시간이 제공된 경우
        if len(args) >= 2:
            job = args[0]
            start_time = args[1]
            existing_schedule = kwargs.get('existing_schedule', {})
            return constraint.check_job_power(job, start_time, existing_schedule)

        # 전력 스케줄이 제공된 경우
        power_schedule = kwargs.get('power_schedule', {})
        return constraint.check(power_schedule)

    def _check_time_constraint(self, constraint: TimeConstraint, *args, **kwargs) -> ConstraintCheckResult:
        """시간 제약 확인"""
        if len(args) >= 3:
            job = args[0]
            start_time = args[1]
            finish_time = args[2]
            return constraint.check(job, start_time, finish_time)
        elif len(args) >= 2:
            job = args[0]
            start_time = args[1]
            finish_time = start_time + timedelta(minutes=job.processing_time)
            return constraint.check(job, start_time, finish_time)

        return ConstraintCheckResult(is_feasible=True)

    def _check_resource_constraint(self, constraint: ResourceConstraint, *args, **kwargs) -> ConstraintCheckResult:
        """자원 제약 확인"""
        machine_schedule = kwargs.get('machine_schedule', {})
        return constraint.check(machine_schedule)

    def calculate_total_penalty(self, violations: List[ConstraintViolation]) -> float:
        """총 패널티 계산"""
        total_penalty = 0.0

        for violation in violations:
            constraint = self.constraints.get(violation.constraint_name)
            if constraint:
                penalty = constraint.get_penalty(violation.violation_ratio)
                total_penalty += penalty

        return total_penalty

    def get_constraint_suggestions(self, violations: List[ConstraintViolation]) -> List[str]:
        """제약 위반 해결 제안"""
        suggestions = []

        # 심각도별로 정렬
        sorted_violations = sorted(violations,
                                 key=lambda v: v.severity.value,
                                 reverse=True)

        for violation in sorted_violations:
            suggestions.extend(violation.suggested_actions)

        # 중복 제거 및 우선순위 정렬
        unique_suggestions = list(dict.fromkeys(suggestions))

        return unique_suggestions

    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        constraint_stats = {}

        for name, constraint in self.constraints.items():
            constraint_stats[name] = constraint.stats.copy()

        return {
            'global_stats': self.global_stats.copy(),
            'constraint_stats': constraint_stats,
            'total_constraints': len(self.constraints)
        }

    def export_configuration(self) -> Dict:
        """제약 조건 설정 내보내기"""
        config = {
            'constraints': {},
            'global_settings': {
                'total_constraints': len(self.constraints)
            }
        }

        for name, constraint in self.constraints.items():
            constraint_config = {
                'type': constraint.constraint_type.value,
                'is_hard': constraint.is_hard,
                'penalty_weight': constraint.penalty_weight
            }

            # 타입별 추가 설정
            if isinstance(constraint, PowerConstraint):
                constraint_config.update({
                    'max_power': constraint.max_power,
                    'safety_margin': constraint.safety_margin
                })
            elif isinstance(constraint, ResourceConstraint):
                constraint_config.update({
                    'machine_capacities': constraint.machine_capacities
                })
            elif isinstance(constraint, PrecedenceConstraint):
                constraint_config.update({
                    'precedence_rules': constraint.precedence_rules
                })

            config['constraints'][name] = constraint_config

        return config


# 팩토리 함수들
def create_power_constraint(max_power: float = 1000.0, safety_margin: float = 0.1) -> PowerConstraint:
    """전력 제약 생성"""
    return PowerConstraint(max_power=max_power, safety_margin=safety_margin)


def create_time_constraint() -> TimeConstraint:
    """시간 제약 생성"""
    return TimeConstraint()


def create_resource_constraint(machine_capacities: Dict[str, int] = None) -> ResourceConstraint:
    """자원 제약 생성"""
    return ResourceConstraint(machine_capacities=machine_capacities)


def create_precedence_constraint(precedence_rules: List[Tuple[str, str]]) -> PrecedenceConstraint:
    """선후행 제약 생성"""
    return PrecedenceConstraint(precedence_rules=precedence_rules)


def create_custom_constraint(name: str, check_function: Callable, penalty_function: Callable = None) -> CustomConstraint:
    """사용자 정의 제약 생성"""
    return CustomConstraint(name, check_function, penalty_function)


def create_constraint_manager() -> ConstraintManager:
    """제약 관리자 생성"""
    return ConstraintManager()


# 사용 예시
if __name__ == "__main__":
    # 제약 관리자 생성
    manager = create_constraint_manager()

    # 사용자 정의 제약 추가
    def custom_check_function(jobs):
        """예시: 총 작업 수 제한"""
        if len(jobs) > 10:
            return False
        return True

    custom_constraint = create_custom_constraint(
        "max_jobs",
        custom_check_function
    )
    manager.add_constraint(custom_constraint)

    # 선후행 제약 추가
    precedence_rules = [("job_1", "job_2"), ("job_2", "job_3")]
    precedence_constraint = create_precedence_constraint(precedence_rules)
    manager.add_constraint(precedence_constraint)

    # 샘플 데이터로 테스트
    from datetime import datetime, timedelta

    class MockJob:
        def __init__(self, job_id, power_requirement, processing_time, due_date):
            self.job_id = job_id
            self.power_requirement = power_requirement
            self.processing_time = processing_time
            self.due_date = due_date
            self.earliest_start = datetime.now()

    # 테스트 작업
    test_job = MockJob(
        job_id="test_job",
        power_requirement=800.0,  # 높은 전력 (제약 위반 가능성)
        processing_time=120,      # 2시간
        due_date=datetime.now() + timedelta(hours=1)  # 1시간 후 납기
    )

    start_time = datetime.now()
    finish_time = start_time + timedelta(minutes=test_job.processing_time)

    # 제약 조건 확인
    result = manager.check_constraints(
        test_job,
        start_time,
        finish_time,
        power_schedule={start_time: 500.0}  # 기존 전력 사용량
    )

    print("=== 제약 조건 확인 결과 ===")
    print(f"실행 가능: {result.is_feasible}")
    print(f"총 위반: {result.total_violations}")
    print(f"최대 위반 비율: {result.max_violation_ratio:.2%}")

    if result.violations:
        print("\n=== 위반 상세 ===")
        for violation in result.violations:
            print(f"- {violation.constraint_name}: {violation.severity.value}")
            print(f"  위반값: {violation.violation_value:.2f}, 제한값: {violation.limit_value:.2f}")
            print(f"  제안사항: {violation.suggested_actions}")
            print()

    # 통계 정보
    print("=== 통계 정보 ===")
    stats = manager.get_statistics()
    print(f"글로벌 통계: {stats['global_stats']}")
    print(f"제약 조건별 통계: {stats['constraint_stats']}")

    # 설정 내보내기
    config = manager.export_configuration()
    print(f"\n=== 설정 ===")
    print(json.dumps(config, indent=2, ensure_ascii=False))