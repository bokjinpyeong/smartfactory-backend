@dataclass
class ScheduleResult:
    """스케줄링 결과"""
    scheduled_jobs: List[Job]
    total_cost: float
    peak_power: float
    makespan: float  # hours
    total_delay: float  # hours
    utilization_rate: float
    objective_value: float
    computation_time: float
    is_feasible: bool
    constraints_violated: List[str] = field(default_factory=list)
    optimization_details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> Dict:
        """결과 요약"""
        return {
            'total_jobs': len(self.scheduled_jobs),
            'feasible': self.is_feasible,
            'total_cost': round(self.total_cost, 2),
            'peak_power': round(self.peak_power, 2),
            'makespan_hours': round(self.makespan, 2),
            'total_delay_hours': round(self.total_delay, 2),
            'utilization_rate': round(self.utilization_rate, 2),
            'objective_value': round(self.objective_value, 2),
            'computation_time': round(self.computation_time, 3),
            'constraints_violated': len(self.constraints_violated),
            'timestamp': self.timestamp.isoformat()
        }

    def to_gantt_data(self) -> List[Dict]:
        """간트 차트용 데이터 변환"""
        gantt_data = []

        for job in self.scheduled_jobs:
            if job.actual_start and job.status != JobStatus.PENDING:
                gantt_data.append({
                    'job_id': job.job_id,
                    'machine_id': job.machine_id,
                    'start': job.actual_start.isoformat(),
                    'finish': job.actual_finish.isoformat() if job.actual_finish else None,
                    'duration': job.processing_time,
                    'power': job.power_requirement,
                    'status': job.status.value,
                    'priority': job.priority
                })

        return gantt_data


class BaseScheduler(ABC):
    """기본 스케줄러 클래스"""

    def __init__(self, scheduler_name: str):
        self.scheduler_name = scheduler_name
        self.logger = get_logger(f"scheduler.{scheduler_name}")
        self.config = get_config()

        # TOU 요금제 모델
        self.tou_model = TOUPricingModel()

        # 제약 조건 관리자
        self.constraint_manager = ConstraintManager()

        # 스케줄링 통계
        self.stats = {
            'total_scheduling': 0,
            'successful_scheduling': 0,
            'average_computation_time': 0.0,
            'average_cost_savings': 0.0
        }

    @abstractmethod
    def schedule(self, jobs: List[Job], objective: SchedulingObjective) -> ScheduleResult:
        """스케줄링 수행"""
        pass

    def add_constraint(self, constraint):
        """제약 조건 추가"""
        self.constraint_manager.add_constraint(constraint)

    def _validate_jobs(self, jobs: List[Job]) -> List[str]:
        """작업 유효성 검증"""
        issues = []

        for job in jobs:
            if job.processing_time <= 0:
                issues.append(f"작업 {job.job_id}: 처리 시간이 0 이하")

            if job.power_requirement < 0:
                issues.append(f"작업 {job.job_id}: 전력 요구량이 음수")

            if job.due_date < datetime.now():
                issues.append(f"작업 {job.job_id}: 납기일이 과거")

            if job.earliest_start and job.latest_finish:
                if job.earliest_start >= job.latest_finish:
                    issues.append(f"작업 {job.job_id}: 시작 시간이 완료 시간 이후")

        return issues

    def _calculate_schedule_metrics(self, jobs: List[Job]) -> Tuple[float, float, float, float]:
        """스케줄 메트릭 계산"""
        if not jobs:
            return 0.0, 0.0, 0.0, 0.0

        # 총 비용 계산
        total_cost = sum(job.cost for job in jobs)

        # 피크 전력 계산
        power_timeline = {}
        for job in jobs:
            if job.actual_start and job.actual_finish:
                start_hour = job.actual_start.hour
                end_hour = job.actual_finish.hour

                for hour in range(start_hour, end_hour + 1):
                    if hour not in power_timeline:
                        power_timeline[hour] = 0
                    power_timeline[hour] += job.power_requirement

        peak_power = max(power_timeline.values()) if power_timeline else 0.0

        # 총 지연 시간 계산
        total_delay = 0.0
        for job in jobs:
            if job.actual_finish and job.actual_finish > job.due_date:
                delay = (job.actual_finish - job.due_date).total_seconds() / 3600
                total_delay += delay

        # 완료 시간 (makespan) 계산
        if jobs and any(job.actual_finish for job in jobs):
            latest_finish = max(job.actual_finish for job in jobs if job.actual_finish)
            earliest_start = min(job.actual_start for job in jobs if job.actual_start)
            makespan = (latest_finish - earliest_start).total_seconds() / 3600
        else:
            makespan = 0.0

        return total_cost, peak_power, total_delay, makespan


class GreedyScheduler(BaseScheduler):
    """그리디 스케줄러 (빠른 근사해)"""

    def __init__(self):
        super().__init__("greedy_scheduler")

    @log_performance
    def schedule(self, jobs: List[Job], objective: SchedulingObjective) -> ScheduleResult:
        """그리디 스케줄링"""
        start_time = datetime.now()

        # 입력 검증
        validation_issues = self._validate_jobs(jobs)
        if validation_issues:
            raise SchedulingError(f"작업 검증 실패: {validation_issues}")

        # 작업 복사 및 정렬
        scheduled_jobs = [job for job in jobs]

        # 목적에 따른 정렬
        if objective == SchedulingObjective.MINIMIZE_COST:
            scheduled_jobs.sort(key=self._cost_priority_key)
        elif objective == SchedulingObjective.MINIMIZE_PEAK:
            scheduled_jobs.sort(key=lambda j: j.power_requirement)
        elif objective == SchedulingObjective.MINIMIZE_MAKESPAN:
            scheduled_jobs.sort(key=lambda j: (j.priority, j.due_date))
        else:
            scheduled_jobs.sort(key=lambda j: (j.priority, j.due_date))

        # 기계별 스케줄 추적
        machine_schedules = {}
        constraints_violated = []

        # 순차적 배치
        for job in scheduled_jobs:
            machine_id = job.machine_id

            if machine_id not in machine_schedules:
                machine_schedules[machine_id] = []

            # 가장 빠른 가능한 시작 시간 찾기
            earliest_possible = self._find_earliest_slot(
                job, machine_schedules[machine_id]
            )

            # 제약 조건 확인
            if not self._check_constraints(job, earliest_possible, constraints_violated):
                # 제약 위반시 다음 가능한 시간으로 이동
                earliest_possible = self._find_next_feasible_time(job, earliest_possible)

            # 작업 배치
            job.actual_start = earliest_possible
            job.actual_finish = earliest_possible + timedelta(minutes=job.processing_time)
            job.status = JobStatus.SCHEDULED

            # 비용 계산
            job.cost = self._calculate_job_cost(job)

            machine_schedules[machine_id].append(job)

        # 메트릭 계산
        total_cost, peak_power, total_delay, makespan = self._calculate_schedule_metrics(scheduled_jobs)

        # 가동률 계산
        total_processing_time = sum(job.processing_time for job in scheduled_jobs) / 60  # hours
        utilization_rate = total_processing_time / (makespan * len(machine_schedules)) if makespan > 0 else 0

        # 목적 함수 값 계산
        objective_value = self._calculate_objective_value(
            objective, total_cost, peak_power, makespan, total_delay
        )

        computation_time = (datetime.now() - start_time).total_seconds()

        # 통계 업데이트
        self._update_stats(computation_time, len(constraints_violated) == 0)

        return ScheduleResult(
            scheduled_jobs=scheduled_jobs,
            total_cost=total_cost,
            peak_power=peak_power,
            makespan=makespan,
            total_delay=total_delay,
            utilization_rate=utilization_rate,
            objective_value=objective_value,
            computation_time=computation_time,
            is_feasible=len(constraints_violated) == 0,
            constraints_violated=constraints_violated,
            optimization_details={
                'algorithm': 'greedy',
                'sorting_criterion': objective.value,
                'total_machines': len(machine_schedules)
            }
        )

    def _cost_priority_key(self, job: Job) -> Tuple:
        """비용 우선순위 키"""
        # TOU 요금대를 고려한 비용 추정
        avg_rate = self.tou_model.get_average_rate(job.earliest_start, job.processing_time)
        estimated_cost = job.power_requirement * (job.processing_time / 60) * avg_rate

        return (job.priority, estimated_cost, job.due_date)

    def _find_earliest_slot(self, job: Job, machine_schedule: List[Job]) -> datetime:
        """기계에서 가장 빠른 가능한 시작 시간 찾기"""
        earliest = max(job.earliest_start, datetime.now())

        if not machine_schedule:
            return earliest

        # 기존 작업들과 겹치지 않는 시간 찾기
        machine_schedule.sort(key=lambda j: j.actual_start)

        for scheduled_job in machine_schedule:
            if scheduled_job.actual_finish <= earliest:
                continue

            if scheduled_job.actual_start <= earliest < scheduled_job.actual_finish:
                earliest = scheduled_job.actual_finish

        return earliest

    def _check_constraints(self, job: Job, start_time: datetime, violations: List[str]) -> bool:
        """제약 조건 확인"""
        # 기본 시간 제약
        finish_time = start_time + timedelta(minutes=job.processing_time)

        if finish_time > job.latest_finish:
            violations.append(f"작업 {job.job_id}: 완료 시간 제약 위반")
            return False

        # 추가 제약 조건들 (constraint_manager 사용)
        constraint_result = self.constraint_manager.check_constraints(job, start_time)
        if not constraint_result.is_feasible:
            violations.extend(constraint_result.violated_constraints)
            return False

        return True

    def _find_next_feasible_time(self, job: Job, current_time: datetime) -> datetime:
        """다음 실행 가능한 시간 찾기"""
        # 간단한 구현: 1시간씩 이동하며 확인
        next_time = current_time
        max_attempts = 24 * 7  # 최대 1주일

        for _ in range(max_attempts):
            next_time += timedelta(hours=1)

            if self._check_constraints(job, next_time, []):
                return next_time

        # 제약을 만족하는 시간을 찾지 못한 경우
        self.logger.warning(f"작업 {job.job_id}에 대한 실행 가능한 시간을 찾지 못함")
        return current_time

    def _calculate_job_cost(self, job: Job) -> float:
        """작업 비용 계산"""
        if not job.actual_start:
            return 0.0

        duration_hours = job.processing_time / 60
        return self.tou_model.calculate_cost(
            job.actual_start, duration_hours, job.power_requirement
        )

    def _calculate_objective_value(self,
                                   objective: SchedulingObjective,
                                   cost: float,
                                   peak: float,
                                   makespan: float,
                                   delay: float) -> float:
        """목적 함수 값 계산"""
        if objective == SchedulingObjective.MINIMIZE_COST:
            return cost
        elif objective == SchedulingObjective.MINIMIZE_PEAK:
            return peak
        elif objective == SchedulingObjective.MINIMIZE_MAKESPAN:
            return makespan
        elif objective == SchedulingObjective.MULTI_OBJECTIVE:
            # 가중 합 (정규화 필요)
            normalized_cost = cost / 10000  # 임시 정규화
            normalized_peak = peak / 1000
            normalized_makespan = makespan / 100
            normalized_delay = delay / 10

            return (0.4 * normalized_cost +
                    0.3 * normalized_peak +
                    0.2 * normalized_makespan +
                    0.1 * normalized_delay)
        else:
            return cost + peak + makespan + delay

    def _update_stats(self, computation_time: float, is_successful: bool):
        """통계 업데이트"""
        self.stats['total_scheduling'] += 1

        if is_successful:
            self.stats['successful_scheduling'] += 1

        # 평균 계산 시간 업데이트
        total = self.stats['total_scheduling']
        self.stats['average_computation_time'] = (
                (self.stats['average_computation_time'] * (total - 1) + computation_time) / total
        )


class OptimalScheduler(BaseScheduler):
    """최적 스케줄러 (MIP 기반)"""

    def __init__(self):
        super().__init__("optimal_scheduler")
        self.time_limit = 300  # 5분

    @log_performance
    def schedule(self, jobs: List[Job], objective: SchedulingObjective) -> ScheduleResult:
        """최적 스케줄링 (MIP)"""
        start_time = datetime.now()

        # 입력 검증
        validation_issues = self._validate_jobs(jobs)
        if validation_issues:
            raise SchedulingError(f"작업 검증 실패: {validation_issues}")

        try:
            # OR-Tools 솔버 생성
            solver = pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                raise OptimizationException("MIP 솔버를 생성할 수 없음")

            # 변수 및 제약 조건 생성
            variables, constraints = self._create_mip_model(solver, jobs, objective)

            # 목적 함수 설정
            self._set_objective(solver, variables, jobs, objective)

            # 솔버 실행
            solver.SetTimeLimit(self.time_limit * 1000)  # milliseconds
            status = solver.Solve()

            # 결과 해석
            if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
                scheduled_jobs = self._extract_solution(solver, variables, jobs)
                is_feasible = True
                constraints_violated = []
            else:
                # 해를 찾지 못한 경우 그리디로 폴백
                self.logger.warning("최적해를 찾지 못함. 그리디 알고리즘으로 폴백")
                greedy_scheduler = GreedyScheduler()
                return greedy_scheduler.schedule(jobs, objective)

        except Exception as e:
            self.logger.error(f"MIP 스케줄링 오류: {e}")
            # 오류 발생시 그리디로 폴백
            greedy_scheduler = GreedyScheduler()
            return greedy_scheduler.schedule(jobs, objective)

        # 메트릭 계산
        total_cost, peak_power, total_delay, makespan = self._calculate_schedule_metrics(scheduled_jobs)

        # 가동률 계산
        machine_count = len(set(job.machine_id for job in jobs))
        total_processing_time = sum(job.processing_time for job in scheduled_jobs) / 60
        utilization_rate = total_processing_time / (makespan * machine_count) if makespan > 0 else 0

        # 목적 함수 값
        objective_value = solver.Objective().Value() if solver else 0

        computation_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(computation_time, is_feasible)

        return ScheduleResult(
            scheduled_jobs=scheduled_jobs,
            total_cost=total_cost,
            peak_power=peak_power,
            makespan=makespan,
            total_delay=total_delay,
            utilization_rate=utilization_rate,
            objective_value=objective_value,
            computation_time=computation_time,
            is_feasible=is_feasible,
            constraints_violated=constraints_violated,
            optimization_details={
                'algorithm': 'mip_optimal',
                'solver_status': status,
                'time_limit': self.time_limit,
                'gap': solver.Objective().BestBound() - solver.Objective().Value() if solver else 0
            }
        )

    def _create_mip_model(self, solver, jobs: List[Job], objective: SchedulingObjective):
        """MIP 모델 생성"""
        # 시간 이산화 (시간 단위)
        time_horizon = 24 * 7  # 1주일
        time_slots = list(range(time_horizon))

        variables = {}
        constraints = []

        # 결정 변수: x[j,t] = 작업 j가 시간 t에 시작하면 1
        for job in jobs:
            for t in time_slots:
                var_name = f"x_{job.job_id}_{t}"
                variables[var_name] = solver.IntVar(0, 1, var_name)

        # 제약 조건 1: 각 작업은 정확히 한 번 시작
        for job in jobs:
            constraint = solver.Constraint(1, 1)
            for t in time_slots:
                var_name = f"x_{job.job_id}_{t}"
                constraint.SetCoefficient(variables[var_name], 1)
            constraints.append(constraint)

        # 제약 조건 2: 기계 용량 제약
        machines = set(job.machine_id for job in jobs)
        for machine in machines:
            machine_jobs = [job for job in jobs if job.machine_id == machine]

            for t in time_slots:
                constraint = solver.Constraint(0, 1)
                for job in machine_jobs:
                    # 작업이 t 시간에 실행 중인지 확인
                    for start_t in range(max(0, t - job.processing_time // 60 + 1), t + 1):
                        if start_t in time_slots:
                            var_name = f"x_{job.job_id}_{start_t}"
                            constraint.SetCoefficient(variables[var_name], 1)
                constraints.append(constraint)

        return variables, constraints

    def _set_objective(self, solver, variables, jobs: List[Job], objective: SchedulingObjective):
        """목적 함수 설정"""
        objective_expr = solver.Objective()

        time_horizon = 24 * 7
        time_slots = list(range(time_horizon))

        if objective == SchedulingObjective.MINIMIZE_COST:
            # 비용 최소화
            for job in jobs:
                for t in time_slots:
                    var_name = f"x_{job.job_id}_{t}"
                    # 간단한 비용 모델 (실제로는 TOU 요금을 고려해야 함)
                    cost_coeff = job.power_requirement * (job.processing_time / 60) * (1.0 + 0.1 * (t % 24))
                    objective_expr.SetCoefficient(variables[var_name], cost_coeff)

        elif objective == SchedulingObjective.MINIMIZE_MAKESPAN:
            # 완료시간 최소화 (보조 변수 필요)
            makespan_var = solver.IntVar(0, time_horizon, "makespan")

            for job in jobs:
                for t in time_slots:
                    var_name = f"x_{job.job_id}_{t}"
                    completion_time = t + job.processing_time // 60

                    # makespan >= completion_time * x[j,t]
                    constraint = solver.Constraint(-solver.infinity(), 0)
                    constraint.SetCoefficient(makespan_var, 1)
                    constraint.SetCoefficient(variables[var_name], -completion_time)

            objective_expr.SetCoefficient(makespan_var, 1)

        objective_expr.SetMinimization()

    def _extract_solution(self, solver, variables, jobs: List[Job]) -> List[Job]:
        """해 추출"""
        scheduled_jobs = []
        time_horizon = 24 * 7

        for job in jobs:
            job_copy = Job(
                job_id=job.job_id,
                machine_id=job.machine_id,
                power_requirement=job.power_requirement,
                processing_time=job.processing_time,
                due_date=job.due_date,
                priority=job.priority,
                earliest_start=job.earliest_start,
                latest_finish=job.latest_finish,
                setup_time=job.setup_time,
                status=JobStatus.SCHEDULED
            )

            # 시작 시간 찾기
            for t in range(time_horizon):
                var_name = f"x_{job.job_id}_{t}"
                if variables[var_name].solution_value() > 0.5:
                    job_copy.actual_start = datetime.now() + timedelta(hours=t)
                    job_copy.actual_finish = job_copy.actual_start + timedelta(minutes=job.processing_time)
                    job_copy.cost = self._calculate_job_cost(job_copy)
                    break

            scheduled_jobs.append(job_copy)

        return scheduled_jobs


class AdaptiveScheduler(BaseScheduler):
    """적응형 스케줄러 (실시간 재스케줄링)"""

    def __init__(self):
        super().__init__("adaptive_scheduler")
        self.current_schedule: Optional[ScheduleResult] = None
        self.rescheduling_threshold = 0.2  # 20% 변화시 재스케줄링
        self.base_scheduler = GreedyScheduler()  # 기본 스케줄러

    def schedule(self, jobs: List[Job], objective: SchedulingObjective) -> ScheduleResult:
        """적응형 스케줄링"""
        # 초기 스케줄링 또는 전체 재스케줄링
        result = self.base_scheduler.schedule(jobs, objective)
        self.current_schedule = result
        return result

    def update_schedule(self,
                        new_jobs: List[Job] = None,
                        completed_jobs: List[str] = None,
                        delayed_jobs: List[str] = None) -> Optional[ScheduleResult]:
        """스케줄 업데이트"""
        if not self.current_schedule:
            return None

        # 변화량 계산
        change_ratio = self._calculate_change_ratio(new_jobs, completed_jobs, delayed_jobs)

        if change_ratio > self.rescheduling_threshold:
            self.logger.info(f"변화량 {change_ratio:.2%} > 임계값 {self.rescheduling_threshold:.2%}, 재스케줄링 수행")

            # 업데이트된 작업 리스트 생성
            updated_jobs = self._update_job_list(new_jobs, completed_jobs, delayed_jobs)

            # 재스케줄링
            new_schedule = self.base_scheduler.schedule(
                updated_jobs,
                SchedulingObjective.MINIMIZE_COST  # 기본 목적
            )

            self.current_schedule = new_schedule
            return new_schedule

        return None

    def _calculate_change_ratio(self,
                                new_jobs: List[Job],
                                completed_jobs: List[str],
                                delayed_jobs: List[str]) -> float:
        """변화율 계산"""
        if not self.current_schedule:
            return 1.0

        total_jobs = len(self.current_schedule.scheduled_jobs)

        changes = 0
        changes += len(new_jobs) if new_jobs else 0
        changes += len(completed_jobs) if completed_jobs else 0
        changes += len(delayed_jobs) if delayed_jobs else 0

        return changes / max(total_jobs, 1)

    def _update_job_list(self,
                         new_jobs: List[Job],
                         completed_jobs: List[str],
                         delayed_jobs: List[str]) -> List[Job]:
        """작업 리스트 업데이트"""
        current_jobs = self.current_schedule.scheduled_jobs.copy()

        # 완료된 작업 제거
        if completed_jobs:
            current_jobs = [job for job in current_jobs
                            if job.job_id not in completed_jobs]

        # 지연된 작업 상태 업데이트
        if delayed_jobs:
            for job in current_jobs:
                if job.job_id in delayed_jobs:
                    job.status = JobStatus.DELAYED
                    # 새로운 시작 시간 설정 (현재 시간 이후)
                    job.earliest_start = max(job.earliest_start, datetime.now())

        # 새 작업 추가
        if new_jobs:
            current_jobs.extend(new_jobs)

        return current_jobs


# 팩토리 함수들
def create_scheduler(scheduler_type: str = "greedy") -> BaseScheduler:
    """스케줄러 생성"""
    if scheduler_type == "greedy":
        return GreedyScheduler()
    elif scheduler_type == "optimal":
        return OptimalScheduler()
    elif scheduler_type == "adaptive":
        return AdaptiveScheduler()
    else:
        raise ValueError(f"지원하지 않는 스케줄러 타입: {scheduler_type}")


def schedule_jobs(jobs: List[Job],
                  objective: SchedulingObjective = SchedulingObjective.MINIMIZE_COST,
                  scheduler_type: str = "greedy") -> ScheduleResult:
    """편의 함수: 작업 스케줄링"""
    scheduler = create_scheduler(scheduler_type)
    return scheduler.schedule(jobs, objective)


# 사용 예시
if __name__ == "__main__":
    # 샘플 작업 생성
    sample_jobs = [
        Job(
            job_id="job_1",
            machine_id="machine_1",
            power_requirement=50.0,
            processing_time=120,  # 2시간
            due_date=datetime.now() + timedelta(hours=8),
            priority=1
        ),
        Job(
            job_id="job_2",
            machine_id="machine_1",
            power_requirement=75.0,
            processing_time=180,  # 3시간
            due_date=datetime.now() + timedelta(hours=12),
            priority=2
        ),
        Job(
            job_id="job_3",
            machine_id="machine_2",
            power_requirement=60.0,
            processing_time=90,  # 1.5시간
            due_date=datetime.now() + timedelta(hours=6),
            priority=1
        )
    ]

    # 스케줄링 수행
    result = schedule_jobs(
        sample_jobs,
        SchedulingObjective.MINIMIZE_COST,
        "greedy"
    )

    print("=== 스케줄링 결과 ===")
    print(f"실행 가능: {result.is_feasible}")
    print(f"총 비용: {result.total_cost:.2f}")
    print(f"피크 전력: {result.peak_power:.2f} kW")
    print(f"완료 시간: {result.makespan:.2f} 시간")
    print(f"지연 시간: {result.total_delay:.2f} 시간")
    print(f"계산 시간: {result.computation_time:.3f} 초")

    # 간트 차트 데이터
    gantt_data = result.to_gantt_data()
    print("\n=== 간트 차트 데이터 ===")
    for item in gantt_data:
        print(f"작업 {item['job_id']}: {item['start']} ~ {item['finish']} ({item['power']} kW)")
        """
스마트팩토리 에너지 관리 시스템 - 스케줄링 최적화 모듈

전력 기반 생산 스케줄링 최적화
- TOU 요금제 기반 비용 최소화
- 피크 전력 제약 조건 준수
- 다중 목적 최적화 (비용, 효율, 납기)
- 실시간 스케줄 조정 및 재최적화
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio

# 최적화 라이브러리
from scipy.optimize import minimize, linprog
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

# Core 모듈
from core.config import get_config
from core.logger import get_logger, log_performance
from core.exceptions import (
    SchedulingError, ConstraintViolationError,
    PeakPowerExceededError, OptimizationException,
    safe_execute
)

# Optimization 모듈
from .tou_pricing import TOUPricingModel
from .constraints import ConstraintManager, PowerConstraint, TimeConstraint


class SchedulingObjective(Enum):
    """스케줄링 목적"""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_PEAK = "minimize_peak"
    MINIMIZE_MAKESPAN = "minimize_makespan"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    MULTI_OBJECTIVE = "multi_objective"


class JobStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """생산 작업"""
    job_id: str
    machine_id: str
    power_requirement: float  # kW
    processing_time: int  # minutes
    due_date: datetime
    priority: int = 1  # 1=highest, 5=lowest
    earliest_start: Optional[datetime] = None
    latest_finish: Optional[datetime] = None
    setup_time: int = 0  # minutes
    status: JobStatus = JobStatus.PENDING
    actual_start: Optional[datetime] = None
    actual_finish: Optional[datetime] = None
    cost: float = 0.0

    def __post_init__(self):
        if self.earliest_start is None:
            self.earliest_start = datetime.now()
        if self.latest_finish is None:
            self.latest_finish = self.due_date

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            'job_id': self.job_id,
            'machine_id': self.machine_id,
            'power_requirement': self.power_requirement,
            'processing_time': self.processing_time,
            'due_date': self.due_date.isoformat(),
            'priority': self.priority,
            'earliest_start': self.earliest_start.isoformat() if self.earliest_start else None,
            'latest_finish': self.latest_finish.isoformat() if self.latest_finish else None,
            'setup_time': self.setup_time,
            'status': self.status.value,
            'actual_start': self.actual_start.isoformat() if self.actual_start else None,
            'actual_finish': self.actual_finish.isoformat() if self.actual_finish else None,
            'cost': self.cost
        }


@dataclass
class ScheduleResult:
    """스케줄링 결과"""
    scheduled_jobs: List[Job]
    total_cost: float
    peak_power: float
    makespan: float  # hours
    total_delay: float  # hours
    utilization_rate: float
    objective_value: float
    computation_time: float
    is_feasible: bool
    constraints_violated: List[str] = field(default_factory=list)
    optimization_details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> Dict:
        """결과 요약"""
        return {
            'total_jobs': len(self.scheduled_jobs),
            'feasible': self.is_feasible,
            'total_cost': round(self.total_cost)