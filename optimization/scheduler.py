"""
Power Constraint Scheduler for Smart Factory Energy Management
Implements peak power constraint-based job scheduling with TOU pricing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import time
import logging
from scipy.optimize import minimize
from pulp import *

from .tou_pricing import TOUPricingModel

logger = logging.getLogger(__name__)


class PowerConstraintScheduler:
    """피크 전력 제약 기반 작업 스케줄링 - 실제 요금표 연동"""

    def __init__(self, peak_power_limit: float = 1000, contract_power: float = 1200):
        self.peak_power_limit = peak_power_limit  # kW
        self.contract_power = contract_power       # 계약전력 kW
        self.tou_model = TOUPricingModel()
        self.tou_model.setup_korean_industrial_rates()

    def create_job_data(self, n_jobs: int = 20) -> pd.DataFrame:
        """작업 데이터 생성 (실제 제조업 기반)"""
        np.random.seed(42)

        # 실제 제조업 설비별 전력 소비 패턴
        equipment_profiles = {
            'CNC': {'power_range': (80, 150), 'time_range': (30, 120)},
            'Press': {'power_range': (200, 400), 'time_range': (5, 30)},
            'Furnace': {'power_range': (300, 600), 'time_range': (60, 240)},
            'Conveyor': {'power_range': (20, 50), 'time_range': (15, 60)},
            'Welder': {'power_range': (100, 250), 'time_range': (10, 45)},
            'Compressor': {'power_range': (150, 300), 'time_range': (30, 90)}
        }

        jobs = []
        for i in range(n_jobs):
            equipment_type = np.random.choice(list(equipment_profiles.keys()))
            profile = equipment_profiles[equipment_type]

            job = {
                'job_id': f'J_{i+1:02d}',
                'equipment_type': equipment_type,
                'processing_time': np.random.randint(*profile['time_range']),  # 분
                'power_consumption': np.random.uniform(*profile['power_range']),  # kW
                'arrival_time': np.random.randint(0, 1200),  # 0-1200분 (20시간)
                'deadline': 1440,  # 24시간 (1440분)
                'priority': np.random.randint(1, 4),  # 1(높음) - 3(낮음)
                'setup_time': np.random.randint(5, 20),  # 준비시간
                'energy_efficiency': np.random.uniform(0.7, 0.95)  # 에너지 효율
            }
            jobs.append(job)

        df = pd.DataFrame(jobs)
        logger.info(f"✅ 실제 제조업 기반 작업 데이터 생성: {len(jobs)}개")
        logger.info(f"   📊 설비별 분포: {df['equipment_type'].value_counts().to_dict()}")

        return df

    def erd_scheduling(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """ERD (Earliest Release Date) 스케줄링"""
        # 도착시간 순으로 정렬
        sorted_jobs = jobs_df.sort_values('arrival_time').copy()

        schedule = []
        current_time = 0

        for _, job in sorted_jobs.iterrows():
            start_time = max(current_time, job['arrival_time'])
            end_time = start_time + job['processing_time']

            schedule.append({
                'job_id': job['job_id'],
                'start_time': start_time,
                'end_time': end_time,
                'power_consumption': job['power_consumption'],
                'processing_time': job['processing_time']
            })

            current_time = end_time

        return pd.DataFrame(schedule)

    def optimize_with_peak_constraint(self, jobs_df: pd.DataFrame, method: str = "lagrange") -> pd.DataFrame:
        """피크 전력 제약 기반 최적화"""

        if method == "lagrange":
            return self._lagrange_optimization(jobs_df)
        elif method == "milp":
            return self._milp_optimization(jobs_df)
        else:
            return self._heuristic_optimization(jobs_df)

    def _lagrange_optimization(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """라그랑주 완화 기법 (논문 알고리즘 단순화)"""
        logger.info("🔧 라그랑주 완화 기법 적용 중...")

        # ERD 기본 스케줄 생성
        base_schedule = self.erd_scheduling(jobs_df)

        # 시간대별 전력 사용량 계산
        hourly_power = self._calculate_hourly_power(base_schedule)

        # 피크 제약 위반 시간대 식별
        violations = []
        for hour, power in enumerate(hourly_power):
            if power > self.peak_power_limit:
                violations.append({
                    'hour': hour,
                    'excess_power': power - self.peak_power_limit,
                    'price': self.tou_model.get_hourly_price(hour % 24)
                })

        if not violations:
            logger.info("✅ 피크 전력 제약 준수")
            return base_schedule

        # 위반 시간대의 작업을 저렴한 시간대로 이동
        optimized_schedule = self._redistribute_jobs(base_schedule, violations)

        return optimized_schedule

    def _milp_optimization(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """혼합 정수 선형 계획법"""
        logger.info("🔧 MILP 최적화 적용 중...")

        try:
            # PuLP 문제 정의
            prob = LpProblem("Energy_Cost_Minimization", LpMinimize)

            n_jobs = len(jobs_df)
            time_slots = list(range(0, 1440, 15))  # 15분 간격

            # 결정변수: x[i][t] = 작업 i가 시간 t에 시작하는지 여부
            x = {}
            for i in range(n_jobs):
                for t in time_slots:
                    x[i, t] = LpVariable(f"x_{i}_{t}", cat='Binary')

            # 목적함수: 총 전력비용 최소화
            total_cost = 0
            for i in range(n_jobs):
                job = jobs_df.iloc[i]
                for t in time_slots:
                    hour = (t // 60) % 24
                    price_info = self.tou_model.get_hourly_price(hour)
                    power = job['power_consumption']
                    duration = job['processing_time']
                    total_cost += x[i, t] * power * (duration/60) * price_info['total_price']

            prob += total_cost

            # 제약조건 1: 각 작업은 정확히 한 번 스케줄링
            for i in range(n_jobs):
                prob += lpSum([x[i, t] for t in time_slots]) == 1

            # 제약조건 2: 피크 전력 제약
            for t in time_slots:
                power_at_t = 0
                for i in range(n_jobs):
                    job = jobs_df.iloc[i]
                    # 시간 t에서 작업 i가 실행 중인지 확인
                    for start_t in time_slots:
                        if start_t <= t < start_t + job['processing_time']:
                            power_at_t += x[i, start_t] * job['power_consumption']

                prob += power_at_t <= self.peak_power_limit

            # 제약조건 3: 도착시간 이후 시작
            for i in range(n_jobs):
                job = jobs_df.iloc[i]
                arrival = job['arrival_time']
                for t in time_slots:
                    if t < arrival:
                        prob += x[i, t] == 0

            # 문제 해결
            prob.solve(PULP_CBC_CMD(msg=0))

            # 결과 추출
            if prob.status == 1:  # Optimal
                schedule = []
                for i in range(n_jobs):
                    job = jobs_df.iloc[i]
                    for t in time_slots:
                        if x[i, t].varValue == 1:
                            schedule.append({
                                'job_id': job['job_id'],
                                'start_time': t,
                                'end_time': t + job['processing_time'],
                                'power_consumption': job['power_consumption'],
                                'processing_time': job['processing_time']
                            })
                            break

                return pd.DataFrame(schedule)
            else:
                logger.warning("⚠️ MILP 최적화 실패, ERD 스케줄 반환")
                return self.erd_scheduling(jobs_df)

        except Exception as e:
            logger.error(f"❌ MILP 최적화 오류: {e}")
            return self.erd_scheduling(jobs_df)

    def _heuristic_optimization(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """휴리스틱 최적화 (그리디 + 로컬 서치)"""
        logger.info("🔧 휴리스틱 최적화 적용 중...")

        # 우선순위 기반 정렬 (전력효율 고려)
        jobs_df = jobs_df.copy()
        jobs_df['power_efficiency'] = jobs_df['processing_time'] / jobs_df['power_consumption']
        sorted_jobs = jobs_df.sort_values(['arrival_time', 'power_efficiency'], ascending=[True, False])

        schedule = []
        current_time = 0

        for _, job in sorted_jobs.iterrows():
            # 최적 시작 시간 탐색
            best_start_time = self._find_optimal_start_time(
                job, current_time, schedule
            )

            end_time = best_start_time + job['processing_time']

            schedule.append({
                'job_id': job['job_id'],
                'start_time': best_start_time,
                'end_time': end_time,
                'power_consumption': job['power_consumption'],
                'processing_time': job['processing_time']
            })

            current_time = max(current_time, end_time)

        return pd.DataFrame(schedule)

    def _find_optimal_start_time(self, job: pd.Series, earliest_time: int, existing_schedule: List[Dict]) -> int:
        """작업의 최적 시작 시간 탐색"""
        earliest_start = max(earliest_time, job['arrival_time'])
        best_start = earliest_start
        best_cost = float('inf')

        # 가능한 시작 시간대 탐색 (1시간 간격)
        for start_candidate in range(earliest_start, job['deadline'] - job['processing_time'], 60):

            # 다른 작업과 충돌 검사
            if self._check_time_conflict(start_candidate, job['processing_time'], existing_schedule):
                continue

            # 전력 제약 검사
            if self._check_power_constraint(start_candidate, job, existing_schedule):
                continue

            # 비용 계산
            cost = self._calculate_job_cost(start_candidate, job)

            if cost < best_cost:
                best_cost = cost
                best_start = start_candidate

        return best_start

    def _check_time_conflict(self, start_time: int, duration: int, schedule: List[Dict]) -> bool:
        """시간 충돌 검사"""
        end_time = start_time + duration

        for existing in schedule:
            if not (end_time <= existing['start_time'] or
                   start_time >= existing['end_time']):
                return True
        return False

    def _check_power_constraint(self, start_time: int, job: pd.Series, schedule: List[Dict]) -> bool:
        """피크 전력 제약 검사"""
        end_time = start_time + job['processing_time']

        # 시간대별 전력 사용량 계산
        for t in range(start_time, end_time):
            total_power = job['power_consumption']

            for existing in schedule:
                if existing['start_time'] <= t < existing['end_time']:
                    total_power += existing['power_consumption']

            if total_power > self.peak_power_limit:
                return True

        return False

    def _calculate_job_cost(self, start_time: int, job: pd.Series, month: int = 7) -> float:
        """작업의 전력 비용 계산 (실제 요금표 기반)"""
        total_cost = 0
        duration = job['processing_time']
        power = job['power_consumption']

        for minute in range(duration):
            time_point = start_time + minute
            hour = (time_point // 60) % 24
            price_info = self.tou_model.get_hourly_price(hour, month)
            # 분당 비용 = kW × 원/kWh × (1시간/60분)
            minute_cost = power * price_info['total_price'] / 60
            total_cost += minute_cost

        return total_cost

    def _calculate_hourly_power(self, schedule: pd.DataFrame) -> List[float]:
        """시간대별 전력 사용량 계산"""
        hourly_power = [0] * 24

        for _, job in schedule.iterrows():
            start_hour = int(job['start_time'] // 60)
            end_hour = int(job['end_time'] // 60)

            for hour in range(start_hour, min(end_hour + 1, 24)):
                hourly_power[hour] += job['power_consumption']

        return hourly_power

    def _redistribute_jobs(self, schedule: pd.DataFrame, violations: List[Dict]) -> pd.DataFrame:
        """작업 재배치 (피크 제약 위반 해결)"""
        optimized = schedule.copy()

        # 위반 시간대의 작업들을 식별하고 재배치
        for violation in violations:
            hour = violation['hour']
            # 해당 시간대 작업들 중 우선순위가 낮은 것 이동
            # (실제 구현에서는 더 정교한 로직 필요)
            pass

        return optimized

    def evaluate_schedule(self, schedule: pd.DataFrame, jobs_df: pd.DataFrame, month: int = 7) -> Dict:
        """스케줄 평가 (실제 요금표 기반)"""
        results = {}

        # 1. 총 비용 계산 (실제 요금표 기반)
        total_energy_cost = 0