"""
Constraint Manager for Power and Energy Management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ConstraintManager:
    """전력 및 에너지 제약 조건 관리"""

    def __init__(self, peak_power_limit: float = 1000, contract_power: float = 1200):
        self.peak_power_limit = peak_power_limit
        self.contract_power = contract_power
        self.violations = []

    def check_peak_power_constraint(self, schedule: pd.DataFrame) -> Dict:
        """피크 전력 제약 검사"""
        hourly_power = self._calculate_hourly_power(schedule)
        violations = []

        for hour, power in enumerate(hourly_power):
            if power > self.peak_power_limit:
                violations.append({
                    'hour': hour,
                    'power': power,
                    'excess': power - self.peak_power_limit,
                    'violation_ratio': power / self.peak_power_limit
                })

        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'max_power': max(hourly_power),
            'total_violations': len(violations)
        }

    def check_contract_power_constraint(self, schedule: pd.DataFrame) -> Dict:
        """계약전력 제약 검사"""
        hourly_power = self._calculate_hourly_power(schedule)
        max_power = max(hourly_power)

        return {
            'is_valid': max_power <= self.contract_power,
            'max_power': max_power,
            'contract_power': self.contract_power,
            'excess': max(0, max_power - self.contract_power)
        }

    def check_deadline_constraint(self, schedule: pd.DataFrame, deadline: int = 1440) -> Dict:
        """마감시간 제약 검사"""
        late_jobs = schedule[schedule['end_time'] > deadline]

        return {
            'is_valid': len(late_jobs) == 0,
            'late_jobs': len(late_jobs),
            'late_job_ids': late_jobs['job_id'].tolist() if len(late_jobs) > 0 else [],
            'max_completion_time': schedule['end_time'].max()
        }

    def _calculate_hourly_power(self, schedule: pd.DataFrame) -> List[float]:
        """시간대별 전력 사용량 계산"""
        hourly_power = [0] * 24

        for _, job in schedule.iterrows():
            start_hour = int(job['start_time'] // 60)
            end_hour = int(job['end_time'] // 60)

            for hour in range(start_hour, min(end_hour + 1, 24)):
                hourly_power[hour] += job['power_consumption']

        return hourly_power

    def validate_all_constraints(self, schedule: pd.DataFrame) -> Dict:
        """모든 제약 조건 검증"""
        peak_check = self.check_peak_power_constraint(schedule)
        contract_check = self.check_contract_power_constraint(schedule)
        deadline_check = self.check_deadline_constraint(schedule)

        all_valid = (peak_check['is_valid'] and
                    contract_check['is_valid'] and
                    deadline_check['is_valid'])

        return {
            'all_valid': all_valid,
            'peak_power': peak_check,
            'contract_power': contract_check,
            'deadline': deadline_check,
            'summary': {
                'total_violations': len(peak_check['violations']),
                'max_power': peak_check['max_power'],
                'late_jobs': deadline_check['late_jobs']
            }
        }