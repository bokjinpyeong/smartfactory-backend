"""
Integrated Energy Management System
통합 에너지 관리 시스템 - TOU 요금제 + 피크 전력 제약 + 이상탐지
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import logging

from .tou_pricing import TOUPricingModel
from .scheduler import PowerConstraintScheduler
from .constraints import ConstraintManager

logger = logging.getLogger(__name__)


class IntegratedEnergyManagementSystem:
    """통합 에너지 관리 시스템"""

    def __init__(self, peak_power_limit: float = 1000, contract_power: float = 1200):
        self.scheduler = PowerConstraintScheduler(peak_power_limit, contract_power)
        self.constraint_manager = ConstraintManager(peak_power_limit, contract_power)
        self.tou_model = TOUPricingModel()
        self.tou_model.setup_korean_industrial_rates()

        # 성능 지표 저장
        self.performance_metrics = {}
        self.optimization_results = {}

    def create_enhanced_industrial_data(self, n_samples: int = 10000,
                                        anomaly_rate: float = 0.05) -> pd.DataFrame:
        """향상된 산업용 데이터 생성 (실제 요금표 기반)"""
        np.random.seed(42)
        logger.info(f"🏭 실제 요금표 기반 산업용 데이터 생성 중... ({n_samples:,}개 샘플)")

        # 시간 정보 생성 (1년간 시간별 데이터)
        start_date = pd.Timestamp('2024-01-01')
        timestamps = pd.date_range(start_date, periods=n_samples, freq='H')

        data = pd.DataFrame({
            'timestamp': timestamps,
            'hour': timestamps.hour,
            'day_of_week': timestamps.dayofweek,
            'month': timestamps.month,
            'day_of_year': timestamps.dayofyear,
        })

        # 기본 센서 데이터 생성
        data = self._generate_sensor_data(data, n_samples)

        # TOU 요금제 적용
        data = self._apply_tou_pricing(data)

        # 이상치 생성
        data = self._generate_anomalies(data, anomaly_rate)

        # 통계 정보 출력
        self._print_data_statistics(data)

        return data

    def _generate_sensor_data(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """센서 데이터 생성"""
        # 기존 센서 데이터
        data['Motor_Temperature'] = np.random.normal(85, 12, n_samples)
        data['Bearing_Temperature'] = np.random.normal(45, 8, n_samples)
        data['Oil_Temperature'] = np.random.normal(60, 10, n_samples)
        data['Ambient_Temperature'] = np.random.normal(25, 5, n_samples)
        data['RMS_Vibration'] = np.random.exponential(2.5, n_samples)
        data['Peak_Vibration'] = np.random.exponential(4, n_samples)
        data['Load_Torque'] = np.random.normal(120, 25, n_samples)
        data['Shaft_Speed_RPM'] = np.random.normal(1800, 180, n_samples)
        data['Total_Current'] = np.random.normal(55, 12, n_samples)
        data['Avg_Voltage'] = np.random.normal(220, 8, n_samples)
        data['Power_Factor'] = np.random.uniform(0.75, 0.95, n_samples)
        data['Workload_Percentage'] = np.random.uniform(15, 95, n_samples)
        data['Production_Rate'] = np.random.normal(100, 15, n_samples)
        data['Efficiency_Index'] = np.random.normal(0.85, 0.1, n_samples)
        data['Humidity'] = np.random.uniform(40, 80, n_samples)
        data['Pressure'] = np.random.normal(1013, 15, n_samples)

        # 범주형 데이터
        data['Machine_Type'] = np.random.choice(
            ['CNC', 'Press', 'Furnace', 'Conveyor', 'Welder'], n_samples
        )
        data['Operational_Mode'] = np.random.choice(
            ['Auto', 'Manual', 'Maintenance'], n_samples
        )
        data['Shift'] = np.random.choice(
            ['Morning', 'Afternoon', 'Night'], n_samples
        )

        # 계절별 온도 보정
        data['Seasonal_Temp_Adj'] = 10 * np.sin(
            2 * np.pi * data['day_of_year'] / 365.25 - np.pi / 2
        )
        data['Ambient_Temperature'] += data['Seasonal_Temp_Adj']

        return data

    def _apply_tou_pricing(self, data: pd.DataFrame) -> pd.DataFrame:
        """TOU 요금제 적용"""
        base_power = 400  # 기본 전력 400kW

        power_consumption = []
        electricity_prices = []
        tou_periods = []
        seasons = []

        for _, row in data.iterrows():
            hour = row['hour']
            month = row['month']

            # 실제 TOU 요금 정보 획득
            price_info = self.tou_model.get_hourly_price(hour, month)
            electricity_prices.append(price_info['total_price'])
            tou_periods.append(price_info['period'])
            seasons.append(price_info['season'])

            # 시간대별 전력 소비 패턴
            hour_factor = self._get_hour_factor(price_info['period'], row['day_of_week'])
            season_factor = self._get_season_factor(month, hour)

            # 물리적 요인 기반 전력 계산
            physical_power = self._calculate_physical_power(row, base_power, hour_factor, season_factor)
            power_consumption.append(np.clip(physical_power, 50, 1500))

        data['Power_Consumption_Real'] = power_consumption
        data['Electricity_Price_Real'] = electricity_prices
        data['TOU_Period_Real'] = tou_periods
        data['Season'] = seasons

        # 실제 전력비용 계산
        data['Energy_Cost_Real'] = data['Power_Consumption_Real'] * data['Electricity_Price_Real']

        return data

    def _get_hour_factor(self, period: str, day_of_week: int) -> float:
        """시간대별 전력 소비 팩터"""
        if period == 'peak':
            factor = np.random.uniform(1.3, 1.8)
        elif period == 'light_load':
            factor = np.random.uniform(0.3, 0.7)
        else:  # mid_load
            factor = np.random.uniform(0.8, 1.2)

        # 주말 보정
        if day_of_week in [5, 6]:
            factor *= 0.6

        return factor

    def _get_season_factor(self, month: int, hour: int) -> float:
        """계절별 전력 소비 팩터"""
        if month in [6, 7, 8]:  # 하절기
            if 10 <= hour <= 16:  # 냉방 부하
                return 1.2
        elif month in [12, 1, 2]:  # 동절기
            if 7 <= hour <= 9 or 17 <= hour <= 20:  # 난방 부하
                return 1.1
        return 1.0

    def _calculate_physical_power(self, row: pd.Series, base_power: float,
                                  hour_factor: float, season_factor: float) -> float:
        """물리적 요인 기반 전력 계산"""
        return (
                base_power * hour_factor * season_factor +
                row['Total_Current'] * row['Avg_Voltage'] * row['Power_Factor'] * 0.001 +
                row['Load_Torque'] * row['Shaft_Speed_RPM'] / 9549 * 0.8 +
                row['Workload_Percentage'] * 2.5 +
                np.maximum(0, row['Motor_Temperature'] - 80) * 1.5 +
                row['RMS_Vibration'] * 5 +
                (1 - row['Efficiency_Index']) * 100 +
                np.random.normal(0, 15)
        )

    def _generate_anomalies(self, data: pd.DataFrame, anomaly_rate: float) -> pd.DataFrame:
        """이상치 생성"""
        n_samples = len(data)
        n_anomalies = int(n_samples * anomaly_rate)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

        for idx in anomaly_indices:
            anomaly_type = np.random.choice([
                'peak_overload', 'equipment_failure',
                'inefficient_operation', 'seasonal_spike'
            ])

            if anomaly_type == 'peak_overload':
                self._create_peak_overload_anomaly(data, idx)
            elif anomaly_type == 'equipment_failure':
                self._create_equipment_failure_anomaly(data, idx)
            elif anomaly_type == 'inefficient_operation':
                self._create_inefficient_operation_anomaly(data, idx)
            elif anomaly_type == 'seasonal_spike':
                self._create_seasonal_spike_anomaly(data, idx)

        # 최종 전력비용 재계산
        data['Energy_Cost_Real'] = data['Power_Consumption_Real'] * data['Electricity_Price_Real']

        return data

    def _create_peak_overload_anomaly(self, data: pd.DataFrame, idx: int):
        """피크 시간대 과부하 이상치"""
        if data.loc[idx, 'TOU_Period_Real'] == 'peak':
            data.loc[idx, 'Power_Consumption_Real'] *= np.random.uniform(1.8, 2.5)
            data.loc[idx, 'Motor_Temperature'] += np.random.uniform(20, 40)
            data.loc[idx, 'Total_Current'] *= np.random.uniform(1.3, 1.7)

    def _create_equipment_failure_anomaly(self, data: pd.DataFrame, idx: int):
        """설비 고장 이상치"""
        data.loc[idx, 'Power_Consumption_Real'] *= np.random.uniform(1.5, 2.2)
        data.loc[idx, 'RMS_Vibration'] *= np.random.uniform(3, 6)
        data.loc[idx, 'Efficiency_Index'] *= np.random.uniform(0.4, 0.7)
        data.loc[idx, 'Motor_Temperature'] += np.random.uniform(25, 50)

    def _create_inefficient_operation_anomaly(self, data: pd.DataFrame, idx: int):
        """비효율적 운영 이상치"""
        data.loc[idx, 'Efficiency_Index'] *= np.random.uniform(0.5, 0.8)
        data.loc[idx, 'Power_Consumption_Real'] *= np.random.uniform(1.2, 1.6)

    def _create_seasonal_spike_anomaly(self, data: pd.DataFrame, idx: int):
        """계절적 급등 이상치"""
        month = data.loc[idx, 'month']
        if month in [7, 8] or month in [1, 12]:
            data.loc[idx, 'Power_Consumption_Real'] *= np.random.uniform(1.4, 1.9)

    def _print_data_statistics(self, data: pd.DataFrame):
        """데이터 통계 정보 출력"""
        logger.info(f"   ✅ 실제 요금표 기반 데이터 생성 완료: {data.shape}")
        logger.info(f"   📊 TOU 기간별 분포:")

        period_dist = data['TOU_Period_Real'].value_counts()
        for period, count in period_dist.items():
            avg_price = data[data['TOU_Period_Real'] == period]['Electricity_Price_Real'].mean()
            avg_power = data[data['TOU_Period_Real'] == period]['Power_Consumption_Real'].mean()
            logger.info(f"      {period}: {count:,}개 (평균 {avg_price:.1f}원/kWh, {avg_power:.1f}kW)")

        logger.info(f"   📊 계절별 분포:")
        season_dist = data['Season'].value_counts()
        for season, count in season_dist.items():
            avg_cost = data[data['Season'] == season]['Energy_Cost_Real'].mean()
            logger.info(f"      {season}: {count:,}개 (평균 {avg_cost:.1f}원/h)")

        logger.info(f"   💰 연간 전력비용 예상: {data['Energy_Cost_Real'].sum():,.0f} 원")
        logger.info(f"   ⚡ 평균 전력소비: {data['Power_Consumption_Real'].mean():.1f} kW")
        logger.info(f"   📈 피크 전력: {data['Power_Consumption_Real'].max():.1f} kW")

    def run_comprehensive_analysis(self, n_jobs: int = 20, n_samples: int = 50000) -> Dict:
        """종합 분석 실행"""
        logger.info("🚀 TOU 기반 통합 에너지 관리 시스템 분석")
        logger.info("=" * 60)

        results = {
            'schedules': {},
            'enhanced_data': None,
            'performance': {},
            'recommendations': []
        }

        try:
            # 1. 스케줄링 최적화
            logger.info("\n📋 1단계: 전력 기반 작업 스케줄링")
            jobs_df = self.scheduler.create_job_data(n_jobs)

            # 여러 방법 비교
            methods = ["ERD", "lagrange", "milp", "heuristic"]

            for method in methods:
                logger.info(f"   🔧 {method.upper()} 방법 실행 중...")
                start_time = time.time()

                if method == "ERD":
                    schedule = self.scheduler.erd_scheduling(jobs_df)
                else:
                    schedule = self.scheduler.optimize_with_peak_constraint(jobs_df, method)

                results['schedules'][method] = schedule

                # 평가
                evaluation = self.scheduler.evaluate_schedule(schedule, jobs_df)
                constraint_check = self.constraint_manager.validate_all_constraints(schedule)
                execution_time = time.time() - start_time

                logger.info(f"      ✅ 완료 ({execution_time:.2f}초)")
                logger.info(f"      💰 총 전력비용: {evaluation['total_energy_cost']:.2f} 원")
                logger.info(f"      ⚡ 피크 전력: {evaluation['peak_power']:.2f} kW")
                logger.info(f"      🚨 제약 위반: {not constraint_check['all_valid']}")

            # 2. 데이터 생성 및 분석
            logger.info(f"\n🧠 2단계: TOU 기반 데이터 분석")
            enhanced_data = self.create_enhanced_industrial_data(n_samples)
            results['enhanced_data'] = enhanced_data

            # 3. 권장사항 생성
            logger.info(f"\n📊 3단계: 권장사항 생성")
            recommendations = self._generate_recommendations(results['schedules'], enhanced_data)
            results['recommendations'] = recommendations

            # 4. 성능 지표 계산
            results['performance'] = self._calculate_performance_metrics(results)

            return results

        except Exception as e:
            logger.error(f"❌ 시스템 실행 오류: {str(e)}")
            return results

    def _generate_recommendations(self, schedules: Dict, data: pd.DataFrame) -> List[Dict]:
        """권장사항 생성"""
        recommendations = []

        # 1. 스케줄링 최적화 권장사항
        if schedules:
            costs = {}
            for method, schedule in schedules.items():
                evaluation = self.scheduler.evaluate_schedule(schedule, pd.DataFrame())
                costs[method] = evaluation['total_energy_cost']

            best_method = min(costs.keys(), key=lambda k: costs[k])
            worst_method = max(costs.keys(), key=lambda k: costs[k])
            savings = (costs[worst_method] - costs[best_method]) / costs[worst_method] * 100

            recommendations.append({
                'category': '스케줄링 최적화',
                'recommendation': f'{best_method.upper()} 방법 사용 권장',
                'benefit': f'최대 {savings:.1f}% 비용 절감 가능',
                'priority': 'High'
            })

        # 2. TOU 요금제 활용 권장사항
        peak_hours_consumption = data[data['TOU_Period_Real'] == 'peak']['Power_Consumption_Real'].mean()
        off_peak_consumption = data[data['TOU_Period_Real'] == 'light_load']['Power_Consumption_Real'].mean()

        if peak_hours_consumption > off_peak_consumption * 1.2:
            recommendations.append({
                'category': 'TOU 요금제 최적화',
                'recommendation': '피크 시간대 전력 소비 20% 감축',
                'benefit': '경부하 시간대로 작업 이동을 통한 전력비 절감',
                'priority': 'High'
            })

        # 3. 피크 전력 관리
        max_power = data['Power_Consumption_Real'].max()
        if max_power > self.scheduler.peak_power_limit:
            recommendations.append({
                'category': '피크 전력 관리',
                'recommendation': f'피크 전력을 {self.scheduler.peak_power_limit}kW 이하로 제한',
                'benefit': '피크 수요 요금 절감',
                'priority': 'Critical'
            })

        return recommendations

    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """성능 지표 계산"""
        metrics = {}

        if results['schedules']:
            # 스케줄링 성능
            erd_cost = self.scheduler.evaluate_schedule(
                results['schedules']['ERD'], pd.DataFrame()
            )['total_energy_cost']

            best_method = min(results['schedules'].keys(),
                              key=lambda m: self.scheduler.evaluate_schedule(
                                  results['schedules'][m], pd.DataFrame()
                              )['total_energy_cost'])

            best_cost = self.scheduler.evaluate_schedule(
                results['schedules'][best_method], pd.DataFrame()
            )['total_energy_cost']

            metrics['scheduling_improvement'] = (erd_cost - best_cost) / erd_cost * 100
            metrics['best_method'] = best_method

        return metrics

    def print_recommendations(self, recommendations: List[Dict]):
        """권장사항 출력"""
        logger.info("\n🎯 최적화 권장사항")
        logger.info("=" * 50)

        priority_order = {'Critical': 1, 'High': 2, 'Medium': 3, 'Low': 4}
        sorted_recommendations = sorted(recommendations,
                                        key=lambda x: priority_order.get(x['priority'], 5))

        for i, rec in enumerate(sorted_recommendations, 1):
            priority_icon = {'Critical': '🚨', 'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
            icon = priority_icon.get(rec['priority'], '⚪')

            logger.info(f"\n{icon} {i}. {rec['category']} [{rec['priority']}]")
            logger.info(f"   📋 권장사항: {rec['recommendation']}")
            logger.info(f"   💰 기대효과: {rec['benefit']}")