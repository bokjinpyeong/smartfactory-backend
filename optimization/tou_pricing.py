"""
스마트팩토리 에너지 관리 시스템 - TOU 요금제 모델

시간대별 차등 요금제(Time-of-Use) 관리
- 다양한 요금제 지원 (피크/오프피크/일반)
- 계절별, 요일별 요금 차등화
- 실시간 요금 계산 및 예측
- 비용 최적화를 위한 시간대 분석
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import calendar

# Core 모듈
from core.config import get_config
from core.logger import get_logger
from core.exceptions import TOUPricingError, safe_execute


class PricePeriod(Enum):
    """요금 시간대"""
    PEAK = "peak"           # 피크 시간대
    OFF_PEAK = "off_peak"   # 오프피크 시간대
    NORMAL = "normal"       # 일반 시간대
    SUPER_OFF_PEAK = "super_off_peak"  # 심야 시간대


class Season(Enum):
    """계절"""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


@dataclass
class PriceRate:
    """요금율 정보"""
    period: PricePeriod
    rate: float  # 원/kWh
    season: Optional[Season] = None
    day_type: Optional[str] = None  # weekday, weekend, holiday

    def __str__(self):
        return f"{self.period.value}: {self.rate}원/kWh"


@dataclass
class TOUSchedule:
    """TOU 시간대 스케줄"""
    start_time: time
    end_time: time
    period: PricePeriod
    weekdays: List[int] = field(default_factory=lambda: list(range(7)))  # 0=월요일
    seasons: List[Season] = field(default_factory=lambda: list(Season))

    def is_applicable(self, dt: datetime) -> bool:
        """해당 시간에 적용되는지 확인"""
        # 요일 확인
        if dt.weekday() not in self.weekdays:
            return False

        # 계절 확인
        current_season = self._get_season(dt)
        if current_season not in self.seasons:
            return False

        # 시간 확인
        current_time = dt.time()
        if self.start_time <= self.end_time:
            # 같은 날 내
            return self.start_time <= current_time <= self.end_time
        else:
            # 다음 날로 넘어가는 경우 (예: 23:00 ~ 06:00)
            return current_time >= self.start_time or current_time <= self.end_time

    def _get_season(self, dt: datetime) -> Season:
        """계절 판별"""
        month = dt.month

        if month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        elif month in [9, 10, 11]:
            return Season.AUTUMN
        else:
            return Season.WINTER


class BaseTOUModel(ABC):
    """기본 TOU 모델 클래스"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = get_logger(f"tou.{model_name}")
        self.config = get_config()

        self.schedules: List[TOUSchedule] = []
        self.rates: Dict[PricePeriod, float] = {}
        self.base_rate = 100.0  # 기본 요금 (원/kWh)

        # 통계
        self.stats = {
            'total_calculations': 0,
            'peak_usage_hours': 0,
            'off_peak_usage_hours': 0,
            'total_cost_calculated': 0.0
        }

    @abstractmethod
    def get_rate(self, dt: datetime) -> float:
        """특정 시간의 요금율 반환"""
        pass

    @abstractmethod
    def get_period(self, dt: datetime) -> PricePeriod:
        """특정 시간의 요금 시간대 반환"""
        pass

    def calculate_cost(self,
                      start_time: datetime,
                      duration_hours: float,
                      power_kw: float) -> float:
        """전력 사용 비용 계산"""
        total_cost = 0.0
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)

        # 시간별로 요금 계산
        while current_time < end_time:
            next_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            interval_end = min(next_hour, end_time)

            # 해당 시간 구간의 사용 시간 계산
            interval_hours = (interval_end - current_time).total_seconds() / 3600

            # 요금율 적용
            rate = self.get_rate(current_time)
            interval_cost = power_kw * interval_hours * rate
            total_cost += interval_cost

            current_time = interval_end

        # 통계 업데이트
        self.stats['total_calculations'] += 1
        self.stats['total_cost_calculated'] += total_cost

        return total_cost

    def get_average_rate(self, start_time: datetime, duration_minutes: int) -> float:
        """평균 요금율 계산"""
        total_rate = 0.0
        samples = max(1, duration_minutes // 15)  # 15분 간격으로 샘플링

        for i in range(samples):
            sample_time = start_time + timedelta(minutes=i * 15)
            total_rate += self.get_rate(sample_time)

        return total_rate / samples

    def find_cheapest_period(self,
                           duration_hours: float,
                           start_window: datetime,
                           end_window: datetime,
                           power_kw: float = 1.0) -> Tuple[datetime, float]:
        """가장 저렴한 시간대 찾기"""
        best_start = start_window
        best_cost = float('inf')

        # 1시간 간격으로 검색
        current_start = start_window
        while current_start + timedelta(hours=duration_hours) <= end_window:
            cost = self.calculate_cost(current_start, duration_hours, power_kw)

            if cost < best_cost:
                best_cost = cost
                best_start = current_start

            current_start += timedelta(hours=1)

        return best_start, best_cost

    def get_daily_profile(self, date: datetime) -> pd.DataFrame:
        """일일 요금 프로파일 생성"""
        hours = []
        rates = []
        periods = []

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)

        for hour in range(24):
            hour_time = start_of_day + timedelta(hours=hour)
            rate = self.get_rate(hour_time)
            period = self.get_period(hour_time)

            hours.append(hour)
            rates.append(rate)
            periods.append(period.value)

        return pd.DataFrame({
            'hour': hours,
            'rate': rates,
            'period': periods
        })

    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        return self.stats.copy()


class StandardTOUModel(BaseTOUModel):
    """표준 TOU 모델"""

    def __init__(self):
        super().__init__("standard_tou")
        self._setup_standard_schedule()

    def _setup_standard_schedule(self):
        """표준 TOU 스케줄 설정"""
        # 기본 요금율 설정
        self.rates = {
            PricePeriod.PEAK: self.config.power.tou_peak_rate * self.base_rate,
            PricePeriod.OFF_PEAK: self.config.power.tou_off_peak_rate * self.base_rate,
            PricePeriod.NORMAL: self.config.power.tou_normal_rate * self.base_rate,
            PricePeriod.SUPER_OFF_PEAK: 0.6 * self.base_rate
        }

        # 피크 시간대 (평일 오전 9시 ~ 오후 6시)
        self.schedules.append(TOUSchedule(
            start_time=time(9, 0),
            end_time=time(18, 0),
            period=PricePeriod.PEAK,
            weekdays=[0, 1, 2, 3, 4],  # 평일
            seasons=list(Season)
        ))

        # 오프피크 시간대 (평일 저녁 6시 ~ 밤 11시, 주말 낮)
        self.schedules.append(TOUSchedule(
            start_time=time(18, 0),
            end_time=time(23, 0),
            period=PricePeriod.OFF_PEAK,
            weekdays=[0, 1, 2, 3, 4],  # 평일
            seasons=list(Season)
        ))

        # 주말 낮 시간
        self.schedules.append(TOUSchedule(
            start_time=time(9, 0),
            end_time=time(18, 0),
            period=PricePeriod.OFF_PEAK,
            weekdays=[5, 6],  # 주말
            seasons=list(Season)
        ))

        # 심야 시간대 (밤 11시 ~ 오전 6시)
        self.schedules.append(TOUSchedule(
            start_time=time(23, 0),
            end_time=time(6, 0),
            period=PricePeriod.SUPER_OFF_PEAK,
            weekdays=list(range(7)),  # 매일
            seasons=list(Season)
        ))

    def get_rate(self, dt: datetime) -> float:
        """특정 시간의 요금율 반환"""
        period = self.get_period(dt)
        return self.rates.get(period, self.base_rate)

    def get_period(self, dt: datetime) -> PricePeriod:
        """특정 시간의 요금 시간대 반환"""
        for schedule in self.schedules:
            if schedule.is_applicable(dt):
                return schedule.period

        # 기본값은 일반 시간대
        return PricePeriod.NORMAL


class SeasonalTOUModel(BaseTOUModel):
    """계절별 TOU 모델"""

    def __init__(self):
        super().__init__("seasonal_tou")
        self._setup_seasonal_schedule()

    def _setup_seasonal_schedule(self):
        """계절별 TOU 스케줄 설정"""
        # 계절별 기본 요금율
        self.seasonal_rates = {
            Season.SUMMER: {
                PricePeriod.PEAK: 1.8 * self.base_rate,      # 여름 피크 (에어컨 사용 증가)
                PricePeriod.OFF_PEAK: 0.9 * self.base_rate,
                PricePeriod.NORMAL: 1.1 * self.base_rate,
                PricePeriod.SUPER_OFF_PEAK: 0.5 * self.base_rate
            },
            Season.WINTER: {
                PricePeriod.PEAK: 1.6 * self.base_rate,      # 겨울 피크 (난방 사용 증가)
                PricePeriod.OFF_PEAK: 0.9 * self.base_rate,
                PricePeriod.NORMAL: 1.0 * self.base_rate,
                PricePeriod.SUPER_OFF_PEAK: 0.6 * self.base_rate
            },
            Season.SPRING: {
                PricePeriod.PEAK: 1.4 * self.base_rate,
                PricePeriod.OFF_PEAK: 0.8 * self.base_rate,
                PricePeriod.NORMAL: 1.0 * self.base_rate,
                PricePeriod.SUPER_OFF_PEAK: 0.7 * self.base_rate
            },
            Season.AUTUMN: {
                PricePeriod.PEAK: 1.4 * self.base_rate,
                PricePeriod.OFF_PEAK: 0.8 * self.base_rate,
                PricePeriod.NORMAL: 1.0 * self.base_rate,
                PricePeriod.SUPER_OFF_PEAK: 0.7 * self.base_rate
            }
        }

        # 여름철 확장 피크 시간 (오전 10시 ~ 오후 5시, 오후 7시 ~ 9시)
        self.schedules.append(TOUSchedule(
            start_time=time(10, 0),
            end_time=time(17, 0),
            period=PricePeriod.PEAK,
            weekdays=[0, 1, 2, 3, 4],
            seasons=[Season.SUMMER]
        ))

        self.schedules.append(TOUSchedule(
            start_time=time(19, 0),
            end_time=time(21, 0),
            period=PricePeriod.PEAK,
            weekdays=[0, 1, 2, 3, 4],
            seasons=[Season.SUMMER]
        ))

        # 겨울철 피크 시간 (오전 8시 ~ 오전 10시, 오후 5시 ~ 오후 8시)
        self.schedules.append(TOUSchedule(
            start_time=time(8, 0),
            end_time=time(10, 0),
            period=PricePeriod.PEAK,
            weekdays=[0, 1, 2, 3, 4],
            seasons=[Season.WINTER]
        ))

        self.schedules.append(TOUSchedule(
            start_time=time(17, 0),
            end_time=time(20, 0),
            period=PricePeriod.PEAK,
            weekdays=[0, 1, 2, 3, 4],
            seasons=[Season.WINTER]
        ))

        # 기본 피크 시간 (봄, 가을)
        self.schedules.append(TOUSchedule(
            start_time=time(9, 0),
            end_time=time(18, 0),
            period=PricePeriod.PEAK,
            weekdays=[0, 1, 2, 3, 4],
            seasons=[Season.SPRING, Season.AUTUMN]
        ))

        # 심야 시간대 (모든 계절 공통)
        self.schedules.append(TOUSchedule(
            start_time=time(23, 0),
            end_time=time(6, 0),
            period=PricePeriod.SUPER_OFF_PEAK,
            weekdays=list(range(7)),
            seasons=list(Season)
        ))

    def get_rate(self, dt: datetime) -> float:
        """계절별 요금율 반환"""
        season = self._get_season(dt)
        period = self.get_period(dt)

        return self.seasonal_rates[season].get(period, self.base_rate)

    def get_period(self, dt: datetime) -> PricePeriod:
        """계절별 요금 시간대 반환"""
        for schedule in self.schedules:
            if schedule.is_applicable(dt):
                return schedule.period

        # 기본값은 일반 시간대
        return PricePeriod.NORMAL

    def _get_season(self, dt: datetime) -> Season:
        """계절 판별"""
        month = dt.month

        if month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        elif month in [9, 10, 11]:
            return Season.AUTUMN
        else:
            return Season.WINTER


class DynamicTOUModel(BaseTOUModel):
    """동적 TOU 모델 (실시간 요금 조정)"""

    def __init__(self):
        super().__init__("dynamic_tou")
        self.base_model = StandardTOUModel()
        self.demand_factor = 1.0
        self.supply_factor = 1.0
        self.price_history = []

    def update_market_conditions(self, demand_level: float, supply_level: float):
        """시장 상황 업데이트"""
        # 수요/공급 비율에 따른 가격 조정 팩터
        self.demand_factor = max(0.5, min(2.0, demand_level))
        self.supply_factor = max(0.5, min(2.0, 1.0 / supply_level))

        self.logger.info(f"시장 상황 업데이트: 수요 팩터 {self.demand_factor:.2f}, 공급 팩터 {self.supply_factor:.2f}")

    def get_rate(self, dt: datetime) -> float:
        """동적 요금율 계산"""
        base_rate = self.base_model.get_rate(dt)

        # 동적 조정 팩터 적용
        dynamic_factor = (self.demand_factor + self.supply_factor) / 2

        # 시간대별 추가 조정
        hour = dt.hour
        if 12 <= hour <= 14:  # 점심 시간대
            dynamic_factor *= 1.1
        elif 18 <= hour <= 20:  # 저녁 피크
            dynamic_factor *= 1.2
        elif 2 <= hour <= 5:   # 심야
            dynamic_factor *= 0.8

        adjusted_rate = base_rate * dynamic_factor

        # 가격 이력 저장
        self.price_history.append({
            'timestamp': dt,
            'base_rate': base_rate,
            'dynamic_factor': dynamic_factor,
            'final_rate': adjusted_rate
        })

        # 이력 크기 제한
        if len(self.price_history) > 1000:
            self.price_history.pop(0)

        return adjusted_rate

    def get_period(self, dt: datetime) -> PricePeriod:
        """기본 모델의 시간대 사용"""
        return self.base_model.get_period(dt)

    def get_price_forecast(self, hours: int = 24) -> List[Dict]:
        """가격 예측"""
        forecast = []
        current_time = datetime.now()

        for hour in range(hours):
            forecast_time = current_time + timedelta(hours=hour)

            # 간단한 예측 (실제로는 더 복잡한 모델 사용)
            predicted_rate = self.get_rate(forecast_time)

            # 불확실성 추가
            uncertainty = 0.1 * predicted_rate  # 10% 불확실성

            forecast.append({
                'timestamp': forecast_time.isoformat(),
                'predicted_rate': predicted_rate,
                'confidence_interval': [
                    predicted_rate - uncertainty,
                    predicted_rate + uncertainty
                ],
                'period': self.get_period(forecast_time).value
            })

        return forecast


class TOUPricingModel:
    """TOU 요금제 통합 모델"""

    def __init__(self, model_type: str = "standard"):
        self.logger = get_logger("tou_pricing")
        self.model_type = model_type

        # 모델 선택
        if model_type == "standard":
            self.model = StandardTOUModel()
        elif model_type == "seasonal":
            self.model = SeasonalTOUModel()
        elif model_type == "dynamic":
            self.model = DynamicTOUModel()
        else:
            raise TOUPricingError(f"지원하지 않는 TOU 모델 타입: {model_type}")

        self.logger.info(f"TOU 모델 초기화: {model_type}")

    def get_rate(self, dt: datetime) -> float:
        """요금율 조회"""
        return self.model.get_rate(dt)

    def get_period(self, dt: datetime) -> PricePeriod:
        """요금 시간대 조회"""
        return self.model.get_period(dt)

    def calculate_cost(self, start_time: datetime, duration_hours: float, power_kw: float) -> float:
        """비용 계산"""
        return self.model.calculate_cost(start_time, duration_hours, power_kw)

    def get_average_rate(self, start_time: datetime, duration_minutes: int) -> float:
        """평균 요금율"""
        return self.model.get_average_rate(start_time, duration_minutes)

    def find_cheapest_period(self, duration_hours: float, start_window: datetime,
                           end_window: datetime, power_kw: float = 1.0) -> Tuple[datetime, float]:
        """최저 비용 시간대 찾기"""
        return self.model.find_cheapest_period(duration_hours, start_window, end_window, power_kw)

    def get_daily_profile(self, date: datetime = None) -> pd.DataFrame:
        """일일 요금 프로파일"""
        if date is None:
            date = datetime.now()
        return self.model.get_daily_profile(date)

    def analyze_cost_savings(self,
                           current_schedule: List[Dict],
                           alternative_schedule: List[Dict]) -> Dict:
        """비용 절감 분석"""
        current_cost = 0.0
        alternative_cost = 0.0

        # 현재 스케줄 비용
        for item in current_schedule:
            start_time = datetime.fromisoformat(item['start_time'])
            duration = item['duration_hours']
            power = item['power_kw']
            current_cost += self.calculate_cost(start_time, duration, power)

        # 대안 스케줄 비용
        for item in alternative_schedule:
            start_time = datetime.fromisoformat(item['start_time'])
            duration = item['duration_hours']
            power = item['power_kw']
            alternative_cost += self.calculate_cost(start_time, duration, power)

        savings = current_cost - alternative_cost
        savings_percentage = (savings / current_cost * 100) if current_cost > 0 else 0

        return {
            'current_cost': current_cost,
            'alternative_cost': alternative_cost,
            'absolute_savings': savings,
            'percentage_savings': savings_percentage,
            'is_beneficial': savings > 0
        }

    def get_optimization_recommendations(self,
                                       jobs: List[Dict],
                                       flexibility_hours: int = 8) -> List[Dict]:
        """최적화 권장사항"""
        recommendations = []

        for job in jobs:
            original_start = datetime.fromisoformat(job['start_time'])
            duration = job['duration_hours']
            power = job['power_kw']

            # 유연성 윈도우 내에서 최적 시간 찾기
            window_start = original_start - timedelta(hours=flexibility_hours//2)
            window_end = original_start + timedelta(hours=flexibility_hours//2)

            optimal_start, optimal_cost = self.find_cheapest_period(
                duration, window_start, window_end, power
            )

            original_cost = self.calculate_cost(original_start, duration, power)
            savings = original_cost - optimal_cost

            if savings > 0:
                recommendations.append({
                    'job_id': job.get('job_id', 'unknown'),
                    'original_start': original_start.isoformat(),
                    'recommended_start': optimal_start.isoformat(),
                    'original_cost': original_cost,
                    'optimized_cost': optimal_cost,
                    'savings': savings,
                    'savings_percentage': (savings / original_cost * 100),
                    'time_shift_hours': (optimal_start - original_start).total_seconds() / 3600
                })

        # 절감액 순으로 정렬
        recommendations.sort(key=lambda x: x['savings'], reverse=True)

        return recommendations

    def get_statistics(self) -> Dict:
        """통계 정보"""
        base_stats = self.model.get_statistics()
        base_stats['model_type'] = self.model_type
        return base_stats


# 팩토리 함수들
def create_tou_model(model_type: str = "standard") -> TOUPricingModel:
    """TOU 모델 생성"""
    return TOUPricingModel(model_type)


def calculate_electricity_cost(start_time: datetime,
                             duration_hours: float,
                             power_kw: float,
                             model_type: str = "standard") -> float:
    """편의 함수: 전력 비용 계산"""
    model = create_tou_model(model_type)
    return model.calculate_cost(start_time, duration_hours, power_kw)


# 사용 예시
if __name__ == "__main__":
    # TOU 모델 생성
    tou_model = create_tou_model("seasonal")

    # 현재 시간 요금 조회
    now = datetime.now()
    current_rate = tou_model.get_rate(now)
    current_period = tou_model.get_period(now)

    print(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"현재 요금: {current_rate:.2f}원/kWh")
    print(f"현재 시간대: {current_period.value}")

    # 일일 요금 프로파일
    daily_profile = tou_model.get_daily_profile()
    print(f"\n오늘의 요금 프로파일:")
    print(daily_profile.to_string(index=False))

    # 비용 계산 예시
    start_time = datetime(2024, 8, 18, 14, 0)  # 오후 2시 시작
    duration = 3.0  # 3시간
    power = 50.0    # 50kW

    cost = tou_model.calculate_cost(start_time, duration, power)
    print(f"\n비용 계산 예시:")
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"지속 시간: {duration}시간")
    print(f"전력: {power}kW")
    print(f"총 비용: {cost:,.0f}원")

    # 최적 시간대 찾기
    window_start = datetime(2024, 8, 18, 8, 0)
    window_end = datetime(2024, 8, 18, 20, 0)

    optimal_start, optimal_cost = tou_model.find_cheapest_period(
        duration, window_start, window_end, power
    )

    print(f"\n최적화 결과:")
    print(f"최적 시작 시간: {optimal_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"최적 비용: {optimal_cost:,.0f}원")
    print(f"절감액: {cost - optimal_cost:,.0f}원")
    print(f"절감률: {(cost - optimal_cost) / cost * 100:.1f}%")