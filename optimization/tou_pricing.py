"""
TOU (Time-of-Use) Pricing Model for Korean Industrial Electricity Rates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TOUPricingModel:
    """TOU(Time-of-Use) 요금제 모델 - 실제 요금표 기반"""

    def __init__(self):
        self.price_schedule = {}
        self.peak_periods = []
        self.off_peak_periods = []
        self.seasonal_rates = {}
        self.demand_charges = {}
        self.power_industry_fund = 3.7  # 전력산업기반기금 (원/kWh)
        self.vat_rate = 0.1  # 부가가치세 10%

    def setup_korean_industrial_rates(self) -> Dict:
        """한국전력 산업용 일반(갑) 요금표 기반 설정"""

        # 2024년 기준 한국전력 산업용 요금표
        self.price_schedule = {
            'peak': {
                'name': '최대부하시간',
                'hours': list(range(10, 12)) + list(range(13, 17)),  # 10-12시, 13-17시
                'summer_price': 109.40,  # 하절기 (6~8월)
                'spring_fall_price': 78.90,  # 춘추계 (3~5월, 9~11월)
                'winter_price': 78.90,  # 동절기 (12~2월)
                'description': '전력수요가 최대인 시간대'
            },
            'light_load': {
                'name': '경부하시간',
                'hours': list(range(23, 24)) + list(range(0, 9)),  # 23-09시
                'summer_price': 58.80,
                'spring_fall_price': 58.80,
                'winter_price': 58.80,
                'description': '전력수요가 적은 야간시간대'
            },
            'mid_load': {
                'name': '중간부하시간',
                'hours': list(range(9, 10)) + [12] + list(range(17, 23)),  # 나머지 시간
                'summer_price': 82.40,
                'spring_fall_price': 58.80,
                'winter_price': 82.40,
                'description': '최대부하와 경부하 사이 시간대'
            }
        }

        # 기본요금 (원/kW)
        self.demand_charges = {
            'contract_power': 8320,  # 계약전력 기본요금
            'peak_demand_summer': 8900,  # 하절기 최대수요전력 요금
            'peak_demand_other': 5950,  # 기타계절 최대수요전력 요금
        }

        # 계절별 구분
        self.seasonal_rates = {
            'summer': [6, 7, 8],  # 6~8월
            'winter': [12, 1, 2],  # 12~2월
            'spring_fall': [3, 4, 5, 9, 10, 11]  # 3~5월, 9~11월
        }

        logger.info("✅ 한국전력 산업용 TOU 요금제 설정 완료")
        return self.price_schedule

    def setup_custom_rates(self, custom_config: Dict) -> Dict:
        """사용자 정의 요금제 설정"""
        self.price_schedule = custom_config.get('price_schedule', {})
        self.demand_charges = custom_config.get('demand_charges', {})
        self.seasonal_rates = custom_config.get('seasonal_rates', {})

        logger.info("✅ 사용자 정의 요금제 설정 완료")
        return self.price_schedule

    def get_season_from_month(self, month: int) -> str:
        """월에 따른 계절 구분"""
        if month in self.seasonal_rates.get('summer', [6, 7, 8]):
            return 'summer'
        elif month in self.seasonal_rates.get('winter', [12, 1, 2]):
            return 'winter'
        else:
            return 'spring_fall'

    def get_hourly_price(self, hour: int, month: int = 7, include_taxes: bool = True) -> Dict:
        """특정 시간의 전력 요금 반환 (실제 요금표 기반)"""

        # 계절 결정
        season = self.get_season_from_month(month)

        # 시간대별 요금 결정
        base_price = 67.2  # 기본값 (경부하)
        period_type = 'light_load'

        for period, config in self.price_schedule.items():
            if hour in config['hours']:
                period_type = period
                if season == 'summer':
                    base_price = config.get('summer_price', config.get('price', 67.2))
                elif season == 'winter':
                    base_price = config.get('winter_price', config.get('price', 67.2))
                else:
                    base_price = config.get('spring_fall_price', config.get('price', 67.2))
                break

        # 전력산업기반기금 추가
        total_price = base_price + self.power_industry_fund

        # 부가가치세 추가 (선택적)
        if include_taxes:
            total_price *= (1 + self.vat_rate)

        return {
            'base_price': base_price,
            'fund': self.power_industry_fund,
            'total_price': total_price,
            'period': period_type,
            'season': season
        }

    def calculate_daily_cost(self, hourly_consumption: List[float],
                             month: int = 7, contract_power: float = 1000) -> Dict:
        """일일 전력 비용 계산 (실제 요금표 기반)"""

        total_energy_cost = 0
        total_consumption = 0
        peak_demand = max(hourly_consumption)
        season = self.get_season_from_month(month)

        # 시간당 전력비 계산
        for hour, consumption in enumerate(hourly_consumption):
            if consumption > 0:
                price_info = self.get_hourly_price(hour, month)
                hourly_cost = consumption * price_info['total_price']
                total_energy_cost += hourly_cost
                total_consumption += consumption

        # 기본요금 계산 (일할 계산)
        daily_basic_charge = (contract_power * self.demand_charges['contract_power']) / 30

        # 최대수요전력 요금 (일할 계산)
        if season == 'summer':
            daily_demand_charge = (peak_demand * self.demand_charges['peak_demand_summer']) / 30
        else:
            daily_demand_charge = (peak_demand * self.demand_charges['peak_demand_other']) / 30

        # 총 비용
        total_cost = total_energy_cost + daily_basic_charge + daily_demand_charge

        return {
            'energy_cost': total_energy_cost,
            'basic_charge': daily_basic_charge,
            'demand_charge': daily_demand_charge,
            'total_cost': total_cost,
            'peak_demand': peak_demand,
            'total_consumption': total_consumption,
            'season': season,
            'average_price': total_energy_cost / total_consumption if total_consumption > 0 else 0
        }

    def get_period_summary(self) -> Dict:
        """요금제 정보 요약"""
        summary = {
            'periods': {},
            'demand_charges': self.demand_charges,
            'seasonal_info': self.seasonal_rates
        }

        for period, config in self.price_schedule.items():
            summary['periods'][period] = {
                'name': config.get('name', period),
                'hours': config['hours'],
                'price_range': {
                    'summer': config.get('summer_price', config.get('price', 0)),
                    'winter': config.get('winter_price', config.get('price', 0)),
                    'spring_fall': config.get('spring_fall_price', config.get('price', 0))
                },
                'description': config.get('description', '')
            }

        return summary

    def print_rate_schedule(self):
        """요금표 출력"""
        print("\n📋 전력 요금표")
        print("=" * 50)

        # 시간대별 요금
        print("\n⏰ 시간대별 전력량 요금 (원/kWh, VAT 포함)")
        print("-" * 50)

        for period, config in self.price_schedule.items():
            name = config.get('name', period)
            hours = config['hours']

            print(f"\n🔸 {name}")
            print(f"   시간대: {min(hours):02d}:00 ~ {max(hours) + 1:02d}:00")
            print(f"   하절기: {config.get('summer_price', 0):.2f}원")
            print(f"   동절기: {config.get('winter_price', 0):.2f}원")
            print(f"   춘추계: {config.get('spring_fall_price', 0):.2f}원")
            if 'description' in config:
                print(f"   설명: {config['description']}")

        # 기본요금
        print(f"\n💰 기본요금")
        print("-" * 30)
        for charge_type, amount in self.demand_charges.items():
            charge_name = {
                'contract_power': '계약전력',
                'peak_demand_summer': '최대수요전력(하절기)',
                'peak_demand_other': '최대수요전력(기타계절)'
            }.get(charge_type, charge_type)
            print(f"   {charge_name}: {amount:,}원/kW")

        # 기타 요금
        print(f"\n📊 기타")
        print("-" * 20)
        print(f"   전력산업기반기금: {self.power_industry_fund}원/kWh")
        print(f"   부가가치세: {self.vat_rate * 100}%")

    def load_price_table_from_excel(self, file_path: Optional[str] = None) -> Optional[Dict]:
        """Excel 요금표 파일 로드"""
        try:
            if file_path:
                # 실제 Excel 파일에서 로드
                df = pd.read_excel(file_path)
                return self._parse_excel_price_data(df)
            else:
                # 기본 한국전력 산업용 요금표 사용
                return self.setup_korean_industrial_rates()
        except Exception as e:
            logger.warning(f"⚠️ 요금표 로드 실패: {e}")
            return self.setup_korean_industrial_rates()

    def _parse_excel_price_data(self, df: pd.DataFrame) -> Dict:
        """Excel 파일에서 요금 데이터 파싱"""
        # Excel 파일 구조에 따라 파싱 로직 구현
        # 현재는 기본 요금표 반환
        logger.info("Excel 파일 파싱 중...")
        return self.setup_korean_industrial_rates()

    def validate_price_schedule(self) -> bool:
        """요금 스케줄 유효성 검증"""
        try:
            # 24시간 모든 시간이 커버되는지 확인
            all_hours = set()
            for period, config in self.price_schedule.items():
                hours = config.get('hours', [])
                all_hours.update(hours)

            missing_hours = set(range(24)) - all_hours
            if missing_hours:
                logger.error(f"❌ 누락된 시간: {sorted(missing_hours)}")
                return False

            # 중복 시간 확인
            hour_periods = {}
            for period, config in self.price_schedule.items():
                for hour in config.get('hours', []):
                    if hour in hour_periods:
                        logger.error(f"❌ 시간 {hour}가 중복됨: {period} vs {hour_periods[hour]}")
                        return False
                    hour_periods[hour] = period

            logger.info("✅ 요금 스케줄 유효성 검증 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 요금 스케줄 검증 실패: {e}")
            return False