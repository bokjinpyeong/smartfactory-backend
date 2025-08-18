"""
스마트팩토리 에너지 관리 시스템 - 메인 실행 파일

전체 시스템 통합 및 실행
- 시스템 초기화 및 설정
- 각 모듈 간 연동
- CLI 인터페이스 제공
- 서비스 관리
"""

import asyncio
import argparse
import sys
import signal
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# FastAPI 및 Uvicorn
from fastapi import FastAPI
import uvicorn

# Core 모듈
from core.config import get_config, initialize_core_system
from core.logger import get_logger, setup_logging
from core.exceptions import SmartFactoryException

# 기능 모듈들
from data.collector import create_collection_manager, create_mqtt_collector
from data.processor import create_default_pipeline
from data.validator import create_validator, ValidationLevel
from models import get_model_manager, create_model
from optimization.scheduler import create_scheduler
from optimization.tou_pricing import create_tou_model
from optimization.constraints import create_constraint_manager

# API 및 대시보드
from api.routes import app as api_app
from dashboard.real_time import get_dashboard_manager


class SmartFactorySystem:
    """스마트팩토리 에너지 관리 시스템"""

    def __init__(self):
        self.logger = None
        self.config = None
        self.is_running = False
        self.components = {}

        # 종료 신호 처리
        self.shutdown_event = asyncio.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """시스템 종료 신호 처리"""
        print(f"\n시스템 종료 신호 수신: {signum}")
        self.shutdown_event.set()

    async def initialize(self, env: str = "development", log_level: str = "INFO"):
        """시스템 초기화"""
        try:
            # Core 시스템 초기화
            self.config, self.logger = initialize_core_system(env, log_level)

            self.logger.info("🚀 스마트팩토리 에너지 관리 시스템 초기화 시작")
            self.logger.info(f"환경: {env}, 로그 레벨: {log_level}")

            # 각 컴포넌트 초기화
            await self._initialize_components()

            self.logger.info("✅ 시스템 초기화 완료")
            return True

        except Exception as e:
            print(f"❌ 시스템 초기화 실패: {e}")
            if self.logger:
                self.logger.error(f"시스템 초기화 실패: {e}")
            return False

    async def _initialize_components(self):
        """각 컴포넌트 초기화"""
        self.logger.info("컴포넌트 초기화 시작...")

        try:
            # 1. 데이터 수집 시스템
            self.logger.info("📊 데이터 수집 시스템 초기화")
            collection_manager = create_collection_manager()

            # MQTT 수집기 추가
            mqtt_collector = create_mqtt_collector()
            collection_manager.add_collector(mqtt_collector)

            self.components['data_collection'] = collection_manager

            # 2. 데이터 처리 파이프라인
            self.logger.info("⚙️ 데이터 처리 파이프라인 초기화")
            processing_pipeline = create_default_pipeline()
            self.components['data_processing'] = processing_pipeline

            # 3. 데이터 검증기
            self.logger.info("🔍 데이터 검증기 초기화")
            data_validator = create_validator(ValidationLevel.STANDARD)
            self.components['data_validation'] = data_validator

            # 4. AI 모델 시스템
            self.logger.info("🤖 AI 모델 시스템 초기화")
            model_manager = get_model_manager()

            # 기본 모델들 생성
            anomaly_model = create_model("anomaly_detector", "main_anomaly_detector")
            power_model = create_model("power_predictor", "main_power_predictor")

            model_manager.register_model("anomaly_detector", anomaly_model)
            model_manager.register_model("power_predictor", power_model)

            self.components['models'] = model_manager

            # 5. 스케줄링 시스템
            self.logger.info("📅 스케줄링 시스템 초기화")
            scheduler = create_scheduler("greedy")
            self.components['scheduler'] = scheduler

            # 6. TOU 요금제 시스템
            self.logger.info("💰 TOU 요금제 시스템 초기화")
            tou_model = create_tou_model("standard")
            self.components['tou_pricing'] = tou_model

            # 7. 제약 조건 관리
            self.logger.info("🔒 제약 조건 관리자 초기화")
            constraint_manager = create_constraint_manager()
            self.components['constraints'] = constraint_manager

            # 8. 실시간 대시보드
            self.logger.info("📈 실시간 대시보드 초기화")
            dashboard_manager = get_dashboard_manager()
            await dashboard_manager.start()
            self.components['dashboard'] = dashboard_manager

            self.logger.info(f"✅ {len(self.components)}개 컴포넌트 초기화 완료")

        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 오류: {e}")
            raise

    async def start_services(self):
        """서비스 시작"""
        try:
            self.is_running = True
            self.logger.info("🚀 서비스 시작")

            # 데이터 수집 시작
            if 'data_collection' in self.components:
                await self.components['data_collection'].start_all_collectors()
                self.logger.info("📊 데이터 수집 서비스 시작")

            # 샘플 데이터로 모델 학습 (데모용)
            await self._train_demo_models()

            self.logger.info("✅ 모든 서비스 시작 완료")

        except Exception as e:
            self.logger.error(f"서비스 시작 오류: {e}")
            raise

    async def _train_demo_models(self):
        """데모용 모델 학습"""
        try:
            self.logger.info("🎯 데모 모델 학습 시작")

            # 샘플 데이터 생성
            import pandas as pd
            import numpy as np

            sample_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
                'machine_id': np.random.choice(['machine_1', 'machine_2', 'machine_3'], 1000),
                'power_consumption': np.random.normal(100, 20, 1000),
                'voltage': np.random.normal(220, 10, 1000),
                'current': np.random.normal(0.5, 0.1, 1000),
                'temperature': np.random.normal(25, 5, 1000)
            })

            # 데이터 처리
            if 'data_processing' in self.components:
                pipeline = self.components['data_processing']
                pipeline.fit(sample_data)
                processed_result = pipeline.transform(sample_data)
                self.logger.info(f"데이터 처리 완료: {processed_result.data.shape}")

            # 모델 학습 (간단한 예시)
            model_manager = self.components.get('models')
            if model_manager:
                for model_name, model in model_manager.models.items():
                    if hasattr(model, 'train'):
                        try:
                            # 실제 모델별 학습 로직 필요
                            self.logger.info(f"모델 학습: {model_name}")
                        except Exception as e:
                            self.logger.warning(f"모델 학습 실패 [{model_name}]: {e}")

            self.logger.info("✅ 데모 모델 학습 완료")

        except Exception as e:
            self.logger.error(f"데모 모델 학습 오류: {e}")

    async def stop_services(self):
        """서비스 중지"""
        try:
            self.is_running = False
            self.logger.info("🛑 서비스 중지 시작")

            # 데이터 수집 중지
            if 'data_collection' in self.components:
                await self.components['data_collection'].stop_all_collectors()
                self.logger.info("📊 데이터 수집 서비스 중지")

            # 대시보드 중지
            if 'dashboard' in self.components:
                await self.components['dashboard'].stop()
                self.logger.info("📈 대시보드 서비스 중지")

            self.logger.info("✅ 모든 서비스 중지 완료")

        except Exception as e:
            self.logger.error(f"서비스 중지 오류: {e}")

    async def run_api_server(self, host: str = None, port: int = None, workers: int = 1):
        """API 서버 실행"""
        host = host or self.config.system.api_host
        port = port or self.config.system.api_port

        self.logger.info(f"🌐 API 서버 시작: http://{host}:{port}")

        config = uvicorn.Config(
            api_app,
            host=host,
            port=port,
            log_level="info",
            workers=workers
        )

        server = uvicorn.Server(config)

        # 백그라운드에서 API 서버 실행
        server_task = asyncio.create_task(server.serve())

        # 종료 신호 대기
        await self.shutdown_event.wait()

        # 서버 종료
        server.should_exit = True
        await server_task

    async def run_data_processing_loop(self):
        """데이터 처리 루프 실행"""
        self.logger.info("🔄 데이터 처리 루프 시작")

        while self.is_running and not self.shutdown_event.is_set():
            try:
                # 실제 구현에서는 큐에서 데이터를 가져와 처리
                await asyncio.sleep(30)  # 30초마다 처리

                # 시스템 상태 로깅
                stats = self.get_system_status()
                self.logger.debug(f"시스템 상태: {stats['summary']}")

            except Exception as e:
                self.logger.error(f"데이터 처리 루프 오류: {e}")
                await asyncio.sleep(5)

        self.logger.info("🔄 데이터 처리 루프 종료")

    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "is_running": self.is_running,
            "components": {}
        }

        # 각 컴포넌트 상태
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_statistics'):
                    status["components"][name] = component.get_statistics()
                elif hasattr(component, 'stats'):
                    status["components"][name] = component.stats
                else:
                    status["components"][name] = {"status": "active"}
            except Exception as e:
                status["components"][name] = {"status": "error", "error": str(e)}

        # 요약 정보
        status["summary"] = {
            "total_components": len(self.components),
            "active_components": sum(1 for comp in status["components"].values()
                                     if comp.get("status") != "error"),
            "models_count": len(self.components.get('models', {}).models) if 'models' in self.components else 0
        }

        return status

    async def run_interactive_mode(self):
        """대화형 모드 실행"""
        print("\n🎯 스마트팩토리 에너지 관리 시스템 - 대화형 모드")
        print("사용 가능한 명령어:")
        print("  status  - 시스템 상태 조회")
        print("  stats   - 통계 정보 조회")
        print("  help    - 도움말")
        print("  exit    - 종료")
        print()

        while self.is_running and not self.shutdown_event.is_set():
            try:
                command = input("smartfactory> ").strip().lower()

                if command == "exit":
                    break
                elif command == "status":
                    status = self.get_system_status()
                    print(json.dumps(status["summary"], indent=2, ensure_ascii=False))
                elif command == "stats":
                    status = self.get_system_status()
                    print(json.dumps(status["components"], indent=2, ensure_ascii=False))
                elif command == "help":
                    print("사용 가능한 명령어: status, stats, help, exit")
                elif command:
                    print(f"알 수 없는 명령어: {command}")

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"명령어 처리 오류: {e}")

        print("대화형 모드 종료")


async def run_full_system(args):
    """전체 시스템 실행"""
    system = SmartFactorySystem()

    try:
        # 시스템 초기화
        if not await system.initialize(args.env, args.log_level):
            return 1

        # 서비스 시작
        await system.start_services()

        # 실행 모드에 따른 처리
        if args.mode == "api":
            # API 서버만 실행
            await system.run_api_server(args.host, args.port, args.workers)

        elif args.mode == "processing":
            # 데이터 처리만 실행
            await system.run_data_processing_loop()

        elif args.mode == "interactive":
            # 대화형 모드
            tasks = await asyncio.gather(
                system.run_data_processing_loop(),
                system.run_interactive_mode(),
                return_exceptions=True
            )

        else:  # full mode
            # 전체 시스템 실행
            tasks = await asyncio.gather(
                system.run_api_server(args.host, args.port, args.workers),
                system.run_data_processing_loop(),
                return_exceptions=True
            )

    except Exception as e:
        print(f"시스템 실행 오류: {e}")
        return 1

    finally:
        # 시스템 정리
        await system.stop_services()
        print("시스템 종료 완료")

    return 0


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="스마트팩토리 에너지 관리 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py --mode full                    # 전체 시스템 실행
  python main.py --mode api --port 8080        # API 서버만 실행
  python main.py --mode processing             # 데이터 처리만 실행
  python main.py --mode interactive            # 대화형 모드
  python main.py --env production              # 운영 환경으로 실행
        """
    )

    parser.add_argument(
        "--mode",
        choices=["full", "api", "processing", "interactive"],
        default="full",
        help="실행 모드 (기본값: full)"
    )

    parser.add_argument(
        "--env",
        choices=["development", "production", "testing"],
        default="development",
        help="실행 환경 (기본값: development)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="로그 레벨 (기본값: INFO)"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API 서버 호스트 (기본값: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API 서버 포트 (기본값: 8000)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="API 서버 워커 수 (기본값: 1)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="스마트팩토리 에너지 관리 시스템 v1.0.0"
    )

    args = parser.parse_args()

    # 시스템 실행
    try:
        exit_code = asyncio.run(run_full_system(args))
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n시스템 중단됨")
        sys.exit(0)

    except Exception as e:
        print(f"시스템 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()