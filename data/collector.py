"""
스마트팩토리 에너지 관리 시스템 - IoT 데이터 수집 모듈

확장 가능한 실시간 센서 데이터 수집 시스템
- MQTT 기반 실시간 데이터 수집
- 다중 센서 동시 처리
- 장애 허용성 및 재연결 로직
- 배치/스트리밍 하이브리드 처리
"""

import json
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import pandas as pd
import numpy as np

# MQTT 클라이언트
import paho.mqtt.client as mqtt

# Core 모듈
from core.config import get_config
from core.logger import get_logger, log_performance
from core.exceptions import (
    SensorDataError, SensorConnectionError,
    DeviceOfflineError, DataValidationError,
    safe_execute
)


@dataclass
class SensorReading:
    """센서 읽기 데이터 클래스"""
    sensor_id: str
    machine_id: str
    timestamp: datetime
    power_consumption: float
    voltage: float
    current: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    status: str = "normal"
    raw_data: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def validate(self) -> bool:
        """데이터 유효성 검증"""
        # 필수 필드 검증
        if not all([self.sensor_id, self.machine_id, self.timestamp]):
            return False

        # 전력 데이터 범위 검증
        if self.power_consumption < 0 or self.power_consumption > 10000:  # 0-10kW
            return False

        if self.voltage < 0 or self.voltage > 500:  # 0-500V
            return False

        if self.current < 0 or self.current > 100:  # 0-100A
            return False

        return True


class BaseDataCollector(ABC):
    """데이터 수집기 기본 클래스"""

    def __init__(self, collector_id: str):
        self.collector_id = collector_id
        self.logger = get_logger(f"collector.{collector_id}")
        self.config = get_config()
        self.is_running = False
        self.callbacks: List[Callable] = []
        self.error_count = 0
        self.last_error_time = None

    @abstractmethod
    async def start_collection(self):
        """데이터 수집 시작"""
        pass

    @abstractmethod
    async def stop_collection(self):
        """데이터 수집 중지"""
        pass

    def add_callback(self, callback: Callable[[SensorReading], None]):
        """데이터 수집 콜백 추가"""
        self.callbacks.append(callback)

    def _notify_callbacks(self, reading: SensorReading):
        """콜백 알림"""
        for callback in self.callbacks:
            try:
                callback(reading)
            except Exception as e:
                self.logger.error(f"콜백 실행 오류: {e}")

    def _handle_error(self, error: Exception):
        """오류 처리"""
        self.error_count += 1
        self.last_error_time = datetime.now()
        self.logger.error(f"수집기 오류 [{self.error_count}]: {error}")


class MQTTDataCollector(BaseDataCollector):
    """MQTT 기반 데이터 수집기"""

    def __init__(self, collector_id: str = "mqtt_collector"):
        super().__init__(collector_id)
        self.client = None
        self.topics = []
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 10
        self.reconnect_count = 0

    def configure_mqtt(self,
                       broker_host: str = None,
                       broker_port: int = None,
                       topics: List[str] = None,
                       username: str = None,
                       password: str = None):
        """MQTT 설정"""
        self.broker_host = broker_host or self.config.system.mqtt_broker
        self.broker_port = broker_port or self.config.system.mqtt_port
        self.topics = topics or [f"{self.config.system.mqtt_topic_prefix}/+/+"]
        self.username = username
        self.password = password

        # MQTT 클라이언트 생성
        self.client = mqtt.Client(client_id=f"{self.collector_id}_{int(time.time())}")

        # 콜백 설정
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # 인증 설정
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        self.logger.info(f"MQTT 설정 완료: {self.broker_host}:{self.broker_port}")

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT 연결 콜백"""
        if rc == 0:
            self.logger.info("MQTT 브로커 연결 성공")
            self.reconnect_count = 0

            # 토픽 구독
            for topic in self.topics:
                client.subscribe(topic)
                self.logger.info(f"토픽 구독: {topic}")
        else:
            self.logger.error(f"MQTT 연결 실패: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """MQTT 연결 해제 콜백"""
        self.logger.warning(f"MQTT 연결 해제: {rc}")

        if self.is_running and self.reconnect_count < self.max_reconnect_attempts:
            self.reconnect_count += 1
            self.logger.info(f"재연결 시도 {self.reconnect_count}/{self.max_reconnect_attempts}")

            time.sleep(self.reconnect_delay)
            try:
                client.reconnect()
            except Exception as e:
                self.logger.error(f"재연결 실패: {e}")

    def _on_message(self, client, userdata, msg):
        """MQTT 메시지 수신 콜백"""
        try:
            # JSON 데이터 파싱
            payload = json.loads(msg.payload.decode('utf-8'))
            topic_parts = msg.topic.split('/')

            # 토픽에서 machine_id 추출
            machine_id = topic_parts[-1] if len(topic_parts) > 2 else "unknown"

            # SensorReading 객체 생성
            reading = self._parse_sensor_data(payload, machine_id)

            if reading and reading.validate():
                self._notify_callbacks(reading)
            else:
                raise DataValidationError("센서 데이터 유효성 검증 실패")

        except Exception as e:
            self._handle_error(e)

    def _parse_sensor_data(self, payload: Dict, machine_id: str) -> Optional[SensorReading]:
        """센서 데이터 파싱"""
        try:
            return SensorReading(
                sensor_id=payload.get('sensor_id', f"sensor_{machine_id}"),
                machine_id=machine_id,
                timestamp=datetime.fromisoformat(payload.get('timestamp', datetime.now().isoformat())),
                power_consumption=float(payload.get('power', 0)),
                voltage=float(payload.get('voltage', 0)),
                current=float(payload.get('current', 0)),
                temperature=payload.get('temperature'),
                humidity=payload.get('humidity'),
                status=payload.get('status', 'normal'),
                raw_data=payload
            )
        except Exception as e:
            self.logger.error(f"데이터 파싱 오류: {e}")
            return None

    async def start_collection(self):
        """데이터 수집 시작"""
        if not self.client:
            raise SensorConnectionError("MQTT 클라이언트가 설정되지 않음")

        self.is_running = True

        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            self.logger.info("MQTT 데이터 수집 시작")

        except Exception as e:
            self.is_running = False
            raise SensorConnectionError(f"MQTT 연결 실패: {e}")

    async def stop_collection(self):
        """데이터 수집 중지"""
        self.is_running = False

        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.logger.info("MQTT 데이터 수집 중지")


class FileDataCollector(BaseDataCollector):
    """파일 기반 데이터 수집기 (시뮬레이션용)"""

    def __init__(self, collector_id: str = "file_collector"):
        super().__init__(collector_id)
        self.file_path = None
        self.replay_speed = 1.0  # 1.0 = 실시간, 2.0 = 2배속
        self.loop_data = True
        self.current_index = 0
        self.data = None

    def load_data_file(self, file_path: str, replay_speed: float = 1.0):
        """데이터 파일 로드"""
        self.file_path = file_path
        self.replay_speed = replay_speed

        try:
            self.data = pd.read_csv(file_path)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.current_index = 0
            self.logger.info(f"데이터 파일 로드 완료: {len(self.data)} 레코드")

        except Exception as e:
            raise SensorDataError(f"파일 로드 실패: {e}")

    async def start_collection(self):
        """파일 데이터 재생 시작"""
        if self.data is None:
            raise SensorDataError("데이터 파일이 로드되지 않음")

        self.is_running = True
        self.logger.info("파일 데이터 수집 시작")

        while self.is_running:
            try:
                # 데이터 전송
                row = self.data.iloc[self.current_index]
                reading = self._row_to_sensor_reading(row)

                if reading and reading.validate():
                    self._notify_callbacks(reading)

                # 다음 인덱스
                self.current_index += 1
                if self.current_index >= len(self.data):
                    if self.loop_data:
                        self.current_index = 0
                        self.logger.info("데이터 파일 재시작")
                    else:
                        break

                # 재생 속도 조절
                await asyncio.sleep(1.0 / self.replay_speed)

            except Exception as e:
                self._handle_error(e)
                await asyncio.sleep(1)

    def _row_to_sensor_reading(self, row) -> Optional[SensorReading]:
        """DataFrame 행을 SensorReading으로 변환"""
        try:
            return SensorReading(
                sensor_id=str(row.get('sensor_id', 'file_sensor')),
                machine_id=str(row.get('machine_id', 'file_machine')),
                timestamp=row['timestamp'],
                power_consumption=float(row.get('power_consumption', 0)),
                voltage=float(row.get('voltage', 0)),
                current=float(row.get('current', 0)),
                temperature=row.get('temperature'),
                humidity=row.get('humidity'),
                status=str(row.get('status', 'normal'))
            )
        except Exception as e:
            self.logger.error(f"행 변환 오류: {e}")
            return None

    async def stop_collection(self):
        """파일 데이터 수집 중지"""
        self.is_running = False
        self.logger.info("파일 데이터 수집 중지")


class DataCollectionManager:
    """데이터 수집 관리자"""

    def __init__(self):
        self.logger = get_logger("collection_manager")
        self.config = get_config()
        self.collectors: Dict[str, BaseDataCollector] = {}
        self.data_buffer = queue.Queue(maxsize=10000)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_thread = None
        self.is_processing = False

        # 통계
        self.stats = {
            'total_readings': 0,
            'valid_readings': 0,
            'error_count': 0,
            'last_reading_time': None
        }

    def add_collector(self, collector: BaseDataCollector):
        """수집기 추가"""
        collector.add_callback(self._on_data_received)
        self.collectors[collector.collector_id] = collector
        self.logger.info(f"수집기 추가: {collector.collector_id}")

    def remove_collector(self, collector_id: str):
        """수집기 제거"""
        if collector_id in self.collectors:
            del self.collectors[collector_id]
            self.logger.info(f"수집기 제거: {collector_id}")

    def _on_data_received(self, reading: SensorReading):
        """데이터 수신 콜백"""
        try:
            self.data_buffer.put_nowait(reading)
            self.stats['total_readings'] += 1
            self.stats['last_reading_time'] = datetime.now()

        except queue.Full:
            self.logger.warning("데이터 버퍼 가득참")
            self.stats['error_count'] += 1

    def start_processing(self):
        """데이터 처리 시작"""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_data)
        self.processing_thread.start()
        self.logger.info("데이터 처리 시작")

    def stop_processing(self):
        """데이터 처리 중지"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
        self.logger.info("데이터 처리 중지")

    def _process_data(self):
        """데이터 처리 스레드"""
        batch_data = []
        batch_size = self.config.data.batch_size or 100

        while self.is_processing:
            try:
                # 배치 수집
                while len(batch_data) < batch_size and self.is_processing:
                    try:
                        reading = self.data_buffer.get(timeout=1)
                        batch_data.append(reading)
                        self.data_buffer.task_done()
                    except queue.Empty:
                        break

                # 배치 처리
                if batch_data:
                    self._process_batch(batch_data)
                    self.stats['valid_readings'] += len(batch_data)
                    batch_data.clear()

            except Exception as e:
                self.logger.error(f"데이터 처리 오류: {e}")
                self.stats['error_count'] += 1
                time.sleep(1)

    @log_performance
    def _process_batch(self, readings: List[SensorReading]):
        """배치 데이터 처리"""
        # DataFrame으로 변환
        df = pd.DataFrame([reading.to_dict() for reading in readings])

        # 여기서 추가 처리 로직 (전처리, 저장 등) 수행
        # 예: 데이터 검증, 이상치 제거, 데이터베이스 저장

        self.logger.debug(f"배치 처리 완료: {len(readings)} 레코드")

    async def start_all_collectors(self):
        """모든 수집기 시작"""
        tasks = []
        for collector in self.collectors.values():
            task = asyncio.create_task(collector.start_collection())
            tasks.append(task)

        self.start_processing()
        self.logger.info(f"{len(tasks)} 개 수집기 시작")

        # 수집기들이 모두 시작될 때까지 대기
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all_collectors(self):
        """모든 수집기 중지"""
        tasks = []
        for collector in self.collectors.values():
            task = asyncio.create_task(collector.stop_collection())
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)
        self.stop_processing()
        self.logger.info("모든 수집기 중지")

    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        return {
            **self.stats,
            'collectors_count': len(self.collectors),
            'buffer_size': self.data_buffer.qsize(),
            'is_processing': self.is_processing
        }


# 팩토리 함수들
def create_mqtt_collector(config: Dict = None) -> MQTTDataCollector:
    """MQTT 수집기 생성"""
    collector = MQTTDataCollector()

    if config:
        collector.configure_mqtt(**config)
    else:
        collector.configure_mqtt()

    return collector


def create_file_collector(file_path: str, replay_speed: float = 1.0) -> FileDataCollector:
    """파일 수집기 생성"""
    collector = FileDataCollector()
    collector.load_data_file(file_path, replay_speed)
    return collector


def create_collection_manager() -> DataCollectionManager:
    """수집 관리자 생성"""
    return DataCollectionManager()


# 사용 예시
if __name__ == "__main__":
    async def main():
        # 수집 관리자 생성
        manager = create_collection_manager()

        # MQTT 수집기 추가
        mqtt_collector = create_mqtt_collector()
        manager.add_collector(mqtt_collector)

        # 파일 수집기 추가 (시뮬레이션용)
        # file_collector = create_file_collector("data/sensor_data.csv", replay_speed=2.0)
        # manager.add_collector(file_collector)

        try:
            # 수집 시작
            await manager.start_all_collectors()

            # 통계 출력
            while True:
                stats = manager.get_statistics()
                print(f"통계: {stats}")
                await asyncio.sleep(10)

        except KeyboardInterrupt:
            print("수집 중지...")
            await manager.stop_all_collectors()


    asyncio.run(main())