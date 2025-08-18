"""
스마트팩토리 에너지 관리 시스템 - 실시간 대시보드

실시간 모니터링 및 대시보드 기능
- 실시간 데이터 스트리밍
- WebSocket 기반 실시간 업데이트
- 알람 및 알림 시스템
- 성능 메트릭 모니터링
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# WebSocket 지원
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

# Core 모듈
from core.config import get_config
from core.logger import get_logger
from core.exceptions import safe_execute

# 기능 모듈들
from models import get_model_manager
from data.collector import DataCollectionManager
from optimization.tou_pricing import create_tou_model


class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """메트릭 타입"""
    POWER_CONSUMPTION = "power_consumption"
    ENERGY_EFFICIENCY = "energy_efficiency"
    ANOMALY_SCORE = "anomaly_score"
    COST_RATE = "cost_rate"
    MACHINE_STATUS = "machine_status"
    JOB_PROGRESS = "job_progress"


@dataclass
class Alert:
    """알림 정보"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        return data


@dataclass
class RealTimeMetric:
    """실시간 메트릭"""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    machine_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['metric_type'] = self.metric_type.value
        return data


class ConnectionManager:
    """WebSocket 연결 관리자"""

    def __init__(self):
        self.logger = get_logger("dashboard.connections")
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, client_info: Dict = None):
        """클라이언트 연결"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = client_info or {}

        client_id = client_info.get('client_id', 'unknown') if client_info else 'unknown'
        self.logger.info(f"클라이언트 연결: {client_id} (총 {len(self.active_connections)}개)")

    def disconnect(self, websocket: WebSocket):
        """클라이언트 연결 해제"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_info = self.connection_info.pop(websocket, {})
            client_id = client_info.get('client_id', 'unknown')
            self.logger.info(f"클라이언트 연결 해제: {client_id} (총 {len(self.active_connections)}개)")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """개별 메시지 전송"""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(message)
        except Exception as e:
            self.logger.error(f"개별 메시지 전송 오류: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        """전체 브로드캐스트"""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_text(message)
                else:
                    disconnected.append(connection)
            except Exception as e:
                self.logger.error(f"브로드캐스트 오류: {e}")
                disconnected.append(connection)

        # 끊어진 연결 정리
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_to_subscribers(self, message: str, topic: str):
        """토픽 구독자에게 브로드캐스트"""
        if not self.active_connections:
            return

        target_connections = [
            conn for conn in self.active_connections
            if topic in self.connection_info.get(conn, {}).get('subscriptions', [])
        ]

        for connection in target_connections:
            await self.send_personal_message(message, connection)

    def get_connection_stats(self) -> Dict:
        """연결 통계"""
        return {
            'total_connections': len(self.active_connections),
            'connections_by_type': self._count_by_type(),
            'active_subscriptions': self._count_subscriptions()
        }

    def _count_by_type(self) -> Dict[str, int]:
        """타입별 연결 수"""
        counts = defaultdict(int)
        for conn_info in self.connection_info.values():
            client_type = conn_info.get('type', 'unknown')
            counts[client_type] += 1
        return dict(counts)

    def _count_subscriptions(self) -> Dict[str, int]:
        """구독별 연결 수"""
        counts = defaultdict(int)
        for conn_info in self.connection_info.values():
            subscriptions = conn_info.get('subscriptions', [])
            for sub in subscriptions:
                counts[sub] += 1
        return dict(counts)


class RealTimeDataStreamer:
    """실시간 데이터 스트리머"""

    def __init__(self, connection_manager: ConnectionManager):
        self.logger = get_logger("dashboard.streamer")
        self.connection_manager = connection_manager
        self.config = get_config()

        # 데이터 버퍼
        self.metric_buffer = deque(maxlen=1000)
        self.alert_buffer = deque(maxlen=100)

        # 업데이트 간격 (초)
        self.update_intervals = {
            MetricType.POWER_CONSUMPTION: 5,
            MetricType.ENERGY_EFFICIENCY: 10,
            MetricType.ANOMALY_SCORE: 15,
            MetricType.COST_RATE: 30,
            MetricType.MACHINE_STATUS: 5,
            MetricType.JOB_PROGRESS: 10
        }

        # 스트리밍 태스크
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.is_streaming = False

        # 외부 데이터 소스
        self.model_manager = get_model_manager()
        self.tou_model = create_tou_model("standard")

    async def start_streaming(self):
        """스트리밍 시작"""
        if self.is_streaming:
            return

        self.is_streaming = True
        self.logger.info("실시간 데이터 스트리밍 시작")

        # 각 메트릭별 스트리밍 태스크 시작
        for metric_type in MetricType:
            task_name = f"stream_{metric_type.value}"
            self.streaming_tasks[task_name] = asyncio.create_task(
                self._stream_metric(metric_type)
            )

        # 알림 스트리밍 태스크
        self.streaming_tasks["stream_alerts"] = asyncio.create_task(
            self._stream_alerts()
        )

    async def stop_streaming(self):
        """스트리밍 중지"""
        if not self.is_streaming:
            return

        self.is_streaming = False
        self.logger.info("실시간 데이터 스트리밍 중지")

        # 모든 태스크 취소
        for task_name, task in self.streaming_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.streaming_tasks.clear()

    async def _stream_metric(self, metric_type: MetricType):
        """메트릭별 스트리밍"""
        interval = self.update_intervals[metric_type]

        while self.is_streaming:
            try:
                # 메트릭 데이터 생성
                metrics = await self._generate_metric_data(metric_type)

                # 버퍼에 추가
                for metric in metrics:
                    self.metric_buffer.append(metric)

                # 클라이언트에 전송
                message = {
                    "type": "metric_update",
                    "metric_type": metric_type.value,
                    "data": [metric.to_dict() for metric in metrics],
                    "timestamp": datetime.now().isoformat()
                }

                await self.connection_manager.broadcast_to_subscribers(
                    json.dumps(message), f"metrics.{metric_type.value}"
                )

                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"메트릭 스트리밍 오류 [{metric_type.value}]: {e}")
                await asyncio.sleep(interval)

    async def _stream_alerts(self):
        """알림 스트리밍"""
        while self.is_streaming:
            try:
                # 새로운 알림 확인
                new_alerts = await self._check_for_alerts()

                for alert in new_alerts:
                    self.alert_buffer.append(alert)

                    # 클라이언트에 전송
                    message = {
                        "type": "alert",
                        "data": alert.to_dict(),
                        "timestamp": datetime.now().isoformat()
                    }

                    await self.connection_manager.broadcast_to_subscribers(
                        json.dumps(message), "alerts"
                    )

                await asyncio.sleep(10)  # 10초마다 알림 확인

            except Exception as e:
                self.logger.error(f"알림 스트리밍 오류: {e}")
                await asyncio.sleep(10)

    async def _generate_metric_data(self, metric_type: MetricType) -> List[RealTimeMetric]:
        """메트릭 데이터 생성"""
        current_time = datetime.now()
        metrics = []

        if metric_type == MetricType.POWER_CONSUMPTION:
            # 기계별 전력 소비량
            for i in range(8):  # 8대 기계
                machine_id = f"machine_{i + 1}"
                power = np.random.normal(100 + i * 10, 15)  # 기계별 기본 전력 + 변동

                metric = RealTimeMetric(
                    metric_type=metric_type,
                    value=max(0, power),
                    unit="kW",
                    timestamp=current_time,
                    machine_id=machine_id
                )
                metrics.append(metric)

        elif metric_type == MetricType.ENERGY_EFFICIENCY:
            # 전체 에너지 효율성
            efficiency = np.random.uniform(0.75, 0.95)

            metric = RealTimeMetric(
                metric_type=metric_type,
                value=efficiency,
                unit="%",
                timestamp=current_time,
                metadata={"target": 0.85}
            )
            metrics.append(metric)

        elif metric_type == MetricType.ANOMALY_SCORE:
            # 이상 점수 (기계별)
            for i in range(8):
                machine_id = f"machine_{i + 1}"
                # 5% 확률로 높은 이상 점수
                if np.random.random() < 0.05:
                    score = np.random.uniform(0.7, 1.0)
                else:
                    score = np.random.exponential(0.1)

                metric = RealTimeMetric(
                    metric_type=metric_type,
                    value=min(1.0, score),
                    unit="score",
                    timestamp=current_time,
                    machine_id=machine_id,
                    metadata={"threshold": 0.7}
                )
                metrics.append(metric)

        elif metric_type == MetricType.COST_RATE:
            # 현재 전력 요금
            current_rate = self.tou_model.get_rate(current_time)

            metric = RealTimeMetric(
                metric_type=metric_type,
                value=current_rate,
                unit="원/kWh",
                timestamp=current_time,
                metadata={
                    "period": self.tou_model.get_period(current_time).value
                }
            )
            metrics.append(metric)

        elif metric_type == MetricType.MACHINE_STATUS:
            # 기계 상태 (가동률)
            for i in range(8):
                machine_id = f"machine_{i + 1}"
                utilization = np.random.uniform(0.3, 0.95)

                metric = RealTimeMetric(
                    metric_type=metric_type,
                    value=utilization,
                    unit="%",
                    timestamp=current_time,
                    machine_id=machine_id
                )
                metrics.append(metric)

        elif metric_type == MetricType.JOB_PROGRESS:
            # 작업 진행률
            progress = np.random.uniform(0.6, 0.95)

            metric = RealTimeMetric(
                metric_type=metric_type,
                value=progress,
                unit="%",
                timestamp=current_time,
                metadata={
                    "total_jobs": 50,
                    "completed_jobs": int(50 * progress)
                }
            )
            metrics.append(metric)

        return metrics

    async def _check_for_alerts(self) -> List[Alert]:
        """새로운 알림 확인"""
        alerts = []
        current_time = datetime.now()

        # 최근 메트릭 데이터에서 알림 조건 확인
        recent_metrics = [m for m in self.metric_buffer
                          if (current_time - m.timestamp).seconds < 60]

        # 전력 소비 이상 확인
        power_metrics = [m for m in recent_metrics
                         if m.metric_type == MetricType.POWER_CONSUMPTION]

        for metric in power_metrics:
            if metric.value > 200:  # 200kW 초과시 경고
                alert = Alert(
                    alert_id=f"power_alert_{metric.machine_id}_{int(time.time())}",
                    level=AlertLevel.WARNING,
                    title="높은 전력 소비",
                    message=f"{metric.machine_id}에서 비정상적으로 높은 전력 소비 감지: {metric.value:.1f}kW",
                    source=metric.machine_id,
                    timestamp=current_time,
                    metadata={
                        "power_value": metric.value,
                        "threshold": 200,
                        "machine_id": metric.machine_id
                    }
                )
                alerts.append(alert)

        # 이상 점수 확인
        anomaly_metrics = [m for m in recent_metrics
                           if m.metric_type == MetricType.ANOMALY_SCORE]

        for metric in anomaly_metrics:
            if metric.value > 0.7:  # 이상 점수 0.7 초과시 경고
                alert = Alert(
                    alert_id=f"anomaly_alert_{metric.machine_id}_{int(time.time())}",
                    level=AlertLevel.ERROR if metric.value > 0.9 else AlertLevel.WARNING,
                    title="이상 행동 감지",
                    message=f"{metric.machine_id}에서 이상 행동 패턴 감지 (점수: {metric.value:.3f})",
                    source=metric.machine_id,
                    timestamp=current_time,
                    metadata={
                        "anomaly_score": metric.value,
                        "threshold": 0.7,
                        "machine_id": metric.machine_id
                    }
                )
                alerts.append(alert)

        # 효율성 저하 확인
        efficiency_metrics = [m for m in recent_metrics
                              if m.metric_type == MetricType.ENERGY_EFFICIENCY]

        for metric in efficiency_metrics:
            if metric.value < 0.7:  # 효율성 70% 미만시 경고
                alert = Alert(
                    alert_id=f"efficiency_alert_{int(time.time())}",
                    level=AlertLevel.WARNING,
                    title="에너지 효율성 저하",
                    message=f"전체 에너지 효율성이 목표치를 하회: {metric.value:.1%}",
                    source="system",
                    timestamp=current_time,
                    metadata={
                        "efficiency": metric.value,
                        "target": 0.85
                    }
                )
                alerts.append(alert)

        return alerts

    def get_latest_metrics(self, metric_type: MetricType = None,
                           limit: int = 100) -> List[RealTimeMetric]:
        """최신 메트릭 데이터 조회"""
        if metric_type:
            filtered_metrics = [m for m in self.metric_buffer
                                if m.metric_type == metric_type]
        else:
            filtered_metrics = list(self.metric_buffer)

        return list(filtered_metrics)[-limit:]

    def get_latest_alerts(self, limit: int = 50) -> List[Alert]:
        """최신 알림 조회"""
        return list(self.alert_buffer)[-limit:]


class DashboardManager:
    """대시보드 관리자"""

    def __init__(self):
        self.logger = get_logger("dashboard.manager")
        self.config = get_config()

        # 컴포넌트들
        self.connection_manager = ConnectionManager()
        self.data_streamer = RealTimeDataStreamer(self.connection_manager)

        # 대시보드 상태
        self.is_active = False

        # 통계
        self.stats = {
            "start_time": None,
            "total_messages_sent": 0,
            "total_alerts_generated": 0,
            "uptime_hours": 0
        }

    async def start(self):
        """대시보드 시작"""
        if self.is_active:
            return

        self.is_active = True
        self.stats["start_time"] = datetime.now()

        await self.data_streamer.start_streaming()

        self.logger.info("실시간 대시보드 시작")

    async def stop(self):
        """대시보드 중지"""
        if not self.is_active:
            return

        self.is_active = False

        await self.data_streamer.stop_streaming()

        # 모든 연결 종료
        for connection in self.connection_manager.active_connections.copy():
            await connection.close()

        self.logger.info("실시간 대시보드 중지")

    async def handle_websocket(self, websocket: WebSocket, client_info: Dict = None):
        """WebSocket 연결 처리"""
        await self.connection_manager.connect(websocket, client_info)

        try:
            while True:
                # 클라이언트 메시지 수신
                data = await websocket.receive_text()
                message = json.loads(data)

                await self._handle_client_message(websocket, message)

        except WebSocketDisconnect:
            self.connection_manager.disconnect(websocket)
        except Exception as e:
            self.logger.error(f"WebSocket 처리 오류: {e}")
            self.connection_manager.disconnect(websocket)

    async def _handle_client_message(self, websocket: WebSocket, message: Dict):
        """클라이언트 메시지 처리"""
        msg_type = message.get("type")

        if msg_type == "subscribe":
            # 토픽 구독
            topics = message.get("topics", [])
            client_info = self.connection_manager.connection_info.get(websocket, {})
            client_info["subscriptions"] = topics

            response = {
                "type": "subscription_confirmed",
                "topics": topics,
                "timestamp": datetime.now().isoformat()
            }
            await self.connection_manager.send_personal_message(
                json.dumps(response), websocket
            )

        elif msg_type == "get_latest_data":
            # 최신 데이터 요청
            metric_type_str = message.get("metric_type")
            limit = message.get("limit", 100)

            if metric_type_str:
                metric_type = MetricType(metric_type_str)
                metrics = self.data_streamer.get_latest_metrics(metric_type, limit)
            else:
                metrics = self.data_streamer.get_latest_metrics(limit=limit)

            response = {
                "type": "latest_data",
                "data": [metric.to_dict() for metric in metrics],
                "timestamp": datetime.now().isoformat()
            }
            await self.connection_manager.send_personal_message(
                json.dumps(response), websocket
            )

        elif msg_type == "get_alerts":
            # 알림 목록 요청
            limit = message.get("limit", 50)
            alerts = self.data_streamer.get_latest_alerts(limit)

            response = {
                "type": "alerts",
                "data": [alert.to_dict() for alert in alerts],
                "timestamp": datetime.now().isoformat()
            }
            await self.connection_manager.send_personal_message(
                json.dumps(response), websocket
            )

        elif msg_type == "acknowledge_alert":
            # 알림 확인
            alert_id = message.get("alert_id")
            # 실제 구현에서는 데이터베이스 업데이트
            self.logger.info(f"알림 확인: {alert_id}")

    def get_dashboard_summary(self) -> Dict:
        """대시보드 요약 정보"""
        current_time = datetime.now()

        if self.stats["start_time"]:
            uptime = (current_time - self.stats["start_time"]).total_seconds() / 3600
            self.stats["uptime_hours"] = uptime

        # 최신 메트릭 요약
        latest_metrics = self.data_streamer.get_latest_metrics(limit=50)
        power_metrics = [m for m in latest_metrics
                         if m.metric_type == MetricType.POWER_CONSUMPTION]

        total_power = sum(m.value for m in power_metrics) if power_metrics else 0

        return {
            "status": "active" if self.is_active else "inactive",
            "connections": self.connection_manager.get_connection_stats(),
            "stats": self.stats,
            "current_summary": {
                "total_power_consumption": total_power,
                "active_machines": len(set(m.machine_id for m in power_metrics if m.machine_id)),
                "recent_alerts": len(self.data_streamer.get_latest_alerts(10)),
                "timestamp": current_time.isoformat()
            }
        }


# 전역 대시보드 관리자 인스턴스
dashboard_manager = DashboardManager()


# 편의 함수들
async def start_dashboard():
    """대시보드 시작"""
    await dashboard_manager.start()


async def stop_dashboard():
    """대시보드 중지"""
    await dashboard_manager.stop()


def get_dashboard_manager() -> DashboardManager:
    """대시보드 관리자 반환"""
    return dashboard_manager


# 사용 예시
if __name__ == "__main__":
    async def test_dashboard():
        # 대시보드 시작
        await start_dashboard()

        # 잠시 실행
        await asyncio.sleep(30)

        # 통계 출력
        summary = dashboard_manager.get_dashboard_summary()
        print("대시보드 요약:", json.dumps(summary, indent=2, ensure_ascii=False))

        # 대시보드 중지
        await stop_dashboard()


    asyncio.run(test_dashboard())