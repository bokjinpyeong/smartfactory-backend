"""
스마트팩토리 에너지 관리 시스템 - API 엔드포인트

RESTful API 엔드포인트 제공
- 데이터 수집 및 조회
- 모델 학습 및 예측
- 스케줄링 최적화
- 실시간 모니터링
- 시각화 데이터
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json
import asyncio
import pandas as pd
import numpy as np

# Core 모듈
from core.config import get_config
from core.logger import get_logger, log_performance
from core.exceptions import (
    SmartFactoryException, ModelException, SchedulingError,
    DataValidationError, safe_execute
)

# 기능 모듈들
from data.collector import create_collection_manager, create_mqtt_collector
from data.processor import create_default_pipeline
from data.validator import create_validator, ValidationLevel
from models import create_model, get_model_manager
from optimization.scheduler import create_scheduler, SchedulingObjective, Job, JobStatus
from optimization.tou_pricing import create_tou_model
from optimization.constraints import create_constraint_manager


# FastAPI 앱 생성
app = FastAPI(
    title="스마트팩토리 에너지 관리 시스템 API",
    description="IoT 기반 전력 최적화 및 이상탐지 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발환경용, 운영시 수정 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
logger = get_logger("api")
config = get_config()
security = HTTPBearer()

# 모델 관리자
model_manager = get_model_manager()

# 데이터 수집 관리자
collection_manager = create_collection_manager()

# 데이터 처리 파이프라인
processing_pipeline = create_default_pipeline()

# 데이터 검증기
data_validator = create_validator(ValidationLevel.STANDARD)

# 스케줄러
scheduler = create_scheduler("greedy")

# TOU 모델
tou_model = create_tou_model("standard")

# 제약 관리자
constraint_manager = create_constraint_manager()


# === Pydantic 모델들 ===

class SensorDataInput(BaseModel):
    """센서 데이터 입력 모델"""
    sensor_id: str
    machine_id: str
    timestamp: datetime
    power_consumption: float = Field(ge=0, le=10000)
    voltage: float = Field(ge=0, le=500)
    current: float = Field(ge=0, le=100)
    temperature: Optional[float] = Field(None, ge=-50, le=100)
    humidity: Optional[float] = Field(None, ge=0, le=100)
    status: str = "normal"


class JobInput(BaseModel):
    """작업 입력 모델"""
    job_id: str
    machine_id: str
    power_requirement: float = Field(ge=0, le=1000)
    processing_time: int = Field(ge=1, le=1440)  # 1분 ~ 24시간
    due_date: datetime
    priority: int = Field(default=1, ge=1, le=5)
    earliest_start: Optional[datetime] = None
    setup_time: int = Field(default=0, ge=0, le=120)

    @validator('due_date')
    def due_date_must_be_future(cls, v):
        if v <= datetime.now():
            raise ValueError('납기일은 미래 시간이어야 합니다')
        return v


class SchedulingRequest(BaseModel):
    """스케줄링 요청 모델"""
    jobs: List[JobInput]
    objective: str = "minimize_cost"
    scheduler_type: str = "greedy"
    time_horizon_hours: int = Field(default=24, ge=1, le=168)  # 1시간 ~ 1주일

    @validator('objective')
    def objective_must_be_valid(cls, v):
        valid_objectives = [obj.value for obj in SchedulingObjective]
        if v not in valid_objectives:
            raise ValueError(f'목적 함수는 다음 중 하나여야 합니다: {valid_objectives}')
        return v


class PredictionRequest(BaseModel):
    """예측 요청 모델"""
    model_type: str = "power_prediction"
    input_data: Dict[str, Any]
    prediction_horizon: int = Field(default=24, ge=1, le=168)


class ModelTrainingRequest(BaseModel):
    """모델 학습 요청 모델"""
    model_type: str
    training_data_path: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)


# === 헬스체크 ===

@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "api": "online",
            "models": len(model_manager.models),
            "data_collection": "active" if collection_manager else "inactive"
        }
    }


# === 데이터 관련 엔드포인트 ===

@app.post("/api/v1/data/sensors")
@log_performance
async def collect_sensor_data(data: SensorDataInput):
    """센서 데이터 수집"""
    try:
        # 데이터 검증
        df_data = pd.DataFrame([data.dict()])
        validation_result = data_validator.validate(df_data)

        if not validation_result.is_valid():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "데이터 검증 실패",
                    "validation_issues": [issue.to_dict() for issue in validation_result.issues]
                }
            )

        # 데이터 처리
        processing_result = processing_pipeline.transform(df_data)

        if processing_result.error_log:
            logger.warning(f"데이터 처리 경고: {processing_result.error_log}")

        return {
            "status": "success",
            "message": "센서 데이터 수집 완료",
            "processed_data_shape": processing_result.data.shape,
            "processing_time": processing_result.processing_time
        }

    except Exception as e:
        logger.error(f"센서 데이터 수집 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/sensors")
async def get_sensor_data(
    machine_id: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000)
):
    """센서 데이터 조회"""
    try:
        # 실제 구현에서는 데이터베이스에서 조회
        # 여기서는 샘플 데이터 반환
        sample_data = []
        for i in range(min(limit, 10)):
            sample_data.append({
                "sensor_id": f"sensor_{i+1}",
                "machine_id": machine_id or f"machine_{i%3+1}",
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                "power_consumption": 50 + i * 10,
                "voltage": 220 + i,
                "current": 0.5 + i * 0.1
            })

        return {
            "status": "success",
            "data": sample_data,
            "total_count": len(sample_data),
            "filters": {
                "machine_id": machine_id,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None
            }
        }

    except Exception as e:
        logger.error(f"센서 데이터 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/data/validate")
async def validate_data(data: Dict[str, Any]):
    """데이터 검증"""
    try:
        # JSON 데이터를 DataFrame으로 변환
        if isinstance(data.get('data'), list):
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame([data])

        # 검증 수행
        validation_result = data_validator.validate(df)

        return {
            "status": "success",
            "validation_result": {
                "is_valid": validation_result.is_valid(),
                "overall_score": validation_result.overall_score,
                "total_issues": validation_result.total_issues,
                "issues_by_level": validation_result.issues_by_level,
                "processing_time": validation_result.processing_time
            },
            "issues": [issue.to_dict() for issue in validation_result.issues[:10]]  # 최대 10개
        }

    except Exception as e:
        logger.error(f"데이터 검증 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === 모델 관련 엔드포인트 ===

@app.get("/api/v1/models")
async def list_models():
    """등록된 모델 목록"""
    try:
        models_info = []
        for model_id, model in model_manager.models.items():
            models_info.append({
                "model_id": model_id,
                "model_type": getattr(model, 'model_type', 'unknown'),
                "is_trained": getattr(model, 'is_trained', False),
                "model_name": getattr(model, 'model_name', model_id)
            })

        return {
            "status": "success",
            "models": models_info,
            "total_count": len(models_info)
        }

    except Exception as e:
        logger.error(f"모델 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/models/train")
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """모델 학습"""
    try:
        # 백그라운드에서 학습 실행
        background_tasks.add_task(
            _train_model_background,
            request.model_type,
            request.training_data_path,
            request.hyperparameters,
            request.validation_split
        )

        return {
            "status": "success",
            "message": f"{request.model_type} 모델 학습이 백그라운드에서 시작되었습니다",
            "model_type": request.model_type
        }

    except Exception as e:
        logger.error(f"모델 학습 요청 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_model_background(model_type: str, data_path: str, hyperparams: Dict, val_split: float):
    """백그라운드 모델 학습"""
    try:
        logger.info(f"모델 학습 시작: {model_type}")

        # 모델 생성 또는 조회
        model = model_manager.get_model(model_type)
        if not model:
            model = create_model(model_type, f"{model_type}_api")
            model_manager.register_model(f"{model_type}_api", model)

        # 샘플 데이터 생성 (실제로는 data_path에서 로드)
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'target': np.random.randn(1000)
        })

        # 모델 학습
        if hasattr(model, 'train'):
            model.train(sample_data)

        logger.info(f"모델 학습 완료: {model_type}")

    except Exception as e:
        logger.error(f"백그라운드 모델 학습 오류: {e}")


@app.post("/api/v1/models/predict")
async def predict(request: PredictionRequest):
    """모델 예측"""
    try:
        model = model_manager.get_model(request.model_type)
        if not model:
            raise HTTPException(status_code=404, detail=f"모델 '{request.model_type}'을 찾을 수 없습니다")

        # 입력 데이터를 DataFrame으로 변환
        input_df = pd.DataFrame([request.input_data])

        # 예측 수행
        if hasattr(model, 'predict'):
            predictions = model.predict(input_df)

            # 결과 형태에 따른 처리
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()
            elif isinstance(predictions, pd.DataFrame):
                predictions = predictions.to_dict('records')
        else:
            raise HTTPException(status_code=400, detail="모델이 예측 기능을 지원하지 않습니다")

        return {
            "status": "success",
            "model_type": request.model_type,
            "predictions": predictions,
            "prediction_horizon": request.prediction_horizon,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"모델 예측 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/{model_id}/status")
async def get_model_status(model_id: str = Path(...)):
    """모델 상태 조회"""
    try:
        model = model_manager.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"모델 '{model_id}'을 찾을 수 없습니다")

        status_info = {
            "model_id": model_id,
            "is_trained": getattr(model, 'is_trained', False),
            "model_type": getattr(model, 'model_type', 'unknown'),
            "model_name": getattr(model, 'model_name', model_id),
            "last_updated": datetime.now().isoformat()
        }

        # 추가 정보 (모델별로 다름)
        if hasattr(model, 'get_model_info'):
            status_info.update(model.get_model_info())

        return {
            "status": "success",
            "model_status": status_info
        }

    except Exception as e:
        logger.error(f"모델 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === 스케줄링 관련 엔드포인트 ===

@app.post("/api/v1/scheduling/optimize")
@log_performance
async def optimize_schedule(request: SchedulingRequest):
    """스케줄링 최적화"""
    try:
        # JobInput을 Job 객체로 변환
        jobs = []
        for job_input in request.jobs:
            job = Job(
                job_id=job_input.job_id,
                machine_id=job_input.machine_id,
                power_requirement=job_input.power_requirement,
                processing_time=job_input.processing_time,
                due_date=job_input.due_date,
                priority=job_input.priority,
                earliest_start=job_input.earliest_start,
                setup_time=job_input.setup_time
            )
            jobs.append(job)

        # 목적 함수 변환
        objective = SchedulingObjective(request.objective)

        # 스케줄러 생성 및 최적화
        scheduler = create_scheduler(request.scheduler_type)
        result = scheduler.schedule(jobs, objective)

        # 결과 반환
        return {
            "status": "success",
            "scheduling_result": {
                "is_feasible": result.is_feasible,
                "total_cost": result.total_cost,
                "peak_power": result.peak_power,
                "makespan_hours": result.makespan,
                "total_delay_hours": result.total_delay,
                "utilization_rate": result.utilization_rate,
                "objective_value": result.objective_value,
                "computation_time": result.computation_time,
                "constraints_violated": result.constraints_violated
            },
            "scheduled_jobs": [job.to_dict() for job in result.scheduled_jobs],
            "gantt_data": result.to_gantt_data(),
            "optimization_details": result.optimization_details
        }

    except Exception as e:
        logger.error(f"스케줄링 최적화 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scheduling/jobs")
async def get_jobs(
    status: Optional[str] = Query(None),
    machine_id: Optional[str] = Query(None),
    limit: int = Query(100, le=1000)
):
    """작업 목록 조회"""
    try:
        # 실제 구현에서는 데이터베이스에서 조회
        # 여기서는 샘플 데이터 반환
        sample_jobs = []
        statuses = ['pending', 'scheduled', 'in_progress', 'completed']

        for i in range(min(limit, 20)):
            job_status = status if status else statuses[i % len(statuses)]
            sample_jobs.append({
                "job_id": f"job_{i+1:03d}",
                "machine_id": machine_id or f"machine_{i%5+1}",
                "power_requirement": 50 + (i % 10) * 10,
                "processing_time": 60 + (i % 8) * 30,
                "due_date": (datetime.now() + timedelta(hours=i+1)).isoformat(),
                "priority": (i % 5) + 1,
                "status": job_status
            })

        return {
            "status": "success",
            "jobs": sample_jobs,
            "total_count": len(sample_jobs),
            "filters": {
                "status": status,
                "machine_id": machine_id
            }
        }

    except Exception as e:
        logger.error(f"작업 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === TOU 요금제 관련 엔드포인트 ===

@app.get("/api/v1/tou/rates")
async def get_current_rates():
    """현재 전력 요금 조회"""
    try:
        now = datetime.now()
        current_rate = tou_model.get_rate(now)
        current_period = tou_model.get_period(now)

        # 일일 프로파일
        daily_profile = tou_model.get_daily_profile(now)

        return {
            "status": "success",
            "current_rate": current_rate,
            "current_period": current_period.value,
            "timestamp": now.isoformat(),
            "daily_profile": daily_profile.to_dict('records')
        }

    except Exception as e:
        logger.error(f"TOU 요금 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/tou/calculate-cost")
async def calculate_cost(
    start_time: datetime,
    duration_hours: float = Query(ge=0.1, le=168),
    power_kw: float = Query(ge=0.1, le=1000)
):
    """전력 사용 비용 계산"""
    try:
        cost = tou_model.calculate_cost(start_time, duration_hours, power_kw)

        return {
            "status": "success",
            "calculation": {
                "start_time": start_time.isoformat(),
                "duration_hours": duration_hours,
                "power_kw": power_kw,
                "total_cost": cost,
                "average_rate": cost / (duration_hours * power_kw) if duration_hours * power_kw > 0 else 0
            }
        }

    except Exception as e:
        logger.error(f"비용 계산 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tou/optimization-recommendations")
async def get_optimization_recommendations():
    """TOU 기반 최적화 권장사항"""
    try:
        # 샘플 작업들
        sample_jobs = [
            {
                "job_id": "job_001",
                "start_time": (datetime.now() + timedelta(hours=2)).isoformat(),
                "duration_hours": 3.0,
                "power_kw": 50.0
            },
            {
                "job_id": "job_002",
                "start_time": (datetime.now() + timedelta(hours=8)).isoformat(),
                "duration_hours": 2.0,
                "power_kw": 75.0
            }
        ]

        recommendations = tou_model.get_optimization_recommendations(sample_jobs)

        return {
            "status": "success",
            "recommendations": recommendations,
            "total_potential_savings": sum(rec['savings'] for rec in recommendations)
        }

    except Exception as e:
        logger.error(f"최적화 권장사항 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === 모니터링 관련 엔드포인트 ===

@app.get("/api/v1/monitoring/realtime")
async def get_realtime_data():
    """실시간 모니터링 데이터"""
    try:
        # 실시간 데이터 시뮬레이션
        current_time = datetime.now()

        realtime_data = {
            "timestamp": current_time.isoformat(),
            "power_consumption": {
                "total": np.random.normal(800, 50),
                "by_machine": {
                    f"machine_{i+1}": np.random.normal(100, 20)
                    for i in range(8)
                }
            },
            "energy_efficiency": np.random.uniform(0.75, 0.95),
            "anomaly_score": np.random.exponential(0.1),
            "active_jobs": np.random.randint(5, 15),
            "peak_power_usage": np.random.uniform(0.6, 0.9),
            "cost_today": np.random.uniform(50000, 100000)
        }

        return {
            "status": "success",
            "realtime_data": realtime_data
        }

    except Exception as e:
        logger.error(f"실시간 데이터 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/monitoring/dashboard")
async def get_dashboard_data():
    """대시보드 데이터"""
    try:
        dashboard_data = {
            "summary": {
                "total_machines": 8,
                "active_machines": 6,
                "total_jobs_today": 45,
                "completed_jobs": 32,
                "energy_cost_today": 78500,
                "efficiency_score": 0.87
            },
            "alerts": [
                {
                    "id": "alert_001",
                    "type": "warning",
                    "message": "Machine_3에서 비정상적인 전력 소비 패턴 감지",
                    "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()
                },
                {
                    "id": "alert_002",
                    "type": "info",
                    "message": "오후 피크 시간대 진입 - 전력 사용 최적화 권장",
                    "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()
                }
            ],
            "recent_anomalies": [
                {
                    "machine_id": "machine_3",
                    "anomaly_score": 0.95,
                    "timestamp": (datetime.now() - timedelta(minutes=20)).isoformat(),
                    "description": "전력 소비량 급증"
                }
            ]
        }

        return {
            "status": "success",
            "dashboard_data": dashboard_data
        }

    except Exception as e:
        logger.error(f"대시보드 데이터 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === 시스템 관리 엔드포인트 ===

@app.get("/api/v1/system/statistics")
async def get_system_statistics():
    """시스템 통계"""
    try:
        stats = {
            "data_collection": collection_manager.get_statistics() if collection_manager else {},
            "data_processing": processing_pipeline.pipeline_stats if processing_pipeline else {},
            "data_validation": data_validator.get_validation_statistics(),
            "constraints": constraint_manager.get_statistics(),
            "tou_pricing": tou_model.get_statistics(),
            "models": {
                "total_models": len(model_manager.models),
                "trained_models": sum(1 for model in model_manager.models.values()
                                    if getattr(model, 'is_trained', False))
            }
        }

        return {
            "status": "success",
            "system_statistics": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"시스템 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/system/reset")
async def reset_system(component: str = Query(None)):
    """시스템 재설정"""
    try:
        reset_results = []

        if component is None or component == "all":
            # 전체 시스템 재설정
            components = ["models", "data_collection", "constraints"]
        else:
            components = [component]

        for comp in components:
            if comp == "models":
                # 모델 초기화
                model_manager.models.clear()
                reset_results.append("모델 관리자 초기화")

            elif comp == "data_collection":
                # 데이터 수집 재설정
                if collection_manager:
                    await collection_manager.stop_all_collectors()
                reset_results.append("데이터 수집 중지")

            elif comp == "constraints":
                # 제약 조건 재설정
                constraint_manager._setup_default_constraints()
                reset_results.append("제약 조건 기본값으로 재설정")

        return {
            "status": "success",
            "message": "시스템 재설정 완료",
            "reset_components": reset_results
        }

    except Exception as e:
        logger.error(f"시스템 재설정 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === 예외 처리 ===

@app.exception_handler(SmartFactoryException)
async def smartfactory_exception_handler(request, exc):
    """스마트팩토리 예외 처리"""
    return JSONResponse(
        status_code=400,
        content={
            "error": "SmartFactory Error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 처리"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 처리"""
    logger.error(f"예상치 못한 오류: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "내부 서버 오류가 발생했습니다",
            "type": type(exc).__name__
        }
    )


# === 시작 이벤트 ===

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작시 실행"""
    logger.info("스마트팩토리 에너지 관리 시스템 API 시작")

    # 기본 모델들 로드
    try:
        # 이상탐지 모델
        anomaly_model = create_model("anomaly_detector", "api_anomaly_detector")
        model_manager.register_model("anomaly_detector", anomaly_model)

        # 전력 예측 모델
        power_model = create_model("power_predictor", "api_power_predictor")
        model_manager.register_model("power_predictor", power_model)

        logger.info(f"기본 모델 로드 완료: {len(model_manager.models)}개")

    except Exception as e:
        logger.error(f"모델 로드 오류: {e}")

    # 데이터 처리 파이프라인 초기화
    try:
        # 샘플 데이터로 파이프라인 사전 학습
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'machine_id': ['machine_1'] * 100,
            'power_consumption': np.random.normal(100, 20, 100),
            'voltage': np.random.normal(220, 10, 100),
            'current': np.random.normal(0.5, 0.1, 100)
        })

        processing_pipeline.fit(sample_data)
        logger.info("데이터 처리 파이프라인 초기화 완료")

    except Exception as e:
        logger.error(f"파이프라인 초기화 오류: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료시 실행"""
    logger.info("스마트팩토리 에너지 관리 시스템 API 종료")

    # 데이터 수집 중지
    if collection_manager:
        try:
            await collection_manager.stop_all_collectors()
            logger.info("데이터 수집 중지 완료")
        except Exception as e:
            logger.error(f"데이터 수집 중지 오류: {e}")


# === 메인 실행 ===

if __name__ == "__main__":
    import uvicorn

    # 설정에서 호스트와 포트 가져오기
    host = config.system.api_host
    port = config.system.api_port
    workers = config.system.api_workers

    logger.info(f"API 서버 시작: http://{host}:{port}")

    uvicorn.run(
        "api.routes:app",
        host=host,
        port=port,
        workers=1,  # 개발환경에서는 1개 워커 사용
        reload=True,  # 개발환경에서만 사용
        log_level="info"
    )