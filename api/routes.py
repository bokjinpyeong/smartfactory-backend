"""
스마트팩토리 에너지 관리 시스템 API 라우트
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from ..core import get_logger, get_config
from ..models import create_complete_integrated_system, get_enhanced_system_status
from ..data import quick_pipeline
from ..optimization import IntegratedEnergyManagementSystem

app = FastAPI(
    title="스마트팩토리 에너지 관리 시스템",
    description="IoT 기반 에너지 최적화 및 이상탐지 시스템",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger("api")
config = get_config()

# 전역 시스템 인스턴스
ai_system = None
energy_system = None


@app.on_event("startup")
async def startup_event():
    """API 시작 시 초기화"""
    global ai_system, energy_system
    logger.info("🚀 API 서버 시작 - 시스템 초기화 중...")

    try:
        # AI 시스템 초기화
        ai_system = create_complete_integrated_system()

        # 에너지 최적화 시스템 초기화
        energy_system = IntegratedEnergyManagementSystem()

        logger.info("✅ 시스템 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 실패: {str(e)}")


# =============================================================================
# 헬스체크 및 시스템 정보 API

@app.get("/health")
async def health_check():
    """시스템 헬스체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.get("/system/status")
async def get_system_status():
    """시스템 상태 조회"""
    try:
        ai_status = get_enhanced_system_status()
        return {
            "ai_system": ai_status,
            "energy_system": {
                "initialized": energy_system is not None,
                "peak_power_limit": energy_system.scheduler.peak_power_limit if energy_system else None
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 데이터 처리 API

@app.post("/data/upload")
async def upload_data(data: Dict):
    """데이터 업로드 및 전처리"""
    try:
        # DataFrame 변환
        df = pd.DataFrame(data['data'])

        # 데이터 전처리 파이프라인 실행
        result = quick_pipeline(df, data.get('target_col', 'Power_Consumption_Realistic'))

        return {
            "message": "데이터 처리 완료",
            "shape": {
                "train": result[0].shape,
                "val": result[1].shape,
                "test": result[2].shape
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/data/generate")
async def generate_sample_data(n_samples: int = 10000, anomaly_rate: float = 0.05):
    """샘플 데이터 생성"""
    try:
        # 현실적인 산업 데이터 생성
        data = energy_system.create_enhanced_industrial_data(n_samples, anomaly_rate)

        return {
            "message": "데이터 생성 완료",
            "samples": len(data),
            "columns": list(data.columns),
            "summary": {
                "power_mean": data['Power_Consumption_Real'].mean(),
                "power_std": data['Power_Consumption_Real'].std(),
                "energy_cost_total": data['Energy_Cost_Real'].sum()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AI 모델 API

@app.post("/ai/train")
async def train_ai_models(background_tasks: BackgroundTasks, data: Dict):
    """AI 모델 학습"""
    try:
        # 데이터 준비
        df = pd.DataFrame(data['data'])
        X = df.drop(columns=[data['target_col']])
        y = df[data['target_col']]

        # 백그라운드에서 학습 실행
        background_tasks.add_task(run_training, X, y)

        return {"message": "AI 모델 학습이 백그라운드에서 시작되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def run_training(X, y):
    """백그라운드 학습 실행"""
    try:
        if ai_system:
            # 데이터 분할
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

            # 통합 시스템 학습
            result = ai_system['integrated_system'].train(
                X_train, y_train, X_val, y_val
            )
            logger.info(f"✅ AI 모델 학습 완료: {result['total_training_time']:.1f}초")
    except Exception as e:
        logger.error(f"❌ AI 모델 학습 실패: {str(e)}")


@app.post("/ai/predict")
async def predict_anomaly_and_power(data: Dict):
    """이상탐지 + 전력예측"""
    try:
        df = pd.DataFrame(data['data'])

        if not ai_system or not ai_system['integrated_system'].is_trained:
            raise HTTPException(status_code=400, detail="AI 시스템이 학습되지 않았습니다")

        # 통합 예측 실행
        predictions = ai_system['integrated_system'].predict_integrated(df)

        return {
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 스케줄링 최적화 API

@app.post("/optimization/schedule")
async def optimize_schedule(data: Dict):
    """작업 스케줄링 최적화"""
    try:
        method = data.get('method', 'lagrange')
        n_jobs = data.get('n_jobs', 20)

        # 작업 데이터 생성
        jobs_df = energy_system.scheduler.create_job_data(n_jobs)

        # 스케줄링 최적화
        if method == 'ERD':
            schedule = energy_system.scheduler.erd_scheduling(jobs_df)
        else:
            schedule = energy_system.scheduler.optimize_with_peak_constraint(jobs_df, method)

        # 평가
        evaluation = energy_system.scheduler.evaluate_schedule(schedule, jobs_df)
        constraints = energy_system.constraint_manager.validate_all_constraints(schedule)

        return {
            "schedule": schedule.to_dict('records'),
            "evaluation": evaluation,
            "constraints": constraints,
            "method": method
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/optimization/tou-analysis")
async def analyze_tou_pricing():
    """TOU 요금제 분석"""
    try:
        # TOU 요금표 정보
        tou_summary = energy_system.tou_model.get_period_summary()

        # 24시간 요금 정보
        hourly_rates = []
        for hour in range(24):
            rate_info = energy_system.tou_model.get_hourly_price(hour)
            hourly_rates.append({
                "hour": hour,
                "price": rate_info['total_price'],
                "period": rate_info['period'],
                "season": rate_info['season']
            })

        return {
            "summary": tou_summary,
            "hourly_rates": hourly_rates
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 실시간 모니터링 API

@app.get("/monitoring/realtime")
async def get_realtime_data():
    """실시간 모니터링 데이터"""
    try:
        # 실시간 데이터 시뮬레이션
        current_hour = datetime.now().hour

        # 현재 시간 TOU 요금
        current_price = energy_system.tou_model.get_hourly_price(current_hour)

        # 모의 실시간 전력 데이터
        simulated_power = np.random.normal(800, 100)
        simulated_anomaly_score = np.random.uniform(0, 1)

        return {
            "timestamp": datetime.now().isoformat(),
            "current_power": simulated_power,
            "anomaly_score": simulated_anomaly_score,
            "is_anomaly": simulated_anomaly_score > 0.7,
            "current_price": current_price,
            "cost_per_hour": simulated_power * current_price['total_price'],
            "peak_limit": energy_system.scheduler.peak_power_limit,
            "peak_utilization": simulated_power / energy_system.scheduler.peak_power_limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 종합 분석 API

@app.post("/analysis/comprehensive")
async def run_comprehensive_analysis(background_tasks: BackgroundTasks, data: Dict):
    """종합 분석 실행"""
    try:
        n_jobs = data.get('n_jobs', 20)
        n_samples = data.get('n_samples', 50000)

        # 백그라운드에서 분석 실행
        background_tasks.add_task(run_analysis, n_jobs, n_samples)

        return {"message": "종합 분석이 백그라운드에서 시작되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def run_analysis(n_jobs: int, n_samples: int):
    """백그라운드 분석 실행"""
    try:
        results = energy_system.run_comprehensive_analysis(n_jobs, n_samples)
        logger.info("✅ 종합 분석 완료")
        # 결과를 데이터베이스나 캐시에 저장할 수 있음
    except Exception as e:
        logger.error(f"❌ 종합 분석 실패: {str(e)}")