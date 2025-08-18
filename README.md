# 스마트팩토리 에너지 관리 시스템

IoT 기반 전력 최적화 및 이상탐지 시스템

## 📋 프로젝트 개요

본 시스템은 스마트팩토리 환경에서 설비별 전력 사용을 실시간으로 수집·분석하고, 에너지 효율화를 위한 최적화 및 이상 탐지 기능을 제공하는 통합 플랫폼입니다.

### 🎯 주요 기능

- **실시간 전력 모니터링**: IoT 센서를 통한 실시간 전력 데이터 수집
- **AI 기반 이상탐지**: 머신러닝을 활용한 비정상 전력 패턴 감지
- **전력 예측**: 시계열 분석을 통한 전력 소비 예측
- **TOU 기반 스케줄링**: 시간대별 차등 요금제를 고려한 생산 스케줄 최적화
- **실시간 대시보드**: WebSocket 기반 실시간 모니터링 인터페이스

## 🏗️ 시스템 아키텍처

```
📁 smartfactory_energy_system/
├── 📁 core/                    # 핵심 시스템
│   ├── config.py              # 설정 관리
│   ├── logger.py              # 로깅 시스템
│   └── exceptions.py          # 예외 처리
├── 📁 data/                   # 데이터 계층
│   ├── collector.py           # IoT 데이터 수집
│   ├── processor.py           # 데이터 전처리
│   └── validator.py           # 데이터 검증
├── 📁 models/                 # AI 모델
│   ├── anomaly_detector.py    # 이상탐지 모델
│   ├── power_predictor.py     # 전력 예측 모델
│   └── base_model.py          # 모델 기본 클래스
├── 📁 optimization/           # 최적화 엔진
│   ├── scheduler.py           # 스케줄링 최적화
│   ├── tou_pricing.py         # TOU 요금제 모델
│   └── constraints.py         # 제약 조건 관리
├── 📁 api/                    # API 서버
│   ├── routes.py              # API 엔드포인트
│   └── middleware.py          # API 미들웨어
├── 📁 dashboard/              # 대시보드
│   ├── real_time.py           # 실시간 대시보드
│   └── visualization.py      # 시각화 컴포넌트
├── 📁 tests/                  # 테스트
├── 📁 scripts/                # 유틸리티 스크립트
├── 📁 config/                 # 환경별 설정
├── requirements.txt           # 의존성 패키지
└── main.py                    # 메인 실행 파일
```

## 🚀 설치 및 실행

### 1. 환경 준비

```bash
# Python 3.8+ 필요
python --version

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 설정 파일 준비

```bash
# config 디렉토리의 설정 파일들을 환경에 맞게 수정
cp config/development.yaml.example config/development.yaml
```

### 4. 시스템 실행

#### 전체 시스템 실행
```bash
python main.py --mode full
```

#### API 서버만 실행
```bash
python main.py --mode api --port 8000
```

#### 데이터 처리만 실행
```bash
python main.py --mode processing
```

#### 대화형 모드
```bash
python main.py --mode interactive
```

### 5. 운영 환경 실행
```bash
python main.py --env production --log-level INFO
```

## 📊 개발 원칙

### 1. 모듈화 및 독립성
- 각 AI 모델을 독립적인 모듈로 설계
- 이상탐지, 예측, 스케줄링 모듈 간 느슨한 결합
- 개별 모델의 업데이트가 전체 시스템에 미치는 영향 최소화

### 2. 확장 가능한 데이터 파이프라인
- 스트리밍 데이터 처리 우선 설계
- 실시간 센서 데이터 수집 → 전처리 → 모델 추론의 일관된 플로우
- 배치 처리와 실시간 처리의 하이브리드 구조

### 3. 장애 허용성 (Fault Tolerance)
- 단일 장애점 제거
- 한 모델의 실패가 전체 시스템 중단으로 이어지지 않도록
- 폴백 메커니즘: 복잡한 모델 실패 시 단순 룰로 대체

## 🤖 AI 모델

### 1. 전력 이상탐지 (Anomaly Detection)
- **알고리즘**: Isolation Forest, Local Outlier Factor
- **특징**: 실시간 이상 패턴 감지, 기계별 특성 학습
- **출력**: 이상 점수, 알림 레벨

### 2. 소비전력 예측 (Load Forecasting)
- **알고리즘**: LSTM, XGBoost, Random Forest
- **특징**: 시계열 예측, 다중 모델 앙상블
- **출력**: 시간별 전력 소비 예측

### 3. 전력 기반 스케줄링 (TOU 최적화)
- **알고리즘**: MIP(Mixed Integer Programming), Greedy
- **특징**: 피크 제약 조건, 비용 최소화
- **출력**: 최적 생산 스케줄

## 🌐 API 엔드포인트

### 데이터 관련
- `POST /api/v1/data/sensors` - 센서 데이터 수집
- `GET /api/v1/data/sensors` - 센서 데이터 조회
- `POST /api/v1/data/validate` - 데이터 검증

### 모델 관련
- `GET /api/v1/models` - 모델 목록 조회
- `POST /api/v1/models/train` - 모델 학습
- `POST /api/v1/models/predict` - 예측 요청
- `GET /api/v1/models/{model_id}/status` - 모델 상태

### 스케줄링 관련
- `POST /api/v1/scheduling/optimize` - 스케줄링 최적화
- `GET /api/v1/scheduling/jobs` - 작업 목록 조회

### TOU 요금제 관련
- `GET /api/v1/tou/rates` - 현재 전력 요금 조회
- `POST /api/v1/tou/calculate-cost` - 비용 계산
- `GET /api/v1/tou/optimization-recommendations` - 최적화 권장사항

### 모니터링 관련
- `GET /api/v1/monitoring/realtime` - 실시간 데이터
- `GET /api/v1/monitoring/dashboard` - 대시보드 데이터

## 📈 실시간 대시보드

WebSocket을 통한 실시간 데이터 스트리밍:

- **실시간 전력 모니터링**: 기계별 전력 소비 현황
- **이상 알림**: 비정상 패턴 감지시 즉시 알림
- **효율성 지표**: 전체 시스템 효율성 모니터링
- **비용 추적**: 실시간 전력 비용 계산

### WebSocket 연결
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('실시간 데이터:', data);
};
```

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함 테스트
pytest --cov=smartfactory_energy_system

# 특정 모듈 테스트
pytest tests/test_models.py
```

## 📝 개발 가이드

### 코드 스타일
```bash
# 코드 포매팅
black .

# 린트 검사
flake8 .

# 타입 검사
mypy .
```

### 새로운 모델 추가

1. `models/` 디렉토리에 새 모델 클래스 생성
2. `BaseModel` 상속하여 구현
3. `models/__init__.py`에 등록
4. 테스트 케이스 작성

### 새로운 API 엔드포인트 추가

1. `api/routes.py`에 새 라우트 추가
2. Pydantic 모델로 요청/응답 스키마 정의
3. 적절한 미들웨어 적용
4. API 문서 업데이트

## 🔧 설정

### 환경 변수
```bash
# .env 파일 생성
SMARTFACTORY_ENV=development
SMARTFACTORY_LOG_LEVEL=INFO
SMARTFACTORY_DB_URL=postgresql://user:password@localhost/smartfactory
SMARTFACTORY_MQTT_BROKER=localhost:1883
SMARTFACTORY_SECRET_KEY=your-secret-key
```

### 설정 파일 구조
```yaml
# config/development.yaml
system:
  api_host: "0.0.0.0"
  api_port: 8000
  mqtt_broker: "localhost"
  mqtt_port: 1883

data:
  batch_size: 1000
  train_ratio: 0.7
  scaling_method: "standard"

power:
  peak_power_limit: 1000.0
  safety_margin: 0.1
  tou_peak_rate: 1.5
  tou_off_peak_rate: 0.8
```

## 🔒 보안

### API 인증
```bash
# JWT 토큰 생성 예시
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# 인증이 필요한 API 호출
curl -X GET "http://localhost:8000/api/v1/system/statistics" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### 데이터 보안
- 센서 데이터 암호화 전송
- API 요청 제한 (Rate Limiting)
- 감사 로그 기록
- 민감한 데이터 마스킹

## 📊 모니터링 및 알림

### 시스템 메트릭
- **전력 소비**: 실시간 전력 사용량 모니터링
- **이상 탐지**: 비정상 패턴 감지 및 알림
- **API 성능**: 응답 시간, 처리량 모니터링
- **시스템 리소스**: CPU, 메모리, 디스크 사용률

### 알림 시스템
```python
# 알림 설정 예시
alerts = {
    "power_threshold": 200,  # kW
    "anomaly_threshold": 0.7,
    "efficiency_threshold": 0.7
}
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. MQTT 연결 실패
```bash
# MQTT 브로커 상태 확인
systemctl status mosquitto

# 포트 확인
netstat -an | grep 1883
```

#### 2. 모델 학습 실패
```bash
# GPU 사용 가능 여부 확인
python -c "import torch; print(torch.cuda.is_available())"

# 메모리 부족시 배치 크기 조정
# config/development.yaml에서 batch_size 값 감소
```

#### 3. API 응답 지연
```bash
# 성능 프로파일링
python -m cProfile main.py --mode api

# 로그 레벨 조정
python main.py --log-level WARNING
```

### 로그 분석
```bash
# 시스템 로그 확인
tail -f logs/smartfactory.log

# 에러 로그만 필터링
grep "ERROR" logs/smartfactory.log

# API 요청 로그 확인
grep "api.requests" logs/smartfactory.log
```

## 🚀 배포

### Docker 배포
```dockerfile
# Dockerfile 예시
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py", "--mode", "full", "--env", "production"]
```

```bash
# Docker 이미지 빌드
docker build -t smartfactory-energy .

# 컨테이너 실행
docker run -p 8000:8000 -e SMARTFACTORY_ENV=production smartfactory-energy
```

### 운영 환경 배포
```bash
# systemd 서비스 등록
sudo cp scripts/smartfactory.service /etc/systemd/system/
sudo systemctl enable smartfactory
sudo systemctl start smartfactory
```

## 📈 성능 최적화

### 데이터베이스 최적화
- 인덱스 최적화
- 파티셔닝 적용
- 쿼리 최적화

### 모델 최적화
- 모델 경량화 (Quantization)
- 배치 처리 최적화
- 캐싱 전략 적용

### API 최적화
- 비동기 처리 활용
- 커넥션 풀링
- 응답 압축

## 🤝 기여 가이드

### 개발 워크플로우
1. Fork 프로젝트
2. 기능 브랜치 생성 (`git checkout -b feature/새기능`)
3. 변경사항 커밋 (`git commit -am '새기능 추가'`)
4. 브랜치 푸시 (`git push origin feature/새기능`)
5. Pull Request 생성

### 코드 리뷰 기준
- 코드 스타일 준수
- 테스트 커버리지 유지
- 문서 업데이트
- 성능 영향 검토

## 📚 참고 자료

### 기술 문서
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [scikit-learn 사용자 가이드](https://scikit-learn.org/stable/user_guide.html)
- [OR-Tools 최적화 가이드](https://developers.google.com/optimization)

### 논문 및 연구
- "Production scheduling problem under peak power constraint" (IEEE 2020)
- Time-of-Use 요금제 기반 에너지 최적화 연구
- 제조업 이상탐지 알고리즘 비교 연구

## 📞 지원 및 문의

### 이슈 리포팅
- GitHub Issues를 통한 버그 리포트
- 기능 요청 및 개선 제안

### 개발팀 연락처
- **PM**: 정해윤 (프로젝트 기획)
- **AI/Data**: 정해윤 (예측 모델 및 분석 엔진)
- **Backend**: 김성훈, 윤재호 (서버 및 데이터 처리)
- **Frontend**: 이건호 (사용자 인터페이스)
- **DevOps**: 복진평 (시스템 통합 및 배포)

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 🎯 향후 개발 계획

### Phase 1 (현재)
- ✅ 기본 이상탐지 시스템
- ✅ 전력 예측 모델
- ✅ TOU 기반 스케줄링
- ✅ 실시간 대시보드

### Phase 2 (계획)
- 🔄 고급 딥러닝 모델 적용
- 🔄 다중 사이트 지원
- 🔄 모바일 앱 개발
- 🔄 고급 시각화 기능

### Phase 3 (미래)
- 📋 AI 기반 자동 최적화
- 📋 디지털 트윈 연동
- 📋 탄소 배출량 추적
- 📋 예지보전 기능

---

*이 문서는 프로젝트와 함께 지속적으로 업데이트됩니다.*
