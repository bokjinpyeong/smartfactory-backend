Smart Factory Energy Optimization — Backend (Express)

Language order: English first, then Korean below.

1) English
Overview

Node.js + Express backend for the IoT-based Smart Factory project.
It exposes REST APIs for authentication, power data queries/aggregation, live price, alerts, and job schedule optimization (via AWS Lambda or local Python).

Entrypoint & Port

Entrypoint: backend/index.cjs

Port: 4000

Health: GET /healthz, GET /api/healthz → { ok: true }

Folder Structure (actual)
backend/
├─ index.cjs                # main server
├─ server.js                # legacy lambda-invoke example (not for prod)
├─ routes/
│  ├─ auth.cjs              # /auth/*
│  ├─ lineOrder.cjs         # /api/equipment/*
│  ├─ powerType.cjs         # /api/powertype/*
│  ├─ powercustom.cjs       # /api/power-custom/*
│  ├─ powerData.cjs         # /api/power-data/*
│  ├─ live.cjs              # /api/live/*
│  ├─ alerts.cjs            # /api/alerts/*
│  └─ workSimul.cjs         # /api/worksimul*
├─ db/connection.cjs        # MySQL connection (mysql2)
├─ middleware/requireAuth.cjs
├─ python/optimizer.py      # optional local fallback
├─ .env                     # environment
└─ package.json             # dependencies (no scripts required)

One-shot Setup & Run (copy/paste)
# 0) move into backend
cd backend

# 1) write .env (no editing needed)
cat > .env <<'EOF'
# Server
PORT=4000
NODE_ENV=production
CORS_ORIGIN=https://api.sensor-tive.com

# DB (local Docker MySQL started below)
DB_HOST=127.0.0.1
DB_USER=sf_user
DB_PASS=sf_pass_2025!
DB_NAME=smartfactory

# Auth
JWT_SECRET=Qf3Q2rY6z9L1u0W8t7J4s1K6b2N8p4D3   # demo secret

# AWS / Lambda
AWS_REGION=ap-northeast-2
SCHEDULE_FN=schedule_optimizer
LAMBDA_SCHEDULE_FN=schedule_optimizer
LAMBDA_FORECAST_FN=forecast_lstm
LAMBDA_ALIAS=

# Features
ALERTS_ENABLED=0
LIVE_PRICE_TIMEOUT_MS=8000

# Local optimizer (optional, used only if prefer=local)
OPTIMIZER_PY=/home/ubuntu/venvs/optimizer/bin/python
EOF

# 2) start MySQL via Docker (data persisted to ./mysql-data)
cat > docker-compose.yml <<'EOF'
version: "3.9"
services:
  mysql:
    image: mysql:8.0
    container_name: sf-mysql
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: root_2025!
      MYSQL_DATABASE: smartfactory
      MYSQL_USER: sf_user
      MYSQL_PASSWORD: sf_pass_2025!
    ports:
      - "3306:3306"
    volumes:
      - ./mysql-data:/var/lib/mysql
      - ./init.sql:/docker-entrypoint-initdb.d/00-init.sql:ro
EOF

# 3) write init.sql (minimal schema)
cat > init.sql <<'EOF'
CREATE TABLE IF NOT EXISTS sensor_power_raw (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id BIGINT NULL,
  device_id VARCHAR(64) NOT NULL,
  voltage FLOAT, current FLOAT, power_w FLOAT,
  Timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY (user_id), KEY (device_id), KEY (Timestamp)
);

CREATE TABLE IF NOT EXISTS power_type (
  user_id BIGINT PRIMARY KEY,
  grp VARCHAR(32) NOT NULL,
  typ VARCHAR(32) NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS used_fac (
  user_id BIGINT NOT NULL,
  product_id VARCHAR(64) NOT NULL,
  fac_id VARCHAR(64) NOT NULL,
  line_order INT NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, product_id, fac_id)
);

CREATE TABLE IF NOT EXISTS facilities (
  fac_id VARCHAR(64) PRIMARY KEY
);
EOF

# 4) bring up DB
docker compose up -d
# wait ~10s until MySQL is ready

# 5) install deps & start API
npm i
node index.cjs
# API listens on :4000

Health & Sanity Checks
curl -s http://127.0.0.1:4000/healthz
curl -s http://127.0.0.1:4000/api/healthz

NGINX (HTTPS reverse proxy)
server {
  listen 80;
  listen [::]:80;
  server_name api.sensor-tive.com;
  return 301 https://$host$request_uri;
}

server {
  listen 443 ssl http2;
  listen [::]:443 ssl http2;
  server_name api.sensor-tive.com;

  ssl_certificate     /etc/letsencrypt/live/api.sensor-tive.com/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/api.sensor-tive.com/privkey.pem;

  location /api/ {
    proxy_pass         http://127.0.0.1:4000/;
    proxy_http_version 1.1;
    proxy_set_header   Host $host;
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;
  }

  location /healthz {
    proxy_pass http://127.0.0.1:4000/healthz;
  }
}

Systemd (EC2)
# /etc/systemd/system/smartfactory-backend.service
[Unit]
Description=SmartFactory Backend
After=network.target docker.service
Requires=docker.service

[Service]
Environment=NODE_ENV=production
WorkingDirectory=/opt/smartfactory/backend
ExecStart=/usr/bin/node index.cjs
Restart=always
User=ubuntu
Group=ubuntu

[Install]
WantedBy=multi-user.target

sudo systemctl daemon-reload
sudo systemctl enable smartfactory-backend
sudo systemctl restart smartfactory-backend
journalctl -u smartfactory-backend -f

API Map (actual)

Base URL: https://api.sensor-tive.com

# Health
GET  /healthz
GET  /api/healthz

# Auth
GET  /auth/
POST /auth/register
POST /auth/login
GET  /auth/me
POST /auth/logout

# Equipment / Line order
GET  /api/equipment/order
POST /api/equipment/order

# Power type (per user)
GET  /api/powertype/:userId
POST /api/powertype/:userId

# Power custom aggregation
GET  /api/power-custom/meta
GET  /api/power-custom/day
POST /api/power-custom/range

# Power data aggregation
GET  /api/power-data/weekly
GET  /api/power-data/monthly

# Live price (Lambda: forecast_lstm)
GET  /api/live/price

# Alerts
GET  /api/alerts/peak

# Work scheduling simulator
GET  /api/worksimul/ping
POST /api/worksimul

2) 한국어
개요

IoT 기반 스마트팩토리 백엔드입니다. 인증, 전력 데이터 집계/조회, 실시간 가격, 알림, 스케줄 최적화(람다/로컬 파이썬)를 제공합니다.

즉시 실행(복붙)
cd backend

# .env 생성(수정 불필요)
cat > .env <<'EOF'
PORT=4000
NODE_ENV=production
CORS_ORIGIN=https://api.sensor-tive.com
DB_HOST=127.0.0.1
DB_USER=sf_user
DB_PASS=sf_pass_2025!
DB_NAME=smartfactory
JWT_SECRET=Qf3Q2rY6z9L1u0W8t7J4s1K6b2N8p4D3
AWS_REGION=ap-northeast-2
SCHEDULE_FN=schedule_optimizer
LAMBDA_SCHEDULE_FN=schedule_optimizer
LAMBDA_FORECAST_FN=forecast_lstm
LAMBDA_ALIAS=
ALERTS_ENABLED=0
LIVE_PRICE_TIMEOUT_MS=8000
OPTIMIZER_PY=/home/ubuntu/venvs/optimizer/bin/python
EOF

# Docker MySQL + 초기 스키마
cat > docker-compose.yml <<'EOF'
version: "3.9"
services:
  mysql:
    image: mysql:8.0
    container_name: sf-mysql
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: root_2025!
      MYSQL_DATABASE: smartfactory
      MYSQL_USER: sf_user
      MYSQL_PASSWORD: sf_pass_2025!
    ports:
      - "3306:3306"
    volumes:
      - ./mysql-data:/var/lib/mysql
      - ./init.sql:/docker-entrypoint-initdb.d/00-init.sql:ro
EOF

cat > init.sql <<'EOF'
CREATE TABLE IF NOT EXISTS sensor_power_raw (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id BIGINT NULL,
  device_id VARCHAR(64) NOT NULL,
  voltage FLOAT, current FLOAT, power_w FLOAT,
  Timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY (user_id), KEY (device_id), KEY (Timestamp)
);
CREATE TABLE IF NOT EXISTS power_type (
  user_id BIGINT PRIMARY KEY,
  grp VARCHAR(32) NOT NULL,
  typ VARCHAR(32) NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS used_fac (
  user_id BIGINT NOT NULL,
  product_id VARCHAR(64) NOT NULL,
  fac_id VARCHAR(64) NOT NULL,
  line_order INT NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, product_id, fac_id)
);
CREATE TABLE IF NOT EXISTS facilities ( fac_id VARCHAR(64) PRIMARY KEY );
EOF

docker compose up -d

npm i
node index.cjs

헬스 체크
curl -s http://127.0.0.1:4000/healthz
curl -s http://127.0.0.1:4000/api/healthz

NGINX (HTTPS 프록시)

위 블록 그대로 사용(도메인/인증서 경로 고정됨).

Systemd (EC2)

위 블록 그대로 사용 후 systemctl 명령 복붙 실행.
