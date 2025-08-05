const express = require('express');
const path = require('path');
const cors = require('cors');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 5000;
const db = require('./db/connection.cjs');

app.use(cors());
app.use(express.json());

const authRoutes = require('./routes/auth.cjs');
const lineOrderRoutes = require('./routes/lineOrder.cjs');


// ✅ 인증 API 먼저 등록 (순서 중요!)
app.use('/auth', authRoutes);
app.use('/api/equipment', lineOrderRoutes);
// ✅ 정적 파일 제공
app.use(express.static(path.join(__dirname, '../femspj/dist')));

// ✅ 새로고침 보완 라우팅 (순서 중요: *가 가장 마지막에 위치해야 함)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../femspj/dist/index.html'));
});

app.listen(port, () => {
  console.log(`🚀 서버 실행 중: http://localhost:${port}`);
});
