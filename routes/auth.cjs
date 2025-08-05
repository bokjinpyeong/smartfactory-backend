const express = require('express');
const router = express.Router();
const db = require('../db/connection.cjs');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

// ✅ .env에서 JWT_SECRET 불러오기
const JWT_SECRET = process.env.JWT_SECRET;

// ✅ 회원가입 API (POST /auth/register)
router.post('/register', async (req, res) => {
  const { company, name, phone, email, password } = req.body;
  console.log('📦 회원가입 요청:', req.body);

  try {
    const hashedPassword = await bcrypt.hash(password, 10);
    const sql = `INSERT INTO users (company, name, phone, email, password, r_alerts) VALUES (?, ?, ?, ?, ?, ?)`;

    db.query(sql, [company, name, phone, email, hashedPassword, 1], (err, result) => {
      if (err) {
        console.error('❌ 회원가입 실패:', {
          code: err.code,
          errno: err.errno,
          sqlMessage: err.sqlMessage,
          sql: err.sql,
        });
        return res.status(500).send('회원가입 실패: 서버 오류');
      }
      console.log('✅ 회원가입 성공:', result);
      res.status(200).send('회원가입 성공');
    });
  } catch (err) {
    console.error('❌ 비밀번호 암호화 실패:', err);
    res.status(500).send('회원가입 실패: 서버 내부 오류');
  }
});

// ✅ 로그인 API (POST /auth/login)
router.post('/login', async (req, res) => {
  const { email, password } = req.body;
  console.log('🔑 로그인 요청:', req.body);

  try {
    const sql = `SELECT * FROM users WHERE email = ?`;
    db.query(sql, [email], async (err, results) => {
      if (err) {
        console.error('❌ 로그인 쿼리 실패:', err);
        return res.status(500).send('로그인 실패: 서버 오류');
      }

      if (results.length === 0) {
        return res.status(401).send('로그인 실패: 사용자 없음');
      }

      const user = results[0];
      const isMatch = await bcrypt.compare(password, user.password);

      if (!isMatch) {
        return res.status(401).send('로그인 실패: 비밀번호 불일치');
      }

      const token = jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, { expiresIn: '1h' });
      res.status(200).json({ message: '로그인 성공', token });
    });
  } catch (err) {
    console.error('❌ 로그인 처리 실패:', err);
    res.status(500).send('로그인 실패: 서버 내부 오류');
  }
});

module.exports = router;
