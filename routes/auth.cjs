// 📁 backend/routes/auth.cjs
const express = require('express');
const router = express.Router();
const db = require('../db/connection.cjs');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const requireAuth = require('../middleware/requireAuth.cjs');

const JWT_SECRET = process.env.JWT_SECRET;

/* ─────────────────────────────────────────────────────────
 * 기본: /auth/ 진입 시 상태/가이드 + CORS 프리플라이트
 * ───────────────────────────────────────────────────────── */
router.options('*', (_req, res) => res.sendStatus(204));

router.get('/', (_req, res) => {
  res.json({
    ok: true,
    service: 'auth',
    hint: '로그인은 POST /auth/login, 회원가입은 POST /auth/register',
    endpoints: [
      'POST /auth/login',
      'POST /auth/register',
      'GET  /auth/me',
      'POST /auth/logout'
    ]
  });
});

/* ─────────────────────────────────────────────────────────
 * 회원가입: POST /auth/register
 * body: { company?, name?, phone?, email, password }
 * 응답: { success, message, userId? }
 * ───────────────────────────────────────────────────────── */
router.post('/register', async (req, res) => {
  const { company, name, phone, email, password } = req.body || {};
  if (!email || !password) {
    return res.status(400).json({ success: false, message: '이메일/비밀번호가 필요합니다.' });
  }

  try {
    const checkSql = 'SELECT id FROM users WHERE email = ? LIMIT 1';
    db.query(checkSql, [email], async (err, rows) => {
      if (err) return res.status(500).json({ success: false, message: '중복 검사 실패' });
      if (rows.length) {
        return res.status(409).json({ success: false, message: '이미 가입된 이메일입니다.' });
      }

      const hashed = await bcrypt.hash(password, 10);
      const insertSql = `
        INSERT INTO users (company, name, phone, email, password)
        VALUES (?, ?, ?, ?, ?)
      `;
      db.query(
        insertSql,
        [company ?? null, name ?? null, phone ?? null, email, hashed],
        (err2, result) => {
          if (err2) {
            return res.status(500).json({ success: false, message: '회원가입에 실패했습니다.', error: err2.code });
          }
          return res.json({ success: true, message: '회원가입 성공', userId: result.insertId });
        }
      );
    });
  } catch {
    return res.status(500).json({ success: false, message: '회원가입 처리 오류' });
  }
});

/* ─────────────────────────────────────────────────────────
 * 로그인: POST /auth/login
 * body: { email, password }
 * 응답: { success, token, user, message? }
 * 프론트는 token 키를 우선 사용(없으면 accessToken/jwt 등 백업키 탐색)
 * ───────────────────────────────────────────────────────── */
router.post('/login', (req, res) => {
  const { email, password } = req.body || {};
  if (!email || !password) {
    return res.status(400).json({ success: false, message: '이메일/비밀번호가 필요합니다.' });
  }

  const sql = 'SELECT id, name, email, password FROM users WHERE email = ? LIMIT 1';
  db.query(sql, [email], async (err, rows) => {
    if (err) return res.status(500).json({ success: false, message: 'DB 오류' });
    if (!rows.length) return res.status(401).json({ success: false, message: '이메일 또는 비밀번호가 올바르지 않습니다.' });

    const user = rows[0];
    try {
      const ok = await bcrypt.compare(password, user.password);
      if (!ok) return res.status(401).json({ success: false, message: '이메일 또는 비밀번호가 올바르지 않습니다.' });

      const token = jwt.sign(
        { id: user.id, email: user.email },
        JWT_SECRET,
        { expiresIn: '7d' }
      );

      // 프론트 호환: token, user 키 제공
      return res.json({
        success: true,
        token,
        user: { id: user.id, name: user.name, email: user.email }
      });
    } catch {
      return res.status(500).json({ success: false, message: '로그인 처리 오류' });
    }
  });
});

/* ─────────────────────────────────────────────────────────
 * 내 정보: GET /auth/me (보호됨)
 * 헤더: Authorization: Bearer <token>
 * 응답: { success, user }
 * 프론트는 data.user 또는 data 를 user로 인식
 * ───────────────────────────────────────────────────────── */
router.get('/me', requireAuth, (req, res) => {
  // 필요 시 DB 재조회 가능. 여기선 미들웨어에서 추출한 클레임 사용
  return res.json({
    success: true,
    user: { id: req.user.id, email: req.user.email }
  });
});

/* ─────────────────────────────────────────────────────────
 * 로그아웃: POST /auth/logout
 * (서버 세션이 없다면 클라이언트 토큰 삭제만으로 충분)
 * 응답: { success }
 * ───────────────────────────────────────────────────────── */
router.post('/logout', (_req, res) => {
  return res.json({ success: true });
});

module.exports = router;
