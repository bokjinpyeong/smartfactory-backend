// 📁 backend/routes/powerType.cjs
const express = require("express");
const router = express.Router();
const db = require("../db/connection.cjs");

// const verifyToken = require("../middleware/verifyToken.cjs"); // [삭제]: 사용 안 함  :삭제
const requireAuth = require("../middleware/requireAuth.cjs");     // [추가]

/**
 * GET /api/powertype/:userId
 * 응답: { group, type } | 204(No Content)
 */
router.get("/:userId", requireAuth, /* [추가] */ (req, res) => {   // [수정] requireAuth 추가
  const { userId } = req.params;

  // [추가] 토큰의 id와 경로의 userId가 같은지 검증 (권한 보호)
  if (String(req.user.id) !== String(userId)) {
    return res.status(403).json({ success: false, message: "Forbidden" });
  }

  const sql = `
    SELECT grp AS \`group\`, typ AS \`type\`
    FROM user_power_type
    WHERE user_id = ?
    LIMIT 1
  `;
  db.query(sql, [userId], (err, rows) => {
    if (err) {
      console.error("❌ GET /powertype DB 오류:", err);
      return res.status(500).json({ success: false, message: "DB 오류" });
    }
    if (rows.length === 0) return res.status(204).end();
    return res.json(rows[0]); // { group, type }
  });
});

/**
 * POST /api/powertype/:userId
 * body: { powerType: { group: '갑 II'|'을', type: '선택 I'|'선택 II'|'선택 III' } }
 * 응답: { success: true }
 */
router.post("/:userId", requireAuth, /* [추가] */ (req, res) => {  // [수정] requireAuth 추가
  const { userId } = req.params;

  // [추가] 토큰의 id와 경로의 userId가 같은지 검증 (권한 보호)
  if (String(req.user.id) !== String(userId)) {
    return res.status(403).json({ success: false, message: "Forbidden" });
  }

  const { powerType } = req.body || {};
  if (!powerType || !powerType.group || !powerType.type) {
    return res.status(400).json({ success: false, message: "잘못된 요청 본문" });
  }

  // [수정] 프로젝트 규칙 적용
  const validGroups = ["갑 II", "을"];
  const validTypes  = ["선택 I", "선택 II", "선택 III"];

  if (!validGroups.includes(powerType.group) || !validTypes.includes(powerType.type)) {
    return res.status(400).json({ success: false, message: "허용되지 않은 값" });
  }

  // (선택) 유저 존재 확인
  const checkUserSql = `SELECT 1 FROM users WHERE id = ? LIMIT 1`;
  db.query(checkUserSql, [userId], (err, userRows) => {
    if (err) {
      console.error("❌ 유저 확인 DB 오류:", err);
      return res.status(500).json({ success: false, message: "DB 오류" });
    }
    if (userRows.length === 0) {
      return res.status(404).json({ success: false, message: "유저 없음" });
    }

    const upsertSql = `
      INSERT INTO user_power_type (user_id, grp, typ)
      VALUES (?, ?, ?)
      ON DUPLICATE KEY UPDATE
        grp = VALUES(grp),
        typ = VALUES(typ),
        updated_at = CURRENT_TIMESTAMP
    `;
    db.query(upsertSql, [userId, powerType.group, powerType.type], (err2) => {
      if (err2) {
        console.error("❌ UPSERT DB 오류:", err2);
        return res.status(500).json({ success: false, message: "DB 오류" });
      }
      return res.json({ success: true });
    });
  });
});

module.exports = router;
