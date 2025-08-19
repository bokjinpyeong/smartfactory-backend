const express = require("express");
const router = express.Router();
const db = require("../db/connection.cjs");
const requireAuth = require("../middleware/requireAuth.cjs");
const util = require("util");

// mysql 콜백 → 프로미스
const query = util.promisify(db.query).bind(db);
const beginTransaction = util.promisify(db.beginTransaction).bind(db);
const commit = util.promisify(db.commit).bind(db);
const rollback = util.promisify(db.rollback).bind(db);

/* ─────────────────────────────────────────────────────
 * 유틸
 * ───────────────────────────────────────────────────── */
function asArray(x) {
  if (Array.isArray(x)) return x;
  if (x && Array.isArray(x.items)) return x.items;
  return [];
}

function validateRow(r) {
  return (
    r &&
    typeof r.product_id === "string" &&
    r.product_id.trim() &&
    typeof r.fac_id === "string" &&
    r.fac_id.trim() &&
    Number.isInteger(r.m_index) &&
    r.m_index >= 0
  );
}

/* ─────────────────────────────────────────────────────
 * GET /api/equipment/order
 * ───────────────────────────────────────────────────── */
router.get("/order", requireAuth, async (req, res) => {
  try {
    const userId = req.user.id;
    const sql = `
      SELECT product_id, fac_id, m_index
      FROM used_fac
      WHERE user_id = ?
      ORDER BY product_id, m_index
    `;
    const rows = await query(sql, [userId]);
    if (!rows || rows.length === 0) return res.status(204).end();
    return res.json(rows);
  } catch (err) {
    console.error("❌ GET /api/equipment/order DB 오류:", err);
    return res.status(500).json({ success: false, message: "DB 오류" });
  }
});

/* ─────────────────────────────────────────────────────
 * POST /api/equipment/order
 *  - 본문: [{product_id, fac_id, m_index}, ...] 또는 { items: [...] }
 *  - 동작: products / facilities 존재 보장 → used_fac 업서트 → m_index 재정렬
 * ───────────────────────────────────────────────────── */
router.post("/order", requireAuth, async (req, res) => {
  const userId = req.user.id;

  const bodyRows = asArray(req.body).map((r) => ({
    product_id: String(r.product_id || "").trim(),
    fac_id: String(r.fac_id || "").trim(),
    m_index: Number(r.m_index),
  }));

  if (bodyRows.length === 0 || !bodyRows.every(validateRow)) {
    return res
      .status(400)
      .json({ success: false, message: "유효하지 않은 요청 형식입니다." });
  }

  // product_id → rows 맵
  const mapByProduct = new Map();
  for (const r of bodyRows) {
    if (!mapByProduct.has(r.product_id)) mapByProduct.set(r.product_id, []);
    mapByProduct.get(r.product_id).push(r);
  }

  // 고유 product / fac 목록
  const productIds = [...new Set(bodyRows.map((r) => r.product_id))];
  const facIds = [...new Set(bodyRows.map((r) => r.fac_id))];

  try {
    await beginTransaction();

    /* 0) 외래키 원본 존재 보장 (없으면 생성)
     *  - PK/UNIQUE 충돌은 무시(이미 있으면 통과)
     *  - 스키마에 NOT NULL 추가 컬럼이 있으면 아래 주석의 대안 INSERT 사용
     */
    if (productIds.length) {
      // 최소 스키마: products(product_id)
      const values = productIds.map((id) => [id]);
      await query(`INSERT IGNORE INTO products (product_id) VALUES ?`, [values]);

      // // 예: products(product_id, product_name NOT NULL) 같은 경우
      // const prodVals = productIds.map((id) => [id, id]); // 기본값=ID
      // await query(
      //   `INSERT INTO products (product_id, product_name)
      //    VALUES ? ON DUPLICATE KEY UPDATE product_name = product_name`,
      //   [prodVals]
      // );
    }

    if (facIds.length) {
      // 최소 스키마: facilities(fac_id)
      const values = facIds.map((id) => [id]);
      await query(`INSERT IGNORE INTO facilities (fac_id) VALUES ?`, [values]);

      // // 예: facilities(fac_id, fac_name NOT NULL) 같은 경우
      // const facVals = facIds.map((id) => [id, id]); // 기본값=ID
      // await query(
      //   `INSERT INTO facilities (fac_id, fac_name)
      //    VALUES ? ON DUPLICATE KEY UPDATE fac_name = fac_name`,
      //   [facVals]
      // );
    }

    /* 1) used_fac 업서트: (user_id, product_id, fac_id) 기준으로 위치 갱신
     *  - 사전조건: used_fac에 UNIQUE(user_id, product_id, fac_id)
     *  - (user_id, product_id, m_index) 유니크는 제거 권장
     */
    const upsertSql = `
      INSERT INTO used_fac (user_id, product_id, fac_id, m_index)
      VALUES (?, ?, ?, ?)
      ON DUPLICATE KEY UPDATE m_index = VALUES(m_index)
    `;

    for (const rows of mapByProduct.values()) {
      for (const r of rows) {
        await query(upsertSql, [userId, r.product_id, r.fac_id, r.m_index]);
      }
    }

    /* 2) product별 m_index 0..n 재정렬 (구멍/중복 제거) */
    for (const [productId] of mapByProduct.entries()) {
      const list = await query(
        `
        SELECT fac_id, m_index
        FROM used_fac
        WHERE user_id = ? AND product_id = ?
        ORDER BY m_index, fac_id
        `,
        [userId, productId]
      );

      let idx = 0;
      for (const row of list) {
        await query(
          `UPDATE used_fac SET m_index = ? WHERE user_id = ? AND product_id = ? AND fac_id = ?`,
          [idx++, userId, productId, row.fac_id]
        );
      }
    }

    await commit();
    return res.json({ success: true });
  } catch (err) {
    try { await rollback(); } catch (_) {}
    console.error("❌ POST /api/equipment/order 처리 실패:", err);
    return res
      .status(500)
      .json({ success: false, message: err?.sqlMessage || "설비 순서 저장에 실패했습니다." });
  }
});

module.exports = router;
