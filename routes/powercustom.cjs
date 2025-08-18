const express = require("express");
const router = express.Router();
const db = require("../db/connection.cjs");
const requireAuth = require("../middleware/requireAuth.cjs");

/* ───────── 공통 유틸 ───────── */
function logSqlError(tag, err) {
  console.error(`❌ ${tag}`, {
    message: err?.message, code: err?.code, sqlState: err?.sqlState,
    sqlMessage: err?.sqlMessage, sql: err?.sql,
  });
}
function normYMD(s) {
  if (!s) return null;
  const p = String(s).slice(0, 10).replace(/[./]/g, "-");
  return /^\d{4}-\d{2}-\d{2}$/.test(p) ? p : null;
}

/* =========================================================
 * ✅ META: 보유 데이터 전체 범위
 *    - min/max: DATE(Timestamp)
 *    - count  : DISTINCT DATE(Timestamp)
 * ========================================================= */
router.get("/meta", requireAuth, (req, res) => {
  const uid = req.user.id;
  const sql = `
    SELECT
      DATE(MIN(\`Timestamp\`)) AS minDate,
      DATE(MAX(\`Timestamp\`)) AS maxDate,
      COUNT(DISTINCT DATE(\`Timestamp\`)) AS days
    FROM \`sensor_power_raw\`
    WHERE \`user_id\` = ?
  `;
  db.query(sql, [uid], (err, rows) => {
    if (err) {
      logSqlError("/power-custom/meta", err);
      return res.status(500).json({ ok:false, message:"메타 조회 실패" });
    }
    const r = rows?.[0] || {};
    res.json({
      ok   : true,
      minDate: r.minDate || null,
      maxDate: r.maxDate || null,
      count  : Number(r.days || 0),
    });
  });
});

/* =========================================================
 * ✅ DAY: 일 단위 집계
 *    - 단일:  date=YYYY-MM-DD
 *    - 구간:  from=YYYY-MM-DD&to=YYYY-MM-DD
 *    - 집계:  SUM(Power_Consumption_Realistic), DATE(Timestamp)
 *    - 구간은 반개구간: start <= ts < end+1day
 *    - 응답: { ok:true, rows:[{date, power, price:0}] }
 * ========================================================= */
router.get("/day", requireAuth, (req, res) => {
  const uid  = req.user.id;
  const date = normYMD(req.query.date);
  const from = normYMD(req.query.from);
  const to   = normYMD(req.query.to);

  // ① 단일 일자
  if (date) {
    const sql = `
      SELECT
        ? AS date,
        ROUND(COALESCE(SUM(\`Power_Consumption_Realistic\`), 0), 2) AS power,
        0 AS price
      FROM \`sensor_power_raw\`
      WHERE \`user_id\` = ?
        AND \`Timestamp\` >= TIMESTAMP(CONCAT(?, ' 00:00:00'))
        AND \`Timestamp\` <  TIMESTAMP(CONCAT(?, ' 00:00:00')) + INTERVAL 1 DAY
    `;
    const params = [date, uid, date, date];
    return db.query(sql, params, (err, rows) => {
      if (err) {
        logSqlError("/power-custom/day(단일)", err);
        return res.status(500).json({ ok:false, message:"집계 실패" });
      }
      const out = rows?.[0]
        ? [{ date, power: Number(rows[0].power || 0), price: 0 }]
        : [];
      return out.length ? res.json({ ok:true, rows: out }) : res.status(204).end();
    });
  }

  // ② 구간(from~to)
  if (!from || !to) {
    return res.status(400).json({ ok:false, message:"date 또는 from/to(YYYY-MM-DD) 필요" });
  }

  const sql = `
    SELECT
      DATE(\`Timestamp\`) AS date,
      ROUND(COALESCE(SUM(\`Power_Consumption_Realistic\`), 0), 2) AS power,
      0 AS price
    FROM \`sensor_power_raw\`
    WHERE \`user_id\` = ?
      AND \`Timestamp\` >= TIMESTAMP(CONCAT(?, ' 00:00:00'))
      AND \`Timestamp\` <  TIMESTAMP(CONCAT(?, ' 00:00:00')) + INTERVAL 1 DAY
    GROUP BY DATE(\`Timestamp\`)
    ORDER BY date ASC
  `;
  const params = [uid, from, to];

  db.query(sql, params, (err, rows) => {
    if (err) {
      logSqlError("/power-custom/day(구간)", err);
      return res.status(500).json({ ok:false, message:"집계 실패" });
    }
    const safe = (rows || []).map(r => ({
      date : r.date,
      power: Number(r.power || 0),
      price: 0,
    }));
    return safe.length ? res.json({ ok:true, rows: safe }) : res.status(204).end();
  });
});

/* =========================================================
 * ✅ RANGE: 구간 집계 (GET/POST 동일)
 *    - 프론트 폴백용으로 /day(구간)과 동일 로직 유지
 * ========================================================= */
function handleRange(req, res) {
  const uid = req.user.id;
  const q   = req.method === "GET" ? req.query : (req.body || {});
  const from = normYMD(q.from || q.start);
  const to   = normYMD(q.to   || q.end);

  if (!from || !to) {
    return res.status(400).json({ ok:false, message:"from/to(YYYY-MM-DD) 필요" });
  }

  const sql = `
    SELECT
      DATE(\`Timestamp\`) AS date,
      ROUND(COALESCE(SUM(\`Power_Consumption_Realistic\`), 0), 2) AS power,
      0 AS price
    FROM \`sensor_power_raw\`
    WHERE \`user_id\` = ?
      AND \`Timestamp\` >= TIMESTAMP(CONCAT(?, ' 00:00:00'))
      AND \`Timestamp\` <  TIMESTAMP(CONCAT(?, ' 00:00:00')) + INTERVAL 1 DAY
    GROUP BY DATE(\`Timestamp\`)
    ORDER BY date ASC
  `;
  const params = [uid, from, to];

  db.query(sql, params, (err, rows) => {
    if (err) {
      logSqlError(`/power-custom/range ${req.method}`, err);
      return res.status(500).json({ ok:false, message:"집계 실패" });
    }
    const safe = (rows || []).map(r => ({
      date : r.date,
      power: Number(r.power || 0),
      price: 0,
    }));
    return safe.length ? res.json({ ok:true, rows: safe }) : res.status(204).end();
  });
}
router.get ("/range", requireAuth, handleRange);
router.post("/range", requireAuth, handleRange);

module.exports = router;
