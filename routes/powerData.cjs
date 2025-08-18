// 📁 backend/routes/powerData.cjs
const express = require("express");
const router = express.Router();
const db = require("../db/connection.cjs");
const requireAuth = require("../middleware/requireAuth.cjs");

function logSqlError(tag, err) {
  console.error(`❌ ${tag}`, {
    message: err?.message, code: err?.code, sqlState: err?.sqlState,
    sqlMessage: err?.sqlMessage, sql: err?.sql,
  });
}

/* =========================================================
 * ✅ 주간: 최신 데이터 시각 기준 "최근 7일" (일별 합)
 *    - 반개구간 사용: [start, end)
 *    - start = DATE(max_ts) - 6일 00:00:00
 *    - end   = DATE(max_ts) + 1일 00:00:00  → 마지막 날을 "하루 전체"로 포함
 * ========================================================= */
router.get("/weekly", requireAuth, (req, res) => {
  const uid = req.user.id;

  const qMax = "SELECT MAX(`Timestamp`) AS max_ts FROM `sensor_power_raw` WHERE `user_id` = ?";
  db.query(qMax, [uid], (err, r) => {
    if (err) {
      logSqlError("max_ts 조회 실패(/weekly)", err);
      return res.status(500).json({ success: false, message: "소비량 조회 실패" });
    }
    const maxTs = r?.[0]?.max_ts;
    if (!maxTs) return res.json({ success: true, rows: [] }); // 데이터 없음

    const q = `
      SELECT
        DATE(\`Timestamp\`) AS date,
        ROUND(COALESCE(SUM(\`Power_Consumption_Realistic\`), 0), 2) AS power
      FROM \`sensor_power_raw\`
      WHERE \`user_id\` = ?
        AND \`Timestamp\` >= DATE(?) - INTERVAL 6 DAY
        AND \`Timestamp\` <  DATE(?) + INTERVAL 1 DAY
      GROUP BY DATE(\`Timestamp\`)
      ORDER BY date;
    `;
    db.query(q, [uid, maxTs, maxTs], (e, rows) => {
      if (e) {
        logSqlError("소비량 조회 실패(/weekly)", e);
        return res.status(500).json({ success: false, message: "소비량 조회 실패" });
      }

      // 숫자/타입 보정
      const safe = (rows || []).map((r) => ({
        date: r.date,                          // 'YYYY-MM-DD'
        power: Number(r.power ?? 0) || 0,      // DECIMAL 문자열 문제 방지
      }));

      res.json({ success: true, rows: safe });
    });
  });
});

/* =========================================================
 * ✅ 월간/범위: from~to가 없으면 max_ts의 "그 달 1일~말일"
 *    - 반개구간 사용: [start, end)
 *    - start = COALESCE(from, DATE_FORMAT(max_ts, '%Y-%m-01'))
 *    - end   = DATE_ADD(COALESCE(to, LAST_DAY(max_ts)), INTERVAL 1 DAY)
 *    - 옵션 필터: machineId, lineId
 * ========================================================= */
// /api/power-data/monthly?from=YYYY-MM-DD&to=YYYY-MM-DD&machineId=&lineId=
router.get("/monthly", requireAuth, (req, res) => {
  const uid = req.user.id;
  const { from, to, machineId, lineId } = req.query;

  const qMax = "SELECT MAX(`Timestamp`) AS max_ts FROM `sensor_power_raw` WHERE `user_id` = ?";
  db.query(qMax, [uid], (err, r) => {
    if (err) {
      logSqlError("max_ts 조회 실패(/monthly)", err);
      return res.status(500).json({ success: false, message: "기간 조회 실패" });
    }
    const maxTs = r?.[0]?.max_ts;

    // from/to 모두 없는 상태에서 max_ts도 없으면 반환할 기본 범위가 없음
    if (!maxTs && (!from || !to)) {
      return res.json({ success: true, rows: [] });
    }

    // 동적 WHERE 구성 (옵션 필터)
    const optMachine = machineId ? "AND spr.`Machine_ID` = ?" : "";
    const optLine = lineId ? "AND spr.`Production_Line_ID` = ?" : "";

    // 반개구간:  start <= ts < endExclusive
    const sql = `
      SELECT
        DATE(spr.\`Timestamp\`) AS date,
        ROUND(COALESCE(SUM(spr.\`Power_Consumption_Realistic\`), 0), 2) AS power
      FROM \`sensor_power_raw\` spr
      WHERE spr.\`user_id\` = ?
        AND spr.\`Timestamp\` >= COALESCE(?, DATE_FORMAT(?, '%Y-%m-01'))
        AND spr.\`Timestamp\` <  DATE_ADD(COALESCE(?, LAST_DAY(?)), INTERVAL 1 DAY)
        ${optMachine}
        ${optLine}
      GROUP BY DATE(spr.\`Timestamp\`)
      ORDER BY date;
    `;

    const params = [
      uid,
      (from || null), maxTs,     // start: COALESCE(?, DATE_FORMAT(?, '%Y-%m-01'))
      (to   || null), maxTs,     // end base: COALESCE(?, LAST_DAY(?)) → DATE_ADD(..., 1 DAY)
      ...(machineId ? [machineId] : []),
      ...(lineId ? [lineId] : []),
    ];

    db.query(sql, params, (e, rows) => {
      if (e) {
        logSqlError("기간 조회 실패(/monthly)", e);
        return res.status(500).json({ success: false, message: "기간 조회 실패" });
      }

      // 숫자/타입 보정
      const safe = (rows || []).map((r) => ({
        date: r.date,                          // 'YYYY-MM-DD'
        power: Number(r.power ?? 0) || 0,
      }));

      res.json({ success: true, rows: safe });
    });
  });
});

module.exports = router;
