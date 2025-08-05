// 📁 routes/lineOrder.cjs
const express = require("express");
const router = express.Router();
const db = require("../db/connection.cjs");

router.post("/order", async (req, res) => {
  const updatedData = req.body;

  const executeQueries = async () => {
    for (const { lineId, equipment } of updatedData) {
      // ✅ 1. facilities 테이블에 lineId가 존재하는지 확인
      const facilityExists = await new Promise((resolve, reject) => {
        const checkFacilitySql = `SELECT COUNT(*) AS count FROM facilities WHERE fac_id = ?`;
        db.query(checkFacilitySql, [lineId], (err, results) => {
          if (err) return reject(err);
          resolve(results[0].count > 0);
        });
      });

      // ✅ 2. 존재하지 않으면 INSERT
      if (!facilityExists) {
        await new Promise((resolve, reject) => {
          const insertFacilitySql = `INSERT INTO facilities (fac_id) VALUES (?)`;
          db.query(insertFacilitySql, [lineId], (err) => {
            if (err) return reject(err);
            resolve();
          });
        });
      }

      // ✅ 3. used_fac 처리 (존재 시 UPDATE, 없으면 INSERT)
      for (let index = 0; index < equipment.length; index++) {
        const equipName = equipment[index];

        const exists = await new Promise((resolve, reject) => {
          const sql = `SELECT COUNT(*) AS count FROM used_fac WHERE fac_id = ? AND product_id = ?`;
          db.query(sql, [lineId, equipName], (err, results) => {
            if (err) return reject(err);
            resolve(results[0].count > 0);
          });
        });

        if (exists) {
          // UPDATE
          const updateSql = `UPDATE used_fac SET m_index = ? WHERE fac_id = ? AND product_id = ?`;
          await new Promise((resolve, reject) => {
            db.query(updateSql, [index, lineId, equipName], (err) => {
              if (err) return reject(err);
              resolve();
            });
          });
        } else {
          // INSERT
          const insertSql = `INSERT INTO used_fac (fac_id, product_id, m_index) VALUES (?, ?, ?)`;
          await new Promise((resolve, reject) => {
            db.query(insertSql, [lineId, equipName, index], (err) => {
              if (err) return reject(err);
              resolve();
            });
          });
        }
      }
    }
  };

  try {
    await executeQueries();
    return res.json({ success: true });
  } catch (err) {
    console.error("❌ 설비 순서 저장 실패:", err);
    return res.status(500).json({ success: false, message: "DB 처리 중 오류 발생" });
  }
});

module.exports = router;
