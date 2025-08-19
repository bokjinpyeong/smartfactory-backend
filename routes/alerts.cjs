const express = require('express');
const router = express.Router();

// ALERTS_ENABLED=0이면 빈 배열 반환(스텁)
router.get('/peak', (req, res) => {
  if (process.env.ALERTS_ENABLED === '0') return res.json([]);
  // 실제 구현 전까지는 204로 무소식 처리
  return res.status(204).end();
});

module.exports = router;
