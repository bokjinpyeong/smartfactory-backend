// routes/live.cjs
const express = require("express");
const { LambdaClient, InvokeCommand } = require("@aws-sdk/client-lambda");

const router = express.Router();

const REGION = process.env.AWS_REGION || "ap-northeast-2";
const FN = process.env.LAMBDA_FORECAST_FN || "forecast_lstm";
const ALIAS = process.env.LAMBDA_ALIAS || ""; // "live"면 forecast_lstm:live
const TIMEOUT_MS = parseInt(process.env.LIVE_PRICE_TIMEOUT_MS || "8000", 10);

const lambda = new LambdaClient({ region: REGION });

function withTimeout(p, ms) {
  return Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("lambda-timeout")), ms)),
  ]);
}

router.get("/live/price", async (req, res) => {
  // 로그인 유저 없으면 .env 기본값 사용
  const user_id = (req.user && req.user.id ? String(req.user.id) : (process.env.USER_ID_DEFAULT || "20"));
  const device_id = process.env.DEFAULT_DEVICE_ID || "M001";
  const horizon_min = parseInt(req.query.h || "60", 10);

  const FunctionName = ALIAS ? `${FN}:${ALIAS}` : FN;
  const Payload = Buffer.from(JSON.stringify({
    detail: { user_id, device_id, horizon_min }
  }));

  try {
    const out = await withTimeout(
      lambda.send(new InvokeCommand({ FunctionName, Payload })),
      TIMEOUT_MS
    );

    if (!out || !out.Payload) throw new Error("empty-payload");
    const resp = JSON.parse(Buffer.from(out.Payload).toString("utf8"));

    // 람다의 {statusCode, body} 형태를 그대로 전달
    const code = (resp && resp.statusCode) || 200;
    const body = resp.body ? resp.body : JSON.stringify(resp);
    res.status(code).type("application/json").send(body);
  } catch (err) {
    console.error("[/api/live/price] lambda fail:", err.message);
    // 빠른 폴백: 프론트는 "-"를 보여주고 리트라이 하도록 202로 리턴
    res.status(202).json({
      price: 0, unit: "KRW", updatedAt: null,
      meta: { fallback: "backend_timeout", reason: err.message }
    });
  }
});

module.exports = router;
