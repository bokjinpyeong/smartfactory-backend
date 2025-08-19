const express = require("express");
const path = require("path");
const { spawn } = require("child_process");
const requireAuth = require("../middleware/requireAuth.cjs");

const { LambdaClient, InvokeCommand } = require("@aws-sdk/client-lambda");
const AWS_REGION = process.env.AWS_REGION || "ap-northeast-2";
const LAMBDA_FN  = process.env.LAMBDA_SCHEDULE_FN || process.env.SCHEDULE_FN || "schedule_optimizer";
const lambda = new LambdaClient({ region: AWS_REGION });

const router = express.Router();

function runLocal(payload) {
  return new Promise((resolve, reject) => {
    const py = process.env.OPTIMIZER_PY || "/home/ubuntu/venvs/optimizer/bin/python";
    const script = path.join(__dirname, "..", "python", "optimizer.py");
    const ps = spawn(py, [script], { env: process.env });

    let out="", err="";
    ps.stdout.on("data", d => out += d);
    ps.stderr.on("data", d => err += d);
    ps.on("close", code => {
      if (code !== 0) return reject(new Error(err || `optimizer exit ${code}`));
      try { resolve(JSON.parse(out)); } catch { resolve({ ok:true, raw: out }); }
    });

    ps.stdin.write(JSON.stringify(payload || {}));
    ps.stdin.end();
  });
}

async function runLambda(payload) {
  const out = await lambda.send(new InvokeCommand({
    FunctionName: LAMBDA_FN,
    Payload: Buffer.from(JSON.stringify({ detail: payload })),
  }));
  const raw = Buffer.from(out.Payload || []).toString("utf-8") || "{}";
  let parsed; try { parsed = JSON.parse(raw); } catch { parsed = { raw }; }
  const body = parsed && typeof parsed === "object" && "body" in parsed
    ? (() => { try { return JSON.parse(parsed.body); } catch { return parsed.body; } })()
    : parsed;

  if (out.FunctionError) throw new Error(`Lambda FunctionError: ${out.FunctionError}`);
  return body;
}

// 헬스
router.get("/worksimul/ping", (_req, res) => res.json({ ok:true, route:"worksimul" }));

// 실제 시뮬레이터
router.post("/worksimul", requireAuth, async (req, res) => {
  const { hour_prices = [], jobs = [], config = {} } = req.body || {};
  if (!Array.isArray(hour_prices) || hour_prices.length === 0) {
    return res.status(400).json({ ok:false, message:"hour_prices(24) 필요" });
  }

  const preferLocal = process.env.FORCE_LOCAL === "1";
  try {
    const result = preferLocal
      ? await runLocal({ hour_prices, jobs, config })
      : await runLambda({ hour_prices, jobs, config });
    return res.json({ ok:true, ...result });
  } catch (e) {
    console.error("[worksimul] primary failed:", e.message);
    if (!preferLocal) {
      try {
        const fb = await runLocal({ hour_prices, jobs, config });
        return res.json({ ok:true, ...fb });
      } catch (e2) {
        console.error("[worksimul] fallback(local) failed:", e2.message);
      }
    }
    return res.status(502).json({ ok:false, message:"worksimul failed" });
  }
});

module.exports = router;