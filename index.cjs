// 📁 backend/index.cjs — 안정본
const path = require("path");
const express = require("express");
const cors = require("cors");
const compression = require("compression");
require("dotenv").config();

const { LambdaClient, InvokeCommand } = require("@aws-sdk/client-lambda");
const app = express();

const PORT = Number(process.env.PORT || 4000);
const AWS_REGION = process.env.AWS_REGION || "ap-northeast-2";
const SCHEDULE_FN = process.env.SCHEDULE_FN || "schedule_optimizer";
const lambda = new LambdaClient({ region: AWS_REGION });

// 기본 미들웨어
app.set("trust proxy", true);
app.use(cors());
app.use(compression());
app.use(express.json({ limit: "2mb" }));
app.use(express.urlencoded({ extended: true }));
if (process.env.NODE_ENV !== "production") {
  app.use((req,_res,next)=>{ console.log(`➡ ${req.method} ${req.originalUrl}`); next(); });
}

// 라우터 로드
const authRoutes         = require("./routes/auth.cjs");
const lineOrderRoutes    = require("./routes/lineOrder.cjs");
const powerTypeRoutes    = require("./routes/powerType.cjs");
const powercustomRoutes  = require("./routes/powercustom.cjs");
const powerDataRoutes    = require("./routes/powerData.cjs");

// 마운트(※ 404보다 위!)
app.use("/auth", authRoutes);
app.use("/api/equipment",    lineOrderRoutes);
app.use("/api/powertype",    powerTypeRoutes);
app.use("/api/power-custom", powercustomRoutes);
// app.use("/api/power-data",   powercustomRoutes);
app.use("/api/power-data",   powerDataRoutes);

// 선택 라우터 (try-catch)
try { const alertRoutes = require("./routes/alerts.cjs"); app.use("/api/alerts", alertRoutes); console.log("[index] alerts router loaded"); } catch(e){ console.error("[index] alerts skip:", e.message); }
try { const liveRoutes  = require("./routes/live.cjs");   app.use("/api", liveRoutes);         console.log("[index] live router loaded"); }   catch(e){ console.error("[index] live skip:", e.message); }
try { const wsRoutes    = require("./routes/workSimul.cjs"); app.use("/api", wsRoutes);        console.log("[index] workSimul router loaded (/api/worksimul)"); } catch(e){ console.error("[index] workSimul skip:", e.message); }

// 헬스
app.get("/api/healthz", (_req,res)=>res.json({ok:true}));
app.get("/healthz",     (_req,res)=>res.json({ok:true}));

// 동기 람다 엔드포인트(유지)
app.post("/api/schedule/optimize", async (req, res) => {
  try {
    const payload = { detail: req.body };
    const out = await lambda.send(new InvokeCommand({
      FunctionName: SCHEDULE_FN,
      Payload: Buffer.from(JSON.stringify(payload)),
    }));
    const raw = out.Payload ? Buffer.from(out.Payload).toString() : "{}";
    let parsed; try { parsed = JSON.parse(raw); } catch { parsed = { raw }; }
    const body = parsed && parsed.body ? JSON.parse(parsed.body) : parsed;
    res.json(body);
  } catch (err) {
    console.error("[/api/schedule/optimize] invoke error:", err.message);
    res.status(500).json({ ok:false, error:"schedule_optimizer invoke failed" });
  }
});

// API 404 (맨 마지막!)
app.use("/api", (_req,res)=>res.status(404).json({ ok:false, message:"Not Found" }));

// 정적 SPA
const distDir = path.join(__dirname, "../femspj/dist");
app.use(express.static(distDir));
app.get("*", (req,res,next)=>{
  if (req.path.startsWith("/api/") || req.path.startsWith("/auth/") || req.path === "/healthz") return next();
  res.sendFile(path.join(distDir, "index.html"));
});

// 서버 시작
app.listen(PORT, ()=> console.log(`🚀 server listening on http://localhost:${PORT}`));
// confirm