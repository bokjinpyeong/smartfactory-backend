const express = require('express');
const { LambdaClient, InvokeCommand } = require('@aws-sdk/client-lambda');

const PORT = process.env.API_PORT || 4000;
const REGION = process.env.AWS_REGION || 'ap-northeast-2';

const app = express();
app.use(express.json());

app.get('/api/healthz', (_req, res) => res.json({ ok: true }));

const lambda = new LambdaClient({ region: REGION });

app.post('/api/schedule/optimize', async (req, res) => {
  try {
    const payload = { detail: req.body }; // 람다에서 event.detail 로 읽음
    const out = await lambda.send(new InvokeCommand({
      FunctionName: 'schedule_optimizer',
      Payload: Buffer.from(JSON.stringify(payload)),
    }));
    const raw = out.Payload ? Buffer.from(out.Payload).toString() : '{}';
    let parsed; try { parsed = JSON.parse(raw); } catch { parsed = { raw }; }
    const body = parsed && parsed.body ? JSON.parse(parsed.body) : parsed;
    res.json(body);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'schedule_optimizer invoke failed' });
  }
});

app.listen(PORT, () => console.log(`API listening on ${PORT}`));
