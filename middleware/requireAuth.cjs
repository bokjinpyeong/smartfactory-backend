// 📌 로컬 JWT 검증 미들웨어
// - Authorization: Bearer <JWT> 형식 허용
// - jsonwebtoken.verify 로 즉시 검증 (외부 API 호출 없음)
// - payload 에서 사용자 id/email 을 꺼내 req.user 에 저장

const jwt = require("jsonwebtoken");

function fail(res, msg = "인증 실패") {
  return res.status(401).json({ success: false, message: msg });
}

module.exports = (req, res, next) => {
  try {
    if (req.method === "OPTIONS") return next();

    const authHeader = req.headers.authorization || "";
    if (!authHeader.startsWith("Bearer ")) {
      return fail(res, "인증 토큰 없음");
    }

    const token = authHeader.split(" ")[1];
    if (!token) return fail(res, "토큰 형식 오류");

    // .env에서 불러온 JWT_SECRET (이미 설정됨)
    const secret = process.env.JWT_SECRET || "qaz020207";
    if (!secret) {
      console.error("❌ JWT_SECRET 미설정");
      return fail(res, "서버 설정 오류(JWT_SECRET)");
    }

    // 로컬에서 즉시 검증
    const decoded = jwt.verify(token, secret, {
      algorithms: ["HS256"],
      clockTolerance: 10, // 서버 시간 오차 허용 (초)
    });

    // payload에서 사용자 ID/email 추출
    const userId =
      decoded.id || decoded.userId || decoded.uid || decoded.sub;
    if (!userId) return fail(res, "유효한 사용자 식별자가 없습니다.");

    req.user = {
      id: userId,
      email: decoded.email || null,
      role: decoded.role || null,
      payload: decoded,
    };

    next();
  } catch (err) {
    const name = err?.name || "AuthError";
    const msg =
      name === "TokenExpiredError" ? "토큰 만료"
      : name === "JsonWebTokenError" ? "유효하지 않은 토큰"
      : "인증 실패";
    console.error(`❌ requireAuth: ${name} - ${err?.message}`);
    return fail(res, msg);
  }
};
