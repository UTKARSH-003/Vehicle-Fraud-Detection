import { Router, type IRouter } from "express";

const router: IRouter = Router();
const servicePort = Number(process.env["FRAUD_SERVICE_PORT"] ?? 5051);
const serviceBase = `http://127.0.0.1:${servicePort}`;

function wait(milliseconds: number) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function fetchModelService(url: URL, init: RequestInit) {
  let lastError: unknown;
  for (let attempt = 0; attempt < 90; attempt += 1) {
    try {
      return await fetch(url, init);
    } catch (error) {
      lastError = error;
      if (attempt < 89) await wait(500);
    }
  }
  throw lastError;
}

router.use("/fraud", async (req, res) => {
  const target = new URL(req.originalUrl, serviceBase);
  const headers = new Headers();
  const accept = req.get("accept");
  if (accept) headers.set("accept", accept);
  if (req.body !== undefined && !["GET", "HEAD"].includes(req.method)) {
    headers.set("content-type", "application/json");
  }

  try {
    const upstream = await fetchModelService(target, {
      method: req.method,
      headers,
      body:
        req.body !== undefined && !["GET", "HEAD"].includes(req.method)
          ? JSON.stringify(req.body)
          : undefined,
    });
    const contentType = upstream.headers.get("content-type");
    if (contentType) res.setHeader("content-type", contentType);
    const cacheControl = upstream.headers.get("cache-control");
    if (cacheControl) res.setHeader("cache-control", cacheControl);
    res.status(upstream.status).send(
      Buffer.from(await upstream.arrayBuffer()),
    );
  } catch (error) {
    req.log.error(
      { err: error, path: target.pathname },
      "Fraud model service is unavailable",
    );
    res.status(503).json({
      error: "The fraud model service is warming up or unavailable. Please retry.",
    });
  }
});

export default router;
