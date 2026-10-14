import app from "./app";
import { logger } from "./lib/logger";
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

function locateFraudApp() {
  const candidates = [
    path.resolve(process.cwd(), "../claim-fraud"),
    path.resolve(process.cwd(), "artifacts/claim-fraud"),
    path.resolve(process.cwd(), "../../artifacts/claim-fraud"),
  ];
  return candidates.find((candidate) =>
    existsSync(path.join(candidate, "fraud_ml", "service.py")),
  );
}

const fraudAppRoot = locateFraudApp();
if (!fraudAppRoot) {
  throw new Error("Could not locate artifacts/claim-fraud/fraud_ml/service.py");
}

const packagedPython = path.resolve(fraudAppRoot, "../../.pythonlibs/bin/python");
const pythonExecutable =
  process.env["PYTHON_EXECUTABLE"] ??
  (existsSync(packagedPython) ? packagedPython : "python3");
const fraudServicePort = Number(process.env["FRAUD_SERVICE_PORT"] ?? 5051);
const childEnvironment = {
  ...process.env,
  FRAUD_SERVICE_PORT: String(fraudServicePort),
  FRAUD_SERVICE_HOST: "127.0.0.1",
  PYTHONPATH: [
    fraudAppRoot,
    process.env["PYTHONPATH"],
  ]
    .filter(Boolean)
    .join(path.delimiter),
  PYTHONUNBUFFERED: "1",
};
const fraudService: ChildProcess = spawn(
  pythonExecutable,
  ["-m", "fraud_ml.service"],
  {
    cwd: fraudAppRoot,
    env: childEnvironment,
    stdio: ["ignore", "pipe", "pipe"],
  },
);

fraudService.stdout?.on("data", (chunk: Buffer) => {
  const line = chunk.toString().trim();
  if (line) logger.info({ service: "fraud-model", message: line });
});
fraudService.stderr?.on("data", (chunk: Buffer) => {
  const line = chunk.toString().trim();
  if (line) logger.warn({ service: "fraud-model", message: line });
});
fraudService.on("error", (err) => {
  logger.error({ err }, "Could not start the fraud model service");
});
fraudService.on("exit", (code, signal) => {
  if (code !== 0) {
    logger.error(
      { code, signal },
      "Fraud model service exited unexpectedly",
    );
  }
});

const server = app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port, fraudServicePort }, "Server listening");
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => {
    fraudService.kill(signal);
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(1), 5000).unref();
  });
}

process.once("exit", () => {
  if (fraudService.exitCode === null && !fraudService.killed) {
    fraudService.kill("SIGTERM");
  }
});
