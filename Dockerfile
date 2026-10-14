
FROM node:22-bookworm-slim

WORKDIR /app

# Install Python and the virtual-environment tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in an isolated environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY artifacts/claim-fraud/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install pnpm and build the React frontend
RUN npm install -g pnpm@10

COPY . .

RUN pnpm install --frozen-lockfile

RUN PORT=5000 BASE_PATH=/ pnpm --filter @workspace/claim-fraud run build

# Render supplies the PORT environment variable
ENV PYTHONPATH=/app/artifacts/claim-fraud
ENV FRAUD_SERVICE_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "export FRAUD_SERVICE_PORT=${PORT:-10000} && exec python -m fraud_ml.service"]
