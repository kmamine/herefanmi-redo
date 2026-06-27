#!/usr/bin/env bash
# GPU-free end-to-end smoke test of the full Docker stack.
# Brings up ChromaDB + all services + the OpenAI stub, seeds sample data,
# then exercises signup -> /medicalTalk and asserts a classification verdict.
set -euo pipefail

cd "$(dirname "$0")/.."

# Point the llm service at the in-cluster stub instead of a host vLLM.
export HRF_LLM_BASE_URL="http://llm-stub:50033/v1"

echo "==> Building and starting the stack (ci profile = OpenAI stub)..."
docker compose --profile ci up -d --build

echo "==> Waiting for backend health..."
for _ in $(seq 1 30); do
  if curl -fsS http://localhost:10000/health >/dev/null 2>&1; then break; fi
  sleep 3
done
curl -fsS http://localhost:10000/health && echo

echo "==> Seeding sample data into SQLite + ChromaDB..."
docker compose run --rm seed

echo "==> Signing up a user..."
TOKEN=$(curl -fsS -X POST http://localhost:10000/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"email":"smoke@test.com","password":"smoke12345"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

echo "==> Running /medicalTalk..."
RESULT=$(curl -fsS -X POST http://localhost:10000/medicalTalk \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"data":"Does vitamin D help bone health?","opinion":"4","backend":"HeReFaNMi LLM"}')

echo "Response: ${RESULT}"
echo "${RESULT}" | python3 -c 'import sys,json; r=json.load(sys.stdin); assert r["label"] in ("Trustworthy","Doubtful","Fake"), r; print("SMOKE OK: label =", r["label"])'

echo "==> Tearing down..."
docker compose --profile ci down
