#!/usr/bin/env bash
# APGuard end-to-end setup without Docker. Needs PostgreSQL and a .env file with
# DATABASE_URL (see .env.example). For the zero-setup route: docker compose up --build
set -euo pipefail

if [ ! -f .env ]; then
  echo "Missing .env: copy .env.example to .env and set DATABASE_URL." >&2
  exit 1
fi

echo "== 1/7 Installing dependencies";   pip install -q -r requirements-dev.txt
PSQL_URL=$(python -c 'import os,dotenv;dotenv.load_dotenv();print(os.environ["DATABASE_URL"].replace("+psycopg2",""))')
echo "== 2/7 Generating synthetic data"; python data/generate_data.py
echo "== 3/7 Applying schema";           psql "$PSQL_URL" -q -f db/schema.sql
echo "== 4/7 Loading data";              python db/load_data.py
echo "== 5/7 Running detection";         python scripts/run_pipeline.py
echo "== 6/7 Simulating reviews";        python scripts/simulate_reviews.py
echo "== 7/7 Evaluating and testing";    python scripts/evaluate.py && pytest -m "not db" -q

echo
echo "Done. Start the app with:"
echo "  uvicorn api.main:app --reload --port 8000          # API, docs at /docs"
echo "  streamlit run dashboard/app.py --server.port 8501  # dashboard"
