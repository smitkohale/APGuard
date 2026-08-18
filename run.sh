#!/usr/bin/env bash
# APGuard end-to-end setup. Requires PostgreSQL running locally with a
# 'postgres' user/password 'apguard' (edit db/load_data.py + api/main.py if different).
set -e

echo "== 1/6 Installing dependencies =="
pip install -r requirements.txt --break-system-packages

echo "== 2/6 Applying schema =="
PGPASSWORD=apguard psql -h localhost -U postgres -d apguard -f db/schema.sql

echo "== 3/6 Generating synthetic data =="
python3 data/generate_data.py

echo "== 4/6 Loading into Postgres =="
python3 db/load_data.py

echo "== 5/6 Running rules + evaluating against ground truth =="
python3 scripts/run_and_evaluate.py

echo "== 6/6 Running tests =="
python3 -m pytest tests/ -v

echo ""
echo "Setup complete. Start the app with:"
echo "  uvicorn api.main:app --reload --port 8000        # API"
echo "  streamlit run dashboard/app.py --server.port 8501 # Dashboard"
