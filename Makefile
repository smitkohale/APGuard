# Local workflow. Requires DATABASE_URL in .env (see .env.example).
PSQL_URL = $(shell python -c 'import os,dotenv;dotenv.load_dotenv();print(os.environ.get("DATABASE_URL","").replace("+psycopg2",""))')

.PHONY: install data db pipeline reviews evaluate test test-db lint api dashboard benchmark all

install:
	pip install -r requirements-dev.txt

data:
	python data/generate_data.py

db:
	psql "$(PSQL_URL)" -q -f db/schema.sql
	python db/load_data.py

pipeline:
	python scripts/run_pipeline.py

reviews:
	python scripts/simulate_reviews.py

evaluate:
	python scripts/evaluate.py

test:
	pytest -m "not db" -q

test-db:
	pytest -m db -v

lint:
	ruff check .

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py --server.port 8501

benchmark:
	python scripts/benchmark.py --legacy

all: data db pipeline reviews evaluate
