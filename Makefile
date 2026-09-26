.PHONY: install dev ingest ask chat api ui test lint docker

install:
	pip install -e ".[ui]"

dev:
	pip install -e ".[ui,dev]"

ingest:
	hybrid-rag ingest

ask:
	hybrid-rag ask "Milk recall: what must each affected store manager do?" --role manager

chat:
	hybrid-rag chat --role manager

api:
	uvicorn hybrid_rag.api:app --reload --port 8000

ui:
	streamlit run app/streamlit_app.py

test:
	LLM_PROVIDER=fake pytest -q

lint:
	ruff check src tests app

docker:
	docker build -t enterprise-hybrid-rag . && docker run --rm -p 8000:8000 --env-file .env enterprise-hybrid-rag
