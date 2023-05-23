.PHONY: start
start:
	uvicorn main:app --reload --port 9000

test:
	python test.py

chat:
	python test.py chat

ingest:
	python ingest.py

.PHONY: format
format:
	black .
	isort .

install:
	pip install -r requirements.txt
	