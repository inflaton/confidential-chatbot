.PHONY: start
start:
	# uvicorn main:app --reload --port 9000
	python -m lcserve deploy local main.py

test:
	python test.py

chat:
	python test.py chat

ingest:
	python ingest.py

mlock:
	sudo prlimit --memlock=15413752832:15413752832 --pid $$$$
	prlimit

.PHONY: format
format:
	black .
	isort .

install:
	pip install -r requirements.txt
	