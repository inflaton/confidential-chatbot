.PHONY: start
start:
	python -m lcserve deploy local main.py

test:
	python test.py

chat:
	python test.py chat

ingest:
	python ingest.py

mlock:
	@echo "To set mlock, please run: sudo prlimit --memlock=15413752832:15413752832 --pid $$$$"
	prlimit --memlock

.PHONY: format
format:
	black .
	isort .

install:
	pip install -r requirements.txt
	