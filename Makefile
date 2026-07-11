.PHONY: run test lint check

run:
	python3 server.py

test:
	python3 -m unittest discover -s tests -v

lint:
	python3 -m compileall -q ai_engine.py server.py tests

check: lint test
