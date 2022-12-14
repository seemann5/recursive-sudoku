
src=src/
MAIN_FILE=$(src)test_RS4.py
PY=python3

infeasible_grid:
	$(PY) $(MAIN_FILE) $@

test:
	$(PY) -m unittest -v $(MAIN_FILE)

rules:
	$(PY) $(MAIN_FILE) $@

example_filling:
	$(PY) $(MAIN_FILE) $@

