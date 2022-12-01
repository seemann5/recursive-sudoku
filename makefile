
MAIN_FILE=test_sudoku.py
PY=python3

test:
	$(PY) -m unittest -v $(MAIN_FILE)

example_algo:
	$(PY) $(MAIN_FILE) example_algo

small_example:
	$(PY) $(MAIN_FILE) 4x4

big_example:
	$(PY) $(MAIN_FILE) 12x12