from recursive_sudoku import *
import unittest
import sys

class TestDepth(unittest.TestCase):

    def test_12_by_12(self):
        print("")
        update_order = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11)]
        sudoku_solver = SudokuSolver(12)
        sudoku_solver.set_filling_steps_starting_with(update_order)
        self.assertEqual(sudoku_solver.get_resulting_depth(), 3)

    def test_4_by_4(self):
        print("")
        update_order = [(1,2),(3,0)]
        sudoku_solver = SudokuSolver(4)
        sudoku_solver.set_filling_steps_starting_with(update_order)
        self.assertEqual(sudoku_solver.get_resulting_depth(), 2)
        self.assertEqual(
            len(sudoku_solver.final_filling_steps[-1].check_rules),
            1)

def case12_algo():
    update_order = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11)]
    sudoku_solver = SudokuSolver(12)
    # Use 3 to speed up, assuming the above test succeeds
    sudoku_solver.set_filling_steps_starting_with(update_order, 
        satisfying_depth=3, verb=True)
    
def case12_solution():
    update_order = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11)]
    sudoku_solver = SudokuSolver(12)
    # Use 3 to speed up, assuming the above test succeeds
    sudoku_solver.set_filling_steps_starting_with(update_order, 
        satisfying_depth=3)
    
    outcomes = [1,1, 2,2, 3,3, 1,2]
    sudoku_solver.set_initial_outcomes_and_complete(outcomes, verb=True)

def case4_algo():
    update_order = [(0,1),(2,3)]
    sudoku_solver= SudokuSolver(4)
    sudoku_solver.set_filling_steps_starting_with(update_order, verb=True)

def case4_solution():
    update_order = [(0,1),(2,3)]
    sudoku_solver= SudokuSolver(4)
    sudoku_solver.set_filling_steps_starting_with(update_order)
    outcomes = [1,1]
    sudoku_solver.set_initial_outcomes_and_complete(outcomes, verb=True)

if __name__=="__main__":
    target = sys.argv[1]

    if target == "example_algo":
        case4_algo()
    elif target == "4x4":
        case4_solution()
    elif target == "12x12":
        case12_solution()
