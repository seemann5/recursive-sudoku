from recursive_sudoku import *
import unittest
import sys

class TestThreeOutcomeSudoku(unittest.TestCase):

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

    def test_4_by_4_completable(self):
        print("")
        update_order = [(1,2),(3,0)]
        sudoku_solver = SudokuSolver(4)
        sudoku_solver.set_filling_steps_starting_with(update_order)
        outcome_choices = [ [1,1], [1,2] ]
        for outcomes in outcome_choices:
            c = sudoku_solver.set_initial_outcomes_and_complete(outcomes)
            self.assertTrue(c)
        update_order = [(0,1),(1,2)]
        sudoku_solver = SudokuSolver(4)
        sudoku_solver.set_filling_steps_starting_with(update_order)
        for outcomes in outcome_choices:
            c = sudoku_solver.set_initial_outcomes_and_complete(outcomes)
            self.assertTrue(c)

    def test_case12_completable(self):
        print("")
        update_order = [
            (0,1),
            (1,2),
            (3,4),
            (4,5),
            (6,7),
            (7,8),
            (9,10),
            (10,11),
            (12,13),
            (13,14)
            ]
        
        sudoku_solver = SudokuSolver(15)
        # Use 3 to speed up, assuming the above test succeeds
        sudoku_solver.set_filling_steps_starting_with(update_order, 
            satisfying_depth=5)
        
        possible_outcomes = [ [1], [1,2] ]
        for i in range(3, 10+1):
            possible_outcomes.append([1,2,3])
        
        for (a1,a2,a3,a4,a5,a6,a7,a8,a9,a10) \
                in itertools.product(*possible_outcomes):
            self.assertTrue(sudoku_solver.set_initial_outcomes_and_complete(
                    [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
                ))
            
def case12_algo():
    update_order = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11)]
    sudoku_solver = SudokuSolver(12)
    # Use 3 to speed up, assuming the above test succeeds
    sudoku_solver.set_filling_steps_starting_with(update_order, 
        satisfying_depth=3, verb=True)

def case4_algo():
    update_order = [(0,1),(2,3)]
    sudoku_solver= SudokuSolver(4)
    sudoku_solver.set_filling_steps_starting_with(update_order, verb=True)

def case4_solution():
    update_order = [(1,2),(3,0)]
    sudoku_solver= SudokuSolver(4)
    sudoku_solver.set_filling_steps_starting_with(update_order)
    outcomes = [1,2]
    sudoku_solver.set_initial_outcomes_and_complete(outcomes, verb=True)

def case4_solution_alt():
    update_order = [(0,1),(1,2)]
    sudoku_solver = SudokuSolver(4)
    sudoku_solver.set_filling_steps_starting_with(update_order)
    outcomes = [1,2]
    sudoku_solver.set_initial_outcomes_and_complete(outcomes, verb=True)

def biggest():
    length = 9
    update_order = []
    for i in range(length):
        s = 3*i
        update_order.append((s,s+1))
        update_order.append((s+1,s+2))
    solver = SudokuSolver(3*length)
    solver.set_filling_steps_starting_with(update_order,
        satisfying_depth=8
        )
    
    outcomes = [
        1,1,
        2,2,
        3,3,
        1,2,
        1,3,
        2,1,
        2,3,
        3,1,
        3,2
    ]

    print("\nWe now fill in all of the 3 (111 type) + 6 (123 type) events",
        "on the diagonal.")

    ok, grid = solver.set_initial_outcomes_and_complete(outcomes,
        show_initial_grid=True)

    assert ok
    print("\nThere exists indeed (at least one) completion into a",
        "consistent grid:")
    print(grid.dense_str())

    swaps = [
        (4,6),
        (5,9),
        (6,10),
        (7,12),
        (8,13),
        (9,15),
        (10,18),
        (11,20),
        (12,21),
        (13,24),
        (14,26),
        (15,24)
    ]

    wprint("Now, let's swap the sources around cleverly to re-arrange this:")

    for s1,s2 in swaps:
        grid.swap_sources(s1,s2)
    print(grid.dense_str())
    
    wprint("This suggests a simple strategy!")

    print("\nSuppose Alice, Bob and Charlie all use the two-input strategy\n")

    strat = [
        [1,2,3],
        [3,1,2],
        [2,3,1]
    ]

    sp = solver.sp

    ret = ""
    for i in range(3):
        for j in range(3):
            ret += sp.color_outcome(strat[i][j]) + " "
        ret += "\n"
    print(ret)

    print("then we do get the target outcome distribution:\n")

    outcomes = (0,1,2)
    triple = [outcomes,outcomes,outcomes]
    for a,b,c in itertools.product(*triple):
        p = 0
        for alpha,beta,gamma in itertools.product(*triple):
            if strat[alpha][beta]-1 == a \
                and strat[beta][gamma]-1 == b \
                and strat[gamma][alpha]-1 == c:
                p += 1
        if p != 0:
            print(
                f"p({sp.color_outcome(a+1)}," \
                + f"{sp.color_outcome(b+1)},{sp.color_outcome(c+1)}" \
                + f") = {p}/27")

    print("\n ... which is then really same-strategy compatible. \n")
                
        


if __name__=="__main__":
    target = sys.argv[1]

    if target == "example_algo":
        case4_algo()
    elif target == "4x4":
        case4_solution()
    elif target == "4x4_alt":
        case4_solution_alt()
    elif target == "biggest":
        biggest()