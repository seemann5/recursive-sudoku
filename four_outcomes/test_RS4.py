
from RS4 import *
import unittest, sys

class TestRS4(unittest.TestCase):

    def test_permutation_rules(self):
        
        sp = SudokuProblem(4)

        for xs in itools.product(*[sp.all_xs for i in range(3)]):
            x,y,z = xs
            if not (x <= y and y <= z):
                continue
            r1 = sp.rb[xs]

            for perm in itools.permutations((0,1,2)):
                r2 = sp.rb[(xs[perm[0]],xs[perm[1]],xs[perm[2]])]

                self.assertEqual(r1.consistent,r2.consistent)
                self.assertEqual(r1.non_trivial_update,r2.non_trivial_update)
                self.assertEqual(r1.full_after_update,r2.full_after_update)

                no_update = r1.new_xs is None \
                    and r2.new_xs is None
                
                if not no_update:
                    for i in range(3):
                        self.assertEqual(r1.new_xs[perm[i]],
                            r2.new_xs[i])
                        self.assertEqual(r1.non_trivial_pos[perm[i]],
                            r2.non_trivial_pos[i])
                else:
                    self.assertTrue(no_update)
                    self.assertTrue(r1.non_trivial_pos is None \
                        and r2.non_trivial_pos is None)

    def test_assumption_about_rules(self):
        
        sp = SudokuProblem(4)

        for xs in itools.product(*[sp.all_xs for i in range(3)]):
            x,y,z = xs

            if sp.x_to_len[x] == 1 \
                or sp.x_to_len[y] == 1 \
                or sp.x_to_len[z] == 1:
                continue
            
            rule = sp.rb[xs]
            self.assertTrue(rule.consistent \
                and not rule.non_trivial_update \
                and not rule.full_after_update)


                        
def show_rule_book():
    sp = SudokuProblem(4)
    sp.print_rule_book()
    
def show_party_dH():
    sp = SudokuProblem(4)
    sp.print_party_dH()

def grid_state_ex():
    
    sp = SudokuProblem(6)

    grid = GridState(sp)

    updates = [
        [0,1,1],
        [1,2,1],
        [3,4,2],
        [4,5,3],
        [5,3,4],
        [0,3,1]
    ]

    for update in updates:
        grid.user_set_outcome(update[0],update[1],update[2])
    
    print("Initial grid:")    
    wprint(grid.to_short_str())

def small_example():

    solver = SudokuSolver(6,branching_attitude="",
    detailed_verb=True,progress_verb=True)

    solver.user_set_outcome(0,1, 1)
    solver.user_set_outcome(1,2, 1)
    solver.user_set_outcome(3,4, 2)
    solver.user_set_outcome(4,5, 3)
    solver.user_set_outcome(5,3, 4)

    print("Status: ",solver.complete_grid())

    swaps = [
        (4,5),
        (0,2),
        (2,4),
        (0,1),
        (1,2),
        (2,3),
    ]
    for s1,s2 in swaps:
        solver.solution_grid.swap_sources(s1,s2)
    print(solver.solution_grid.to_short_str())
    
def old_main_bis():

    solver = SudokuSolver(12)
    
    for d in range(4):
        o = d+1
        s = 3*d
        solver.user_set_outcome(  s, s+1, o)
        solver.user_set_outcome(s+1, s+2, o)
    
    assert solver.complete_grid()

    swaps = [
        (2,7),
        (1,2)
    ]
    solver.solution_grid.swap_source_list(swaps)

    diag_info = [
        (0,6,2),
        (7,11,4)
    ]
    solver.solution_grid.extend_to_strat(diag_info)
    
def oooooooooold_main():

    solver = SudokuSolver(15)

    for d in range(4):
        o = d+1
        s = 3*d
        solver.user_set_outcome(  s, s+1, o)
        solver.user_set_outcome(s+1, s+2, o)
    
    solver.user_set_outcome(12, 13, 1)
    solver.user_set_outcome(13, 14, 2)
    solver.user_set_outcome(14, 12, 4)

    assert solver.complete_grid()

    swaps = [
        (11,13),
        (7,12),
        (8,10),
        (8,9),
        (0,12),
        (1,13),
        (2,14),
        (9,13),
        (12,14),
        (13,14)
    ]

    solver.solution_grid.swap_source_list(swaps)

    # No matter the value of x & y, there's no solution!
    x = 4
    y = 4
    diag_info = [
        (0,2,1),
        (3,9,2),
        (10,12,4),
        (13,13,x),
        (14,14,y)
    ]
    solver.solution_grid.extend_to_strat(diag_info)

def main_v18():

    solver = SudokuSolver(18)

    for d in range(4):
        o = d+1
        s = 3*d
        solver.user_set_outcome(  s, s+1, o)
        solver.user_set_outcome(s+1, s+2, o)
    
    solver.user_set_outcome(12, 13, 1)
    solver.user_set_outcome(13, 14, 2)
    solver.user_set_outcome(14, 12, 4)

    solver.user_set_outcome(15,16, 2)
    solver.user_set_outcome(16,17, 3)
    solver.user_set_outcome(17,15, 4)

    assert solver.complete_grid()

    swaps = [
        (1,2),
        (7,9),
        (8,12),
        (9,15),
        (10,16),
        (1,16),
        (11,16),
        (1,0),
    ]

    solver.solution_grid.swap_source_list(swaps)

    # No value of x & y workds
    x = 4
    y = 4
    diag_info = [
        (0,0,x),
        (1,1,y),
        (2,10,2),
        (11,13,4),
        (14,17,1)
    ]
    solver.solution_grid.extend_to_strat(diag_info)

    print(f"x,y = {x,y}")

def infeasible_main():

    solver = SudokuSolver(
        order=24, 
        branching_attitude="Entropy",
        detailed_verb=False,
        progress_verb=True)

    # for d in range(4):
    #     o = d+1
    #     s = 3*d
    #     solver.user_set_outcome(  s, s+1, o)
    #     solver.user_set_outcome(s+1, s+2, o)
    
    # solver.user_set_outcome(12, 13, 1)
    # solver.user_set_outcome(13, 14, 2)
    # solver.user_set_outcome(14, 12, 3)

    # solver.user_set_outcome(15,16, 1)
    # solver.user_set_outcome(16,17, 2)
    # solver.user_set_outcome(17,15, 4)

    # solver.user_set_outcome(18,19, 1)
    # solver.user_set_outcome(19,20, 3)
    # solver.user_set_outcome(20,18, 4)

    # solver.user_set_outcome(21,22, 2)
    # solver.user_set_outcome(22,23, 3)
    # solver.user_set_outcome(23,21, 4)

    events = [
        (1,2,3),
        (1,2,4),
        (1,3,4),
        (2,3,4),
        (1,1,1),
        (2,2,2),
        (3,3,3),
        (4,4,4)
    ]
    for d, (a,b,c) in enumerate(events):
        solver.set_diag_event(d, a,b,c)

    status = solver.complete_grid()

    print(f"Status: {status}")

    swaps = [
    ]

    #solver.solution_grid.swap_source_list(swaps)

    # No value of x & y workds
    x = 4
    y = 4
    diag_info = [
    ]
    #solver.solution_grid.extend_to_strat(diag_info)

    print(f"x,y = {x,y}")

def main():

    solver = SudokuSolver(
        order=18,
        branching_attitude="Entropy",
        detailed_verb=True,
        progress_verb=True)

    # for d in range(4):
    #     o = d+1
    #     s = 3*d
    #     solver.user_set_outcome(  s, s+1, o)
    #     solver.user_set_outcome(s+1, s+2, o)
    
    # solver.user_set_outcome(12, 13, 1)
    # solver.user_set_outcome(13, 14, 2)
    # solver.user_set_outcome(14, 12, 3)

    # solver.user_set_outcome(15,16, 1)
    # solver.user_set_outcome(16,17, 2)
    # solver.user_set_outcome(17,15, 4)

    # solver.user_set_outcome(18,19, 1)
    # solver.user_set_outcome(19,20, 3)
    # solver.user_set_outcome(20,18, 4)

    # solver.user_set_outcome(21,22, 2)
    # solver.user_set_outcome(22,23, 3)
    # solver.user_set_outcome(23,21, 4)

    events = [
        (1,2,3),
        (1,2,4),
        (1,3,4),
        (2,3,4),
        (1,1,1),
        (2,2,2),
    ]
    for d, (a,b,c) in enumerate(events):
        solver.set_diag_event(d, a,b,c)

    status = solver.complete_grid()

    print(f"Status: {status}")

    swaps = [
    ]

    #solver.solution_grid.swap_source_list(swaps)

    # No value of x & y workds
    x = 4
    y = 4
    diag_info = [
    ]
    #solver.solution_grid.extend_to_strat(diag_info)

    print(f"x,y = {x,y}")

def smaller_grids_main_not_sure():
    """Smaller infeasible grids?"""

    solver = SudokuSolver(
        order=15,
        branching_attitude="Entropy",
        detailed_verb=False,
        progress_verb=True
    )

    events = [
        (1,2,3),
        (3,2,1),
        (1,2,4),
        (2,3,4),
        (1,3,4),
        #(4,3,1),
        #(1,1,1)
    ]

    for d, (a,b,c) in enumerate(events):
        solver.set_diag_event(d, a,b,c)

    status = solver.complete_grid()
    
    print(f"Status: {status}")


if __name__=="__main__":
    #target = sys.argv[1]

    if False:
        import cProfile
        cProfile.run("infeasible_main()", "stats_15")
        import pstats
        p = pstats.Stats("stats_15")
        p.strip_dirs().sort_stats("tottime").print_stats(10)
    else:
        #infeasible_main()
        # small_example()
        main()


    if False:
        pass
        if target == "default":
            main()
        elif target == "rules":
            show_rule_book()
        elif target == "entropy":
            show_party_dH()