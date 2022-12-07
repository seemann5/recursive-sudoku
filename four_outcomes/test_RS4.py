
from RS4 import *
import unittest, sys

class TestRS4(unittest.TestCase):

    def test_permutation_rules(self):
        
        sp = SudokuProblem(4)

        for x,y,z in itools.product(*[sp.all_xs for i in range(3)]):
            if not (x <= y and y <= z):
                continue

            xs = [x,y,z]
            r1 = sp.rb[x][y][z]
            for perm in itools.permutations((0,1,2)):
                r2 = sp.rb[xs[perm[0]]][xs[perm[1]]][xs[perm[2]]]

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

        for x,y,z in itools.product(*[sp.all_xs for i in range(3)]):

            if sp.x_to_len[x] == 1 \
                or sp.x_to_len[y] == 1 \
                or sp.x_to_len[z] == 1:
                continue
            
            rule = sp.rb[x][y][z]
            self.assertTrue(rule.consistent \
                and not rule.non_trivial_update \
                and not rule.full_after_update)


                        
def show_rule_book():
    sp = SudokuProblem(4)
    sp.print_rule_book()
    
def show_party_dH():
    sp = SudokuProblem(4)
    sp.print_party_dH()

def old_main():
    
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

def main():

    solver = SudokuSolver(6)

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
    

    
    



if __name__=="__main__":
    target = sys.argv[1]

    if target == "default":
        main()
    elif target == "rules":
        show_rule_book()
    elif target == "entropy":
        show_party_dH()