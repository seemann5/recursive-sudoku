
from RS4 import *
import unittest, sys

class TestRS4(unittest.TestCase):

    def test_permutation_rules(self):
        
        sp = SudokuProblem(4)

        for x,y,z in itertools.product(*[sp.all_xs for i in range(3)]):
            if not (x <= y and y <= z):
                continue

            xs = [x,y,z]
            r1 = sp.rb[x][y][z]
            for perm in itertools.permutations((0,1,2)):
                r2 = sp.rb[xs[perm[0]]][xs[perm[1]]][xs[perm[2]]]

                self.assertEqual(r1.consistent,r2.consistent)
                self.assertEqual(r1.non_trivial_update,r2.non_trivial_update)
                self.assertEqual(r1.full_after_update,r2.full_after_update)

                no_update = r1.update_rule is None \
                    and r2.update_rule is None
                
                if not no_update:
                    for i in range(3):
                        self.assertEqual(r1.update_rule[perm[i]],
                            r2.update_rule[i])
                        self.assertEqual(r1.non_trivial_pos[perm[i]],
                            r2.non_trivial_pos[i])
                else:
                    self.assertTrue(no_update)
                    self.assertTrue(r1.non_trivial_pos is None \
                        and r2.non_trivial_pos is None)

    def test_assumption_about_rules(self):
        
        sp = SudokuProblem(4)

        for x,y,z in itertools.product(*[sp.all_xs for i in range(3)]):

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

def main():
    
    sp = SudokuProblem(4)
    



if __name__=="__main__":
    target = sys.argv[1]

    if target == "default":
        main()
    elif target == "rules":
        show_rule_book()