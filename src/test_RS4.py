
from RS4 import *
import unittest

class TestRS4(unittest.TestCase):

    def test_permutation_rules(self):
        
        sp = SudokuProblem(4)

        for xs in itools.product(*[sp.all_xs for i in range(3)]):
            x,y,z = xs
            if not (x <= y and y <= z):
                continue
            r1 = sp.rb[xs[0]][xs[1]][xs[2]]

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

        for xs in itools.product(*[sp.all_xs for i in range(3)]):
            x,y,z = xs

            if sp.x_to_len[x] == 1 \
                or sp.x_to_len[y] == 1 \
                or sp.x_to_len[z] == 1:
                continue
            
            rule = sp.rb[xs[0]][xs[1]][xs[2]]
            self.assertTrue(rule.consistent \
                and not rule.non_trivial_update \
                and not rule.full_after_update)

def wait():
    wprint("\n    [ENTER] to continue...")

def show_rule_book():
    sp = SudokuProblem(4)

    print("\nWelcome to the " + cG("rule book") + "!")

    print("\nRecall that we are working in the triangle scenario, and",
        "we are trying to show that\n(the support of) a specific",
        "distribution is " + cR("incompatible") + " with a certain",
        "symmetric\nclassical triangle implementation.")
    
    def event_to_str(os) -> str:
        ret = ""
        for i,o in enumerate(os):
            if i > 0:
                ret += ","
            ret += color_outcome(o)
        return ret

    ret = "\nSpecifically, only the events "
    for i in range(NOUT):
        if i > 0:
            ret += " ; "
        ret += event_to_str((i+1,i+1,i+1))
    ret += "\nand "
    for i, os in enumerate(((1,2,3), (1,2,4), (4,3,2))):
        if i > 0:
            ret += " ; "
        if i == 2:
            ret += " ... ; "
        ret += event_to_str(os)
    ret += " are " + cG("allowed.")
    print(ret)

    print("\nThis means that any event of the form",
        event_to_str((1,1,2)),";",event_to_str((1,3,3)),"; ... is",
        cR("inconsistent."))

    wait()

    print("\nInternally, this code assigns to each Alice in the inflation",
        "an outcome possibility\n(psb), i.e., a subset of the outcomes",
        f"{color_outcome(1)},{color_outcome(2)}," \
        + f"{color_outcome(3)},{color_outcome(4)}",
        "that the corresponding Alice may\noutput. For instance:\n")
    for psb in ((0,), (1,2), (1,2,3), (0,1,2,3)):
        ret = " - the psb "
        ret += sp.x_to_long_str[sp.psb_to_x[psb]]
        ret += " corresponds to Alice outputting "
        for i, o in enumerate(psb):
            if i > 0:
                ret += " or "
            ret += color_outcome(o+1)
        print(ret + ";")
    
    print("\nOverall, the 2^4-1 = 15 psb are: ")
    for x in sp.all_xs:
        print(sp.x_to_long_str[x])

    wait()

    print("\nThe inconsistent events can be lifted to give rise to",
        "update rules at the level\nof psb triplets. The simplest such rule",
        "is:")
    sp.wprint_rule((0,0,14),False)

    print("\nThis rule says that if Alice & Bob output",color_outcome(1),
        "then we must update Charlie\nto also be outputting",
        f"{color_outcome(1)}.",
        "(Note that these rules are symmetric under party exchange.)")

    print("\nAnother type of rule is the following:")

    sp.wprint_rule((0,1,COMPLETELY_UNKNOWN),False)

    print("\nThis rule says that once we know that Alice and Bob",
        "output different outcomes,\nCharlie must output something",
        "different (although we are not yet able to say what exactly).")

    wait()

    print(f"\n------ {cG('Now printing the whole rule book.')}",
        "Ctrl+C to",
        "exit, ENTER to move to next rule. -------")
    
    for xs in itools.product(*[sp.all_xs for i in range(3)]):
        sp.wprint_rule(xs)
    
def example_filling():

    solver = SudokuSolver(6,detailed_verb=True,progress_verb=False)

    print("We now define the inflation featuring 6 sources,",
        "and for each i != j, and Alice labeled A(i,j)\nwhose left (right)",
        "input is drawn from source i (j).",
        "This inflation is such that any triplet\nof sources (i,j,k)",
        "with either i < j < k or i > j > k, corresponding to the Alices",
        "A(i,j),\nA(j,k) and A(k,i),",
        "is isomorphic to the original triangle scenario. This lets us",
        "apply\nthe deduction rules for each such triplet.")
    
    wait()
    
    print("\nWe label the sources with grey number for indicative purposes.",
        "For each Alice, we display\nthe current best guess for her",
        "outcome possibilities.",
        "This guess is based on the rule book\nthat is",
        "constantly checking out the cycles (i,j,k) (i < j < k or i > j > k)",
        "\nof the inflation to apply updates.")

    wait()

    print("\nSpecifically, we will:")
    print("  1 - Fill in some initial events that must necessarily arise",
        "based on d-separation,")
    print("  2 - Attempt to complete the initial event to a definite one",
        "that respects the\n",
        "     consistency rules (support compatibility).")

    wait()

    print(cY("\n --- Filling in the initial events... ---\n"))
    solver.set_diag_event(0, 1,2,3)
    solver.set_diag_event(1, 2,3,4)

    print(cY(" --- ... done. --- \n"))

    solver.complete_grid()
    
def big_infeasible_grid():

    solver = SudokuSolver(
        order=24, 
        detailed_verb=False,
        progress_verb=True)

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

    solver.complete_grid()

def small_infeasible_grid():

    solver = SudokuSolver(
        order=18,
        detailed_verb=False,
        progress_verb=True)

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

    solver.complete_grid()

if __name__=="__main__":
    target = sys.argv[1]

    if target == "infeasible_grid":
        small_infeasible_grid()
    elif target == "rules":
        show_rule_book()
    elif target == "example_filling":
        example_filling()