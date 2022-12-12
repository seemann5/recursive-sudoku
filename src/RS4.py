
import itertools as itools
import time, sys, copy
from typing import List, Tuple
import numpy as np
from coloring import *

# --------------------------------------------
# ------------ Utilities ---------------------
# --------------------------------------------

def wprint(arg):
    input(arg)

INT = np.uint8

# --------------------------------------------
# ------------ SudokuProblem Class -----------
# --------------------------------------------

NOUT=4
COMPLETELY_UNKNOWN=14

class SudokuProblem:
    """Stores useful information about the inflation graph,
    the possible events and associated update rules"""

    def __init__(self, order):
        # The size of the grid/inflation (i.e., number of sources)
        self.order = order
        # The Alices & Cycles (copies of the original triangle) part
        self.n_alices = order*(order-1)
        self.n_cycles = int(2 * order*(order-1)*(order-2) / (3*2))
        # k denotes the hash of an Alice
        self.k_to_str = ["" for k in range(self.n_alices)]
        # l denotes the hash of a cycle
        self.l_to_str = ["" for l in range(self.n_cycles)]
        self.ij_to_k = dict()
        # The Alices (k's) that the cycle (l) contains
        self.l_to_ks = [[] for l in range(self.n_cycles)]
        # The cycles (l's) that Alice (k) partipates to
        self.k_to_ls = [[] for k in range(self.n_alices)]
        # The two other k's in l apart from k
        self.kl_to_other_ks = [
            [() for l in range(self.n_cycles)] 
            for k in range(self.n_alices)]

        self.create_kl_links()

        # x denotes a hash of a psb (possibility) such as (0,1,3) for
        # outcomes 0, 1 or 3
        # The empty psb is excluded.
        self.all_xs = [INT(x) for x in range(0,2**NOUT-1)]
        self.x_to_long_str = ["" for x in self.all_xs]
        self.x_to_short_str = [""for x in self.all_xs]
        self.x_to_len = [0 for x in self.all_xs]
        self.psb_to_x = dict()
        self.x_to_psb = [() for x in self.all_xs]
        
        self.create_x_stuff()

        # The rule book, which states how to update a triple of x's
        self.rb = [
            [
                [None for _ in self.all_xs] for _ in self.all_xs
            ]
            for _ in self.all_xs]
        
        self.create_rule_book()
        
    def create_kl_links(self):
        # Alices
        alices = [(i,j) 
            for i in range(self.order) 
                for j in range(self.order)
                if i != j
        ]

        # self.k_to_str
        # self.ij_to_k
        for k, a in enumerate(alices):
            i,j = a
            self.k_to_str[k] = f"A({i},{j})" 
            self.ij_to_k[(i,j)] = k

        # Cycles
        cycles = list(
            itools.chain.from_iterable(
                ((x,y,z),(z,y,x))
                    for x in range(self.order) 
                        for y in range(self.order) if y > x 
                            for z in range(self.order) if z > y
            )
        )
        
        # self.l_to_str
        for l, c in enumerate(cycles):
            self.l_to_str[l] = str(c)

        # helper function
        def l_contains_k(l: int, k: int) -> bool:
            i,j = alices[k]
            x,y,z = cycles[l]
            return (x == i and y == j) \
                or (y == i and z == j) \
                or (z == i and x == j)
        # Relations between k & l
        self.k_to_ls = [
                [l for l in range(self.n_cycles) if l_contains_k(l, k)]
                for k in range(self.n_alices)
            ]
        self.l_to_ks = [
                [
                    self.ij_to_k[(cycle[0],cycle[1])],
                    self.ij_to_k[(cycle[1],cycle[2])],
                    self.ij_to_k[(cycle[2],cycle[0])]
                ]
                for cycle in cycles
            ]

        # k,l to other k's
        for l in range(self.n_cycles):
            ks = self.l_to_ks[l]
            for i in range(3):
                self.kl_to_other_ks[ks[i]][l] = (ks[(i+1)%3], ks[(i+2)%3])

    # X STUFF ----------------

    def create_x_stuff(self):
        # A psb is defined as a tuple (0,2,3) etc
        for cs in itools.product((0,1),(0,1),(0,1),(0,1)):
            if sum(cs) == 0:
                # Not a valid psb
                continue

            psb = []
            for i in range(NOUT):
                if cs[3-i] == 1:
                    psb.append(i)
            psb = tuple(psb)

            x = -1
            for o in psb:
                x += 2**o
            self.x_to_psb[INT(x)] = psb
            self.psb_to_x[psb] = INT(x)
            self.x_to_len[x] = len(psb)

            # Long string
            x_str = ""
            if sum(cs) == 1:
                for i in range(NOUT):
                    if cs[3-i] == 1:
                        x_str = f"  {color_outcome(i+1)} "
            elif sum(cs) == 2:
                x_str = "("
                for i in range(NOUT):
                    if cs[3-i] == 1:
                        x_str += color_outcome(i+1)
                x_str += ")"
            elif sum(cs) == 3:
                for i in range(NOUT):
                    if cs[3-i] == 0:
                        x_str = f" !{color_outcome(i+1)} "
            else:
                x_str = "  . "
            self.x_to_long_str[x] = x_str

            # Short string
            if sum(cs) == 1:
                for i in range(NOUT):
                    if cs[3-i] == 1:
                        self.x_to_short_str[x] = color_outcome(i+1)
            elif sum(cs) == 4:
                self.x_to_short_str[x] = " "
            else:
                self.x_to_short_str[x] = "."

    # RULE BOOK STUFF ---------------------------

    def is_valid_event(self, o1: int, o2: int, o3: int) -> bool:
        """Gives True for an outcome triplet that is in the target support"""
        return (o1==o2 and o2==o3) or (o1!=o2 and o2!=o3 and o3!=o1)

    def simplify_xs(self, xs: Tuple[INT,INT,INT]) -> Tuple[INT,INT,INT]:
        """Returns (the hash of) the updated x's, stating that e.g.
        1,1,. goes necessarily to 1,1,1."""
            
        # Now check for consistency & simplifications at the same time.
        triple_psb = [self.x_to_psb[x] for x in xs]
        
        ps_to_check = {0,1,2}
        was_inconsistent = False

        while len(ps_to_check) > 0:
            p = ps_to_check.pop()
            keep_outcomes = []

            psb1, psb2, psb3 = (triple_psb[(p+i)%3] for i in range(3))

            for o1 in psb1:
                for o2, o3 in itools.product(psb2, psb3):
                    if self.is_valid_event(o1,o2,o3):
                        keep_outcomes.append(o1)
                        break
            
            if len(keep_outcomes) == len(psb1):
                # We haven't removed anything. Nothing to do!
                pass
            elif len(keep_outcomes) == 0:
                # There is nothing to keep... Inconsistency:
                was_inconsistent = True
                return was_inconsistent, None
            else:
                # Making a consistent update. Must re-check the other p's...
                for other_p in range(3):
                    if other_p != p:
                        ps_to_check.add(other_p)
                # ... AND update:
                triple_psb[p] = tuple(keep_outcomes)
        
        new_xs = tuple(self.psb_to_x[psb] for psb in triple_psb)

        return was_inconsistent, new_xs

    def create_rule_book(self):
        """Iterates over all patterns of triplets of psb's, and
        generate the corresponding update rules"""
        
        for xs in itools.product(* [self.all_xs] * 3):

            # So now we're trying to see how to update.
            was_inconsistent, new_xs = self.simplify_xs(xs)
            
            if was_inconsistent:
                self.rb[xs[0]][xs[1]][xs[2]] = Rule(False)
                continue

            # We're done, and we haven't reached an inconsistency. 
            # Hence, we have some sort of consistent update.

            # Check for non-trivial:
            non_trivial_pos = tuple(
                self.x_to_len[new_x] < self.x_to_len[x]
                for x, new_x in zip(xs, new_xs)
            )
            non_trivial_update = True in non_trivial_pos

            # Check for full:
            full_after_update = True
            for x in new_xs:
                if self.x_to_len[x] != 1:
                    full_after_update = False
                    break
            
            # Append
            self.rb[xs[0]][xs[1]][xs[2]] = Rule(True,
                                    non_trivial_update,
                                    full_after_update,
                                    new_xs,
                                    non_trivial_pos)

    def __str__(self) -> str:
        return cG("\n----- Inflation: ") \
                + f"{self.order} sources, " \
                + f"{self.n_alices} Alices, " \
                + f"{self.n_cycles} triangle-isomorphic subgraphs " \
                + cG("-----")

    def wprint_rule(self, xs: Tuple[INT,INT,INT],wait=True):
        x1,x2,x3 = xs
        if not (x1 <= x2 and x2 <= x3):
            return

        #Test 

        rule = self.rb[x1][x2][x3]

        ret = "\n Pattern "
        offset = " " * len(" Pattern ")

        for x in xs:
            ret += self.x_to_long_str[x] + " "

        if not rule.consistent:
            ret += "is inconsistent"
        else:
            if rule.non_trivial_update:
                ret += "can be updated to\n" + offset
                for p in range(3):
                    ret += self.x_to_long_str[rule.new_xs[p]] + " "
            if rule.full_after_update:
                if rule.non_trivial_update:
                    ret += "and "
                ret += "is full"
                if rule.non_trivial_update:
                    ret += " after update"
            else:
                if not rule.non_trivial_update:
                    ret += "cannot be simplified yet"

        if wait:
            wprint(ret)
        else:
            print(ret)

# -----------------------------------------
# -------------- Rule Class ---------------
# -----------------------------------------

class Rule:
    """An update rule, associated to a triple of x's (psb's)."""

    def __init__(self,
                consistent: bool,
                non_trivial_update: bool = None,
                full_after_update: bool = None,
                new_xs: Tuple[INT,INT,INT] = None,
                non_trivial_pos: Tuple[bool,bool,bool] = None
                ):
        self.consistent = consistent
        # The following two are independent but assume that consistent==True
        self.non_trivial_update = non_trivial_update
        self.full_after_update = full_after_update
        # This will be a tuple of the form (x1, x2, x3)
        self.new_xs = new_xs
        # A tuple of the form (True, False, True) if first & third values
        # need to be updated
        self.non_trivial_pos = non_trivial_pos

# -----------------------------------------
# ------------ GridState Class ------------
# -----------------------------------------

class GridState:
    """Stores a possibilistic filling of a grid/inflation"""

    def __init__(self, 
            sp: SudokuProblem, 
            detailed_verb: bool,
            depth_offset: int):
        self.sp = sp
        self.detailed_verb = detailed_verb
        self.depth_offset = depth_offset

        # A k-indexed list of values to be understood as "x"'s 
        # (see SudokuProblem) - essentially knowledge state about an outcome
        self.xs = np.full((sp.n_alices), COMPLETELY_UNKNOWN, dtype=np.uint8)
        # This one will be the branch-width optimized k.
        self.k_to_update = None

        self.perturbed_ks = set()

        # self.ltrust[l] = True if the cycle is full & has been checked
        self.ltrust = np.full((sp.n_cycles), False, dtype=np.bool8)

    def user_set_outcome(self, i, j, outcome):
        """User supplies math-indexed outcome"""
        k = self.sp.ij_to_k[(i,j)]
        x = self.xs[k]
        assert (outcome-1) in self.sp.x_to_psb[x]

        # Will need to update this guy anyway. Don't need to put the statement
        # in the internal set_outcome because this one will only ever
        # be called on a "fresh" GridState.
        self.k_to_update = None

        assert self.set_outcome(k, outcome-1)

    def set_outcome(self, k, outcome) -> bool:
        """Returns true if consistent. Assume CS-indexed outcome"""

        self.xs[k] = self.sp.psb_to_x[(outcome,)]

        if self.detailed_verb:
            print(f"setting {self.sp.k_to_str[k]}",
                f"= {color_outcome(outcome+1)}")
            wprint(self.to_long_str())

        self.perturbed_ks.add(k)

        return self.simplify_from_perturbed_ks()

    def is_full(self) -> bool:
        """Returns True is all the outcomes are specified throughout the grid,
        and without inconsistencies"""
        return self.ltrust.all()

    def set_k_to_update(self):
        """Find the Alice (k) that has the minimal number of possible
        outcomes, given the present state of knowledge"""

        best_width = 5

        for k, x in enumerate(self.xs):
            b_x = self.sp.x_to_len[x]
            if b_x == 1:
                continue
            elif b_x < best_width:
                best_width = b_x
                self.k_to_update = k
                if best_width == 2:
                    break

    def simplify_from_perturbed_ks(self) -> bool:
        """Look for the Alices that were modified (`perturbed`), and
        see whether any of the cycles they are connected to yield some
        non-trivial udpates"""

        while len(self.perturbed_ks) > 0:
            k = self.perturbed_ks.pop()

            if not self.simplify_from_k(k):
                return False

        if not self.is_full():
            self.set_k_to_update()
        
        return True

    def print_update(self, k, l):
        if self.detailed_verb:
            print(f"Updating {self.sp.k_to_str[k]}",
                f"based on cycle {self.sp.l_to_str[l]}:")
            wprint(self.to_long_str())

    def simplify_from_k(self, base_k: int) -> bool:
        """Checks the cycles connected to k and
        1 - tries to update other k's
        2 - checks for inconsistencies
        3 - return True or False depending on whether inconsistencies were
            found, and appends the updated ks to ks_to_check if updates
            were possible."""

        l_to_other_ks = self.sp.kl_to_other_ks[base_k]
        rules_for_x = self.sp.rb[self.xs[base_k]]
        
        # For all involved cycles, check status and try to update
        for l in self.sp.k_to_ls[base_k]:
            
            # NO UDPATES --------------

            # Keep track of the full & trusted l's
            if self.ltrust[l]:
                continue

            other_k1, other_k2 = l_to_other_ks[l]
            rule = rules_for_x[self.xs[other_k1]][self.xs[other_k2]]

            if not rule.consistent:
                if self.detailed_verb:
                    print(cR("INCONSISTENT"),"- see cycle",
                        f"{self.sp.l_to_str[l]} in the above!")
                return False

            if rule.full_after_update:
                # Will stay that way!
                self.ltrust[l] = True
            
            if not rule.non_trivial_update:
                # else will eventually be updatable. In any case continue.
                continue
            
            # UPDATES -----------------

            # If updating the base_k...
            if rule.non_trivial_pos[0]:
                self.xs[base_k] = rule.new_xs[0]
                self.perturbed_ks.add(base_k)
                self.print_update(base_k, l)
                break

            if rule.non_trivial_pos[1]:
                self.xs[other_k1] = rule.new_xs[1]
                self.perturbed_ks.add(other_k1)
                self.print_update(other_k1, l)
            
            if rule.non_trivial_pos[2]:
                self.xs[other_k2] = rule.new_xs[2]
                self.perturbed_ks.add(other_k2)
                self.print_update(other_k2, l)

        return True

    def copy_from(self, other):
        np.copyto(self.xs, other.xs)
        self.k_to_update = None
        np.copyto(self.ltrust, other.ltrust)

    def to_long_str(self) -> str:
        ret = "\n"
        for i in range(self.sp.order):
            if i > 0:
                ret += "\n\n"
            ret += "  "
            ret += " " * (4*self.depth_offset)
            for j in range(self.sp.order):
                if i == j:
                    ret += f"  {i % 10} "
                else:
                    x = self.xs[self.sp.ij_to_k[(i,j)]]
                    ret += self.sp.x_to_long_str[x]
                # ret += " "
        ret += "\n"
        return ret

    def to_short_str(self) -> str:
        ret = ""
        for i in range(self.sp.order):
            ret += "\n  "
            for j in range(self.sp.order):
                if i == j:
                    ret += f"{i % 10}"
                else:
                    x = self.xs[self.sp.ij_to_k[(i,j)]]
                    ret += self.sp.x_to_short_str[x]
                ret += " "
        ret += "\n"
        return ret

# ---------------------------------------
# ------------ SudokuSolver -------------
# ---------------------------------------

class SudokuSolver:
    """Takes in an inflation size, an initial filling (partial event),
    then coordinates the completion of the inflation event into a
    consistent one"""

    MAX_DEPTH=90

    def __init__(self, 
            order: int, 
            detailed_verb: bool = False,
            progress_verb: bool = False):
        
        self.sp = SudokuProblem(order)
        print(self.sp,"\n")

        self.time = time.time()
        self.wait_time = 0.15
        self.progress_time = time.time() + self.wait_time

        self.detailed_verb = detailed_verb
        self.progress_verb = progress_verb

        self.grid_states = [
            GridState(self.sp,self.detailed_verb,d) 
            for d in range(SudokuSolver.MAX_DEPTH)
            ]

        self.solution_grid = None

    def user_set_outcome(self, i, j, outcome):
        if self.detailed_verb:
            print(f"Initial filling: ",end="")
        self.grid_states[0].user_set_outcome(i,j,outcome)

    def set_diag_event(self, d, a,b,c):
        if a == b:
            self.user_set_outcome(  3*d, 3*d+1, a)
            self.user_set_outcome(3*d+1, 3*d+2, a)
        else:
            self.user_set_outcome(  3*d, 3*d+1, a)
            self.user_set_outcome(3*d+1, 3*d+2, b)
            self.user_set_outcome(3*d+2,   3*d, c)

    def get_time(self) -> str:
        return f"{round(time.time() - self.time,3)}s"

    def complete_grid(self) -> bool:
        """Wrapper around the recursive_complete_grid method"""

        print("The initial grid is setup to be")
        if self.detailed_verb:
            print(self.grid_states[0].to_long_str())
        else:
            print(self.grid_states[0].to_short_str())

        wprint("Press ENTER to complete...")
        print("")

        status = self.recursive_complete_grid(0)

        if not status:
            print(f"\nDid not find a solution grid. Took {self.get_time()}",
                "to complete search.\n")

        return status

    def recursive_complete_grid(self,current_depth: int) -> bool:
        """Guess the outcome of an Alice, update grid, and repeat"""

        cur_grid = self.grid_states[current_depth]

        if self.progress_verb and time.time() > self.progress_time \
                and current_depth >= 3:
            # Go to end of line
            s = go_right(2 + 3*12)
            s += "-" * (current_depth)
            s += str(current_depth)
            s = f"{s: <50}\r"
            print(s,end="",flush=True)
            self.progress_time = time.time() + self.wait_time

        # If we trust everything...
        if cur_grid.ltrust.all():
            self.solution_grid = cur_grid
            s = "\nFound a solution grid"
            if not self.detailed_verb:
                s += f"after {self.get_time()}:"
            print(s)
            wprint(self.solution_grid.to_short_str())
            return True

        # Assume rubbish
        child_grid = self.grid_states[current_depth+1]

        k_to_update = cur_grid.k_to_update
        for o in self.sp.x_to_psb[cur_grid.xs[k_to_update]]:
            
            if self.progress_verb and current_depth < 3:
                s = go_right(2+current_depth*12)
                s += f"{self.sp.k_to_str[k_to_update]} = {color_outcome(o+1)}"
                print(s,end="\r",flush=True)
            elif self.detailed_verb:
                print(cG(f"At depth {current_depth}:"),"try ",end="")

            child_grid.copy_from(cur_grid)

            ok_so_far = child_grid.set_outcome(k_to_update, o)

            if ok_so_far:
                completable = \
                    self.recursive_complete_grid(current_depth+1)
                if completable:
                    return True
                else:
                    if self.progress_verb and current_depth == 2:
                        print(go_right(2 + 3*12)+cR("inconsistent") + (" "*50),
                            end="\n")

        return False
