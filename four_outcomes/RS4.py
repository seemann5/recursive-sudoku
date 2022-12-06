
import itertools, time, sys, copy
from typing import List, Tuple
import numpy as np
from coloring import *

# --------------------------------------------
# ------------ Utilities ---------------------
# --------------------------------------------

def wprint(arg):
    input(arg)

# --------------------------------------------
# ------------ SudokuProblem Class -----------
# --------------------------------------------

NOUT=4
COMPLETELY_UNKNOWN=14
    
class SudokuProblem:
    """Stores basic Alice&Cycle info"""

    def __init__(self, order):
        # The size of the grid
        self.order = order
        # The alice & cycles part
        self.n_alices = order*(order-1)
        self.n_cycles = int(2 * order*(order-1)*(order-2) / (3*2))
        self.k_to_str = ["" for k in range(self.n_alices)]
        self.l_to_str = ["" for l in range(self.n_cycles)]
        self.ij_to_k = dict()
        # The Alices (ks) that the cycle (l) touches
        self.l_to_ks = [[] for l in range(self.n_cycles)]
        # The cycles that Alice partipates to
        self.k_to_ls = [[] for k in range(self.n_alices)]

        self.create_kl_links()

        # The 0 pattern is excluded
        self.all_xs = range(0,2**NOUT-1)
        self.x_to_long_str = ["" for x in self.all_xs]
        self.x_to_short_str = [""for x in self.all_xs]
        self.x_to_len = [0 for x in self.all_xs]
        self.psb_to_x = dict()
        self.rb = [
                [
                    [None for z in self.all_xs]
                for y in self.all_xs]
            for x in self.all_xs]
        self.create_x_stuff()
        
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
            itertools.chain.from_iterable(
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

    def create_x_stuff(self):

        # A psb is defined as a tuple (0,2,3) etc
        list_psb = []
        for cs in itertools.product((0,1),(0,1),(0,1),(0,1)):
            if sum(cs) == 0:
                # Not a valid psb
                continue

            psb = []
            for i in range(NOUT):
                if cs[3-i] == 1:
                    psb.append(i)
            psb = tuple(psb)
            list_psb.append(psb)

            x = -1
            for o in psb:
                x += 2**o
            self.psb_to_x[psb] = x
            self.x_to_len[x] = len(psb)

            x_str = "["
            for i in range(NOUT):
                if cs[3-i] == 1:
                    x_str += color_outcome(i+1)
                else:
                    x_str += " "
            x_str += "]"
            self.x_to_long_str[x] = x_str

            if sum(cs) == 1:
                for i in range(NOUT):
                    if cs[3-i] == 1:
                        self.x_to_short_str[x] = color_outcome(i+1)
            else:
                self.x_to_short_str[x] = "."

        # Create rulebook
        for triple_psb in itertools.product(list_psb,list_psb,list_psb):
            xs = tuple(self.psb_to_x[psb] for psb in triple_psb)

            # So now we're trying to see how to update.
            
            # Now check for consistency & simplifications at the same time.
            update_triple_psb = [list(psb) for psb in triple_psb]
            
            ps_to_check = {0,1,2}

            was_inconsistent = False

            while len(ps_to_check) > 0:

                p = ps_to_check.pop()

                keep_indices = []

                for i1, o1 in enumerate(update_triple_psb[p]):

                    for o2, o3 in itertools.product(
                            update_triple_psb[(p+1)%3],
                            update_triple_psb[(p+2)%3]):

                        if (o1 == o2 and o2 == o3) \
                            or (o1 != o2 and o2 != o3 and o3 != o1):
                            keep_indices.append(i1)
                            break
                
                if len(keep_indices) == len(update_triple_psb[p]):
                    # We haven't removed anything.
                    # Nothing to do!
                    pass
                elif len(keep_indices) == 0:
                    # There is nothing to keep... Inconsistency:
                    self.rb[xs[0]][xs[1]][xs[2]] = Rule(False)
                    #self.wprint_rule(xs)
                    was_inconsistent = True
                    break
                else:
                    # We're making a consistent update. Must re-check the 
                    # other p's...
                    for other_p in range(2):
                        if other_p != p:
                            ps_to_check.add(other_p)
                    # ... AND update:
                    update_triple_psb[p] = [
                        update_triple_psb[p][i]
                        for i in keep_indices
                    ]
            
            if was_inconsistent:
                continue

            # We're done, and we haven't reached an inconsistency. 
            # Hence, we have some sort of consistent update.

            # Check for non-trivial:

            non_trivial_update = False
            non_trivial_pos = None

            for p in range(3):
                if len(update_triple_psb[p]) < len(triple_psb[p]):
                    non_trivial_update = True

                    if non_trivial_pos is None:
                        non_trivial_pos = [False for p in range(3)]
                    
                    non_trivial_pos[p] = True

            update_rule = None
            
            if non_trivial_update:
                update_rule = [
                    self.psb_to_x[tuple(update_triple_psb[p])]
                    for p in range(3)
                ]

            # Check for full:

            full_after_update = True

            for p in range(3):
                if len(update_triple_psb[p]) != 1:
                    full_after_update = False
                    break
            
            # Append
            self.rb[xs[0]][xs[1]][xs[2]] = Rule(
                True,
                non_trivial_update,
                full_after_update,
                update_rule,
                non_trivial_pos)
        
    def __str__(self) -> str:
        return cG("----- Inflation:") \
                + f"{self.order} sources, " \
                + f"{self.n_alices} Alices, " \
                + f"{self.n_cycles} triangle-isomorphic subgraphs " \
                + cG("-----")

    def wprint_rule(self, xs: Tuple[np.uint8,np.uint8,np.uint8]):
        x1,x2,x3 = xs
        if not (x1 <= x2 and x2 <= x3):
            return
        ret = "\n"
        ret += " " * 4
        for x in xs:
            ret += self.x_to_long_str[x] + " "

        rule = self.rb[x1][x2][x3]

        if not rule.consistent:
            ret += "\n"
            ret += " " * (4+6*3+3)
            ret += "is inconsistent"
        else:
            if rule.non_trivial_update:
                ret += "can be updated to\n"
                ret += " " * 4 
                for i in range(3):
                    if rule.non_trivial_pos[i]:
                        ret += self.x_to_long_str[rule.update_rule[i]]
                    else:
                            ret += " " * 6
                    ret += " "
            else:
                ret += "\n"
                ret += " " * (4 + 6 * 3 + 3)
            if rule.full_after_update:
                ret += "[FULL]"

        wprint(ret)

    def print_rule_book(self):
        print("------ Now printing the whole rule book. Press Ctrl+C to",
            "exit the program, and ENTER to continue. -------")
        
        for xs in itertools.product(*[self.all_xs for i in range(3)]):
            self.wprint_rule(xs)
            
# -----------------------------------------
# -------------- RuleBook Class -----------
# -----------------------------------------

class Rule:

    def __init__(self,
                consistent: bool,
                non_trivial_update: bool = False,
                full_after_update: bool = False,
                update_rule: Tuple[np.uint8,np.uint8,np.uint8] = None,
                non_trivial_pos: Tuple[bool,bool,bool] = None
                ):
        self.consistent = consistent
        # The following two are independent but assume that consistent==True
        self.non_trivial_update = non_trivial_update
        self.full_after_update = full_after_update
        # This will be a tuple of the form (x1, x2, x3)
        self.update_rule = update_rule
        # A tuple of the form (True, False, True) if first & third values
        # need to be updated
        self.non_trivial_pos = non_trivial_pos

    def is_equal_to(self, other):
        return self.consistent == other.consistent \
            and self.non_trivial_update == other.non_trivial_update \
            and self.full_after_update == other.full_after_update \
            and (
                (self.update_rule is None and other.update_rule is None) \
                or
                len(
                    [x for x,y in zip(self.update_rule,other.update_rule)
                    if x != y]
                    ) == 0 
                ) \
            and (
                (self.non_trivial_pos is None and other.non_trivial_pos is None) \
                or (
                len(
                    [x for x,y in zip(self.non_trivial_pos,other.non_trivial_pos)
                    if x != y]
                ) == 0
                and
                    len(self.non_trivial_pos) == len(other.non_trivial_pos)
                ))

# -----------------------------------------
# ------------ GridState Class ------------
# -----------------------------------------

UNKNOWN=15

class GridState:
    """Stores a numbered filling of a grid"""

    def __init__(self, sp: SudokuProblem):
        self.sp = sp

        # A k-indexed list of values to be understood as "x"'s 
        # (see SudokuProblem) - essentially knowledge state about an outcome
        self.xs = np.full((sp.n_alices), UNKNOWN, dtype=np.uint8)

        self.perturbed_ks = set()

        # self.ltrust[l] = True if the cycle is full & has been checked
        self.ltrust = np.full((sp.n_cycles), False, dtype=np.bool8)

    def get_rule(self, l: int) -> Tuple[Rule,List[int]]:
        """Returns the rule associated to the cycle l, together with the
        three concerned k-indices (Alices)"""
        #TODO
        pass
        # return (self.sp.rb[self.xs.])

    def check_cycles_for_updates_and_inconsistencies(self, k: int) -> bool:
        """Checks the cycles connected to k and
        1 - tries to update other k's
        2 - checks for inconsistencies
        3 - return True or False depending on whether inconsistencies were
            found, and appends the updated ks to ks_to_check if updates
            were possible."""
        
        # For all involved cycles, check status and try to update
        for l in self.np.k_to_ls[k]:
            
            # CONTINUE's --------------

            # Keep track of the full & trusted l's
            if self.ltrust[l]:
                continue

            rule, ks = self.get_rule(l)

            if "consistent, un-updatable and full":
                # Will stay that way!
                self.ltrust[l] = True
                continue

            if "consistent, un-updatable but not full":
                # Will eventually be updatable
                continue

            # INCONSISTENT ------------
            if "inconsistent":
                # Needs to be computed here
                return False
            
            # UPDATES -----------------

            if "consistent, updatable":
                # TODO: update cycle
                # TODO: append to perturbed_cycles

                if "consistent, updatable and full":
                    self.ltrust[l] = True
            

            # Only invoke the method for non-full things, but 
            # check if return is full
            k1,k2,k3 = self.sp.l_to_ks[l]
            consistent, full, nothing_changed, update_rules = \
                "TODO"

            
        return True

                
                




    # def update_based_on(self, fill_step: FillingStep, outcome: int) -> bool:
    #     """Returns True if sensible update, False otherwise"""
    #     self.event[fill_step.initial_k] = outcome

    #     # Update things, no possibility of inconsistency here
    #     for k_update, k1, k2 in fill_step.update_rules:
    #         self.event[k_update] = \
    #             self.sp.ab_to_c[(self.event[k1],self.event[k2])]
        
    #     # Check for inconsistency
    #     for k1,k2,k3 in fill_step.check_rules:
    #         if not (self.event[k1],self.event[k2],self.event[k3]) \
    #                 in self.sp.valid_events:
    #             return False

    #     return True

    # def copy_from(self, other):
    #     np.copyto(self.event, other.event)

    # def __str__(self) -> str:
    #     ret = "\n"
    #     for i in range(self.sp.order):
    #         ret += "  "
    #         for j in range(self.sp.order):
    #             if j > 0:
    #                 ret += " "
    #             if i == j:
    #                 ret += str(i % 10)
    #             else:
    #                 outcome = self.event[self.sp.ij_to_k[(i,j)]]
    #                 if outcome == 0:
    #                     ret += " "
    #                 else:
    #                     ret += color_outcome(outcome)
    #         ret += "\n"
    #     return ret

    # def swap_sources(self, s1, s2):
    #     for i in range(self.sp.order):
    #         if i != s1 and i != s2:
    #             self.event[self.sp.ij_to_k[(i,s1)]], \
    #             self.event[self.sp.ij_to_k[(i,s2)]] = \
    #             self.event[self.sp.ij_to_k[(i,s2)]], \
    #             self.event[self.sp.ij_to_k[(i,s1)]]

    #             self.event[self.sp.ij_to_k[(s1,i)]], \
    #             self.event[self.sp.ij_to_k[(s2,i)]] = \
    #             self.event[self.sp.ij_to_k[(s2,i)]], \
    #             self.event[self.sp.ij_to_k[(s1,i)]]
        
    #     self.event[self.sp.ij_to_k[(s1,s2)]], \
    #     self.event[self.sp.ij_to_k[(s2,s1)]] = \
    #     self.event[self.sp.ij_to_k[(s2,s1)]], \
    #     self.event[self.sp.ij_to_k[(s1,s2)]]
