
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

FLOAT = np.single
INT = np.uint8

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
        # The two other k's in l part from k
        self.kl_to_other_ks = [
            [() for l in range(self.n_cycles)] 
            for k in range(self.n_alices)]

        self.create_kl_links()

        # The 0 pattern is excluded
        self.all_xs = [INT(x) for x in range(0,2**NOUT-1)]
        self.x_to_long_str = ["" for x in self.all_xs]
        self.x_to_short_str = [""for x in self.all_xs]
        self.x_to_len = [0 for x in self.all_xs]
        self.psb_to_x = dict()
        self.x_to_psb = [() for x in self.all_xs]
        
        self.create_x_stuff()

        self.xs_to_party_dH = dict()
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
            x_str = "["
            for i in range(NOUT):
                if cs[3-i] == 1:
                    x_str += color_outcome(i+1)
                else:
                    x_str += " "
            x_str += "]"
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
        return (o1==o2 and o2==o3) or (o1!=o2 and o2!=o3 and o3!=o1)

    def simplify_xs(self, xs: Tuple[INT,INT,INT]) -> Tuple[INT,INT,INT]:
            
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

    def H_cycle(self, xs: Tuple[INT,INT,INT]) -> FLOAT:
        return sum((FLOAT(np.log2(self.x_to_len[x])) for x in xs))

    def get_party_dH_cycle(self, xs: Tuple[INT,INT,INT]) -> FLOAT:
        """Return the average entropy change resulting from fixing
        the *first* outcome, i.e., that of xs[0]."""
        
        party_dH = FLOAT(0.)

        x1, x2, x3 = xs

        for o1 in self.x_to_psb[x1]:
            new_xs = (self.psb_to_x[(o1,)], x2, x3)
            inconsistent, new_xs = self.simplify_xs(new_xs)
            assert not inconsistent #otherwise non sensible

            count = 0
            for o2,o3 in itools.product(self.x_to_psb[x2],self.x_to_psb[x3]):
                if self.is_valid_event(o1, o2, o3):
                    count += 1
            
            party_dH += FLOAT(np.log2(FLOAT(count)))
            for other_p in (1,2):
                party_dH -= FLOAT(np.log2(FLOAT(self.x_to_len[xs[other_p]])))
            
        party_dH /= FLOAT(self.x_to_len[x1])

        return party_dH

    def create_rule_book(self):
        
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
            
            # Compute entropy changes:
            total_dH = FLOAT(0.)
            if non_trivial_update:
                total_dH = self.H_cycle(new_xs) - self.H_cycle(xs)
            
            # Append
            self.rb[xs[0]][xs[1]][xs[2]] = Rule(True,
                                    non_trivial_update,
                                    full_after_update,
                                    new_xs,
                                    non_trivial_pos,
                                    total_dH)

            # Using this opportunity to update the xs_to_party_dH, since
            # we are effectively iterating over all consistent rules.
            if self.x_to_len[new_xs[0]] > 1:
                self.xs_to_party_dH[new_xs] = self.get_party_dH_cycle(new_xs)
        
    def __str__(self) -> str:
        return cG("\n----- Inflation: ") \
                + f"{self.order} sources, " \
                + f"{self.n_alices} Alices, " \
                + f"{self.n_cycles} triangle-isomorphic subgraphs " \
                + cG("-----")

    def wprint_rule(self, xs: Tuple[INT,INT,INT]):
        x1,x2,x3 = xs
        if not (x1 <= x2 and x2 <= x3):
            return

        rule = self.rb[xs]

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
                    ret += ","
            else:
                if not rule.non_trivial_update:
                    ret += "cannot be simplified yet"

            ret += "\n" + offset
            ret += " " * (7 * 3) 
            ret += f"total_dH = {np.round(rule.total_dH,3)}"

        wprint(ret)

    def print_rule_book(self):
        print("------ Now printing the whole rule book. Press Ctrl+C to",
            "exit the program, and ENTER to continue. -------")
        
        for xs in itools.product(*[self.all_xs for i in range(3)]):
            self.wprint_rule(xs)

    def print_party_dH(self):
        print("------ Now printing all party_dH. Press Ctrl+C to",
            "exit the program, and ENTER to continue. -------")

        for key, val in self.xs_to_party_dH.items():
            ret = "\n Pattern "
            for i in range(3):
                ret += self.x_to_long_str[key[i]] + " "
            ret += f"-> entropy change of {round(float(val),3)} on avg over"
            ret += " 1st player outcomes, neglecting correl. prior to fixing,"
            ret += " counting correl. after fixing)"
            wprint(ret)
            
# -----------------------------------------
# -------------- Rule Class ---------------
# -----------------------------------------

class Rule:

    def __init__(self,
                consistent: bool,
                non_trivial_update: bool = None,
                full_after_update: bool = None,
                new_xs: Tuple[INT,INT,INT] = None,
                non_trivial_pos: Tuple[bool,bool,bool] = None,
                total_dH: FLOAT = None
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
        # The change of entropy of the cycle after update.
        self.total_dH = total_dH

# -----------------------------------------
# -------------- Strat Class --------------
# -----------------------------------------

class Strat:

    def __init__(self, n):
        self.n = n
        self.data = [[0 for j in range(n)] for i in range(n)]

    def __setitem__(self,ij,outcome):
        i,j = ij
        self.data[i][j] = outcome

    def __str__(self) -> str:
        ret = ""
        for i in range(self.n):
            ret += "  "
            for j in range(self.n):
                ret += color_outcome(self.data[i][j]) + " "
            ret += "\n"
        return ret

    def get_p(self, a,b,c) -> float:
        count = 0
        for alpha,beta,gamma in itools.product(*[range(self.n)] * 3):

            if self.data[alpha][beta] == a \
            and self.data[beta][gamma] == b \
            and self.data[gamma][alpha] == c:
                count += 1

        if count == 0:
            return 0.
        
        return round(float(count)/float(self.n ** 3),3)

    def to_p_str(self,a,b,c) -> Tuple[bool,str]:
        p = self.get_p(a,b,c)
        ret = "p("
        ret += color_outcome(a) + "," + color_outcome(b) + ","
        ret += color_outcome(c) + ") = " + f"{str(p): <6}"
        if p == 0.:
            return True, ret
        else:
            return False, ret

# -----------------------------------------
# ------------ GridState Class ------------
# -----------------------------------------

class GridState:
    """Stores a numbered filling of a grid"""
    INF = FLOAT(1e9)

    def __init__(self, 
            sp: SudokuProblem, 
            use_entropy: bool,
            detailed_verb: bool):
        self.sp = sp
        self.use_entropy = use_entropy
        self.detailed_verb = detailed_verb

        # A k-indexed list of values to be understood as "x"'s 
        # (see SudokuProblem) - essentially knowledge state about an outcome
        self.xs = np.full((sp.n_alices), COMPLETELY_UNKNOWN, dtype=np.uint8)
        self.total_H = sp.n_alices*FLOAT(2.) # this assumes four outcomes
        # This one will be computed based on entropy heuristics
        self.k_to_update = None
        # Set a value larger than anything we'll see here as a starting point.
        self.optimal_score = GridState.INF

        self.perturbed_ks = set()

        # self.ltrust[l] = True if the cycle is full & has been checked
        self.ltrust = np.full((sp.n_cycles), False, dtype=np.bool8)

    def user_set_outcome(self, i, j, outcome):
        """User supplies math-indexed outcome"""
        k = self.sp.ij_to_k[(i,j)]
        x = self.xs[k]
        assert (outcome-1) in self.sp.x_to_psb[x]

        # Will need to update these guys anyway. Don't need to put them
        # in the internal set_outcome because this one will only ever
        # be called on a "fresh" GridState.
        self.k_to_update = None
        self.optimal_score = GridState.INF

        assert self.set_outcome(k, outcome-1)

    def set_outcome(self, k, outcome) -> bool:
        """Returns true if consistent. Assume CS-indexed outcome"""
        # Remove old local entropy
        self.total_H -= FLOAT(np.log2(self.sp.x_to_len[self.xs[k]]))

        self.xs[k] = self.sp.psb_to_x[(outcome,)]

        self.perturbed_ks.add(k)

        return self.simplify_from_perturbed_ks()

    def is_full(self) -> bool:
        return self.ltrust.all()

    def set_k_to_update(self) -> bool:
        best_breadth = 5

        for x in self.xs:
            b_x = self.sp.x_to_len[x]
            if b_x == 1:
                continue
            elif b_x < best_breadth:
                best_breadth = b_x
                if best_breadth == 2:
                    break

        if (best_breadth == 2 and self.use_entropy) or not self.use_entropy:
            # Breadth-optimal k
            self.k_to_update = next(
                    (k
                    for k in range(self.sp.n_alices)
                    if self.sp.x_to_len[self.xs[k]] == best_breadth)
                )
        else:
            # Entropic optimal k
            for k in range(self.sp.n_alices):
                b_k = self.sp.x_to_len[self.xs[k]]
                if b_k != best_breadth:
                    # If k full or too entropic, essentially. 
                    # May want to store this info as a
                    # boolean array? #TODO think about it
                    continue
                
                total_dH = FLOAT(0.)

                for l in self.sp.k_to_ls[k]:
                    # For all cycle, add the expected entropy change.
                    # First need the xs in the right order:
                    triple_ks = self.sp.l_to_ks[l]
                    index_k_in_l = triple_ks.index(k)
                    ordered_ks = tuple(
                        triple_ks[(index_k_in_l+i)%3] for i in range(3)
                        )
                    ordered_xs = tuple(
                        self.xs[k_bis] for k_bis in ordered_ks
                    )
                    total_dH += self.sp.xs_to_party_dH[ordered_xs]
                
                #score = FLOAT(np.log2(b_k)) * (self.total_H + total_dH)
                score = total_dH

                if score < self.optimal_score:
                    self.optimal_score = score
                    self.k_to_update = k

    def simplify_from_perturbed_ks(self) -> bool:
        while len(self.perturbed_ks) > 0:
            k = self.perturbed_ks.pop()

            if not self.simplify_from_k(k):
                return False

        if not self.is_full():
            self.set_k_to_update()
        
        return True

    def get_rule(self, l: int) -> Tuple[Rule,Tuple[int,int,int]]:
        """Returns the rule associated to the cycle l, together with the
        three concerned k-indices (Alices)"""
        ks = self.sp.l_to_ks[l]
        # xs = tuple(self.xs[k] for k in ks)
        # rule = self.sp.rb[(self.xs[ks[0]],self.xs[ks[1]],self.xs[ks[2]])]
        return self.sp.rb[self.xs[ks[0]]][self.xs[ks[1]]][self.xs[ks[2]]], ks

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
            #rule, ks = self.get_rule(l)

            if not rule.consistent:
                if self.detailed_verb:
                    print("\n\n\n INCONSISTENT: \n")
                    print(self.to_short_str())
                    #print(f"H = {self.total_H}",end="",flush=True)
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
                break

            if rule.non_trivial_pos[1]:
                self.xs[other_k1] = rule.new_xs[1]
                self.perturbed_ks.add(other_k1)
            
            if rule.non_trivial_pos[2]:
                self.xs[other_k2] = rule.new_xs[2]
                self.perturbed_ks.add(other_k2)

            # for  k,       new_x,       is_non_trivial in zip(
            #     ks, rule.new_xs, rule.non_trivial_pos):

            #     if is_non_trivial:
            #         if self.detailed_verb:
            #             s = "Can update " + self.sp.k_to_str[k] + " to "
            #             s += self.sp.x_to_long_str[new_x]
            #             print(s)
            #             print(str(self))

            #         # Update the cycle
            #         self.xs[k] = new_x
            #         # Append:
            #         self.perturbed_ks.add(k)

            #if rule.full_after_update:
            #    self.ltrust[l] = True

            # Total entropy
            # self.total_H += rule.total_dH
            
        return True

    def copy_from(self, other):
        np.copyto(self.xs, other.xs)
        self.total_H = other.total_H
        self.optimal_score = GridState.INF
        self.k_to_update = None
        np.copyto(self.ltrust, other.ltrust)

    def get_metadata(self) -> str:
        ret = ""
        if self.k_to_update is not None:
            ret = f"\n\n    Total entropy: {round(float(self.total_H),3)}"
            ret += f"\n    Should update {self.sp.k_to_str[self.k_to_update]}"
            ret += f" next, score = {round(float(self.optimal_score),3)}"
        ret += "\n"
        return ret

    def to_long_str(self) -> str:
        ret = "\n"
        for i in range(self.sp.order):
            if i > 0:
                ret += "\n\n\n"
            ret += "  "
            for j in range(self.sp.order):
                if i == j:
                    ret += f"{i % 10: >6}"
                else:
                    x = self.xs[self.sp.ij_to_k[(i,j)]]
                    if x == COMPLETELY_UNKNOWN:
                        s = " "
                        ret += f"{s: >6}"
                    else:
                        ret += self.sp.x_to_long_str[x]
                ret += " "
        ret += self.get_metadata()
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
        ret += self.get_metadata()
        return ret

    def swap_sources(self, s1, s2):
        for i in range(self.sp.order):
            if i != s1 and i != s2:
                self.xs[self.sp.ij_to_k[(i,s1)]], \
                self.xs[self.sp.ij_to_k[(i,s2)]] = \
                self.xs[self.sp.ij_to_k[(i,s2)]], \
                self.xs[self.sp.ij_to_k[(i,s1)]]

                self.xs[self.sp.ij_to_k[(s1,i)]], \
                self.xs[self.sp.ij_to_k[(s2,i)]] = \
                self.xs[self.sp.ij_to_k[(s2,i)]], \
                self.xs[self.sp.ij_to_k[(s1,i)]]
        
        self.xs[self.sp.ij_to_k[(s1,s2)]], \
        self.xs[self.sp.ij_to_k[(s2,s1)]] = \
        self.xs[self.sp.ij_to_k[(s2,s1)]], \
        self.xs[self.sp.ij_to_k[(s1,s2)]]

    def swap_source_list(self, swaps):
        for s1,s2 in swaps:
            self.swap_sources(s1,s2)

        print("After swaps, have:")
        print(self.to_short_str())

    def extend_to_strat(self,diag_info):
        strat = Strat(self.sp.order)

        diag = [0 for _ in range(self.sp.order)]
        for s1,s2,outcome in diag_info:
            for i in range(s1,s2+1):
                diag[i] = outcome

        for i in range(self.sp.order):
            for j in range(self.sp.order):
                if i == j:
                    strat[(i,i)] = diag[i]
                else:
                    strat[(i,j)] = 1 + self.sp.x_to_psb[
                            self.xs[
                                self.sp.ij_to_k[(i,j)]
                            ]
                        ][0]

        print("Strategy:\n")
        print(strat)

        print(f"Outcome distribution:")

        print("\n  --- 111 --- ")
        for a in range(1,NOUT+1):
            is_zero, ret = strat.to_p_str(a,a,a)
            if is_zero:
                ret += cR("-> shouldn't be zero")
            print(ret)
            
        print("\n --- 123 --- ")
        for a,b,c in itools.product(*[range(1,NOUT+1)] * 3):
            if not (a!=b and b!=c and c!=a):
                continue        
            is_zero, ret = strat.to_p_str(a,b,c)
            if is_zero:
                ret += cR("-> shouldn't be zero")
            print(ret)
        
        first_error = True
        for a,b,c in itools.product(*[range(1,NOUT+1)] * 3):
            if (a==b and b==c) or (a!=b and b!=c and c!=a):
                continue
            is_zero, ret = strat.to_p_str(a,b,c)
            if not is_zero:
                if first_error:
                    print("\n --- 112 ---")
                    first_error = False
                print(ret + cR("-> should be zero"))

# ---------------------------------------
# ------------ SudokuSolver -------------
# ---------------------------------------

class SudokuSolver:
    MAX_DEPTH=90

    def __init__(self, 
            order: int, 
            branching_attitude: str = "BreadthOpt",
            detailed_verb: bool = False,
            progress_verb: bool = False):
        
        self.sp = SudokuProblem(order)
        print(self.sp)

        self.time = time.time()

        self.use_entropy = False
        if branching_attitude == "Entropy":
            self.use_entropy = True
            print(cY(" *** USING ENTROPY! *** "))
        else:
            print(cY(" *** USING BREADTH OPTIMALITY! ***"))

        self.detailed_verb = detailed_verb
        self.progress_verb = progress_verb

        self.grid_states = [
            GridState(self.sp,self.use_entropy,self.detailed_verb) 
            for d in range(SudokuSolver.MAX_DEPTH)
            ]

        self.solution_grid = None

    def user_set_outcome(self, i, j, outcome):
        self.grid_states[0].user_set_outcome(i,j,outcome)

    def set_diag_event(self, d, a,b,c):
        if a == b:
            self.user_set_outcome(  3*d, 3*d+1, a)
            self.user_set_outcome(3*d+1, 3*d+2, a)
        else:
            self.user_set_outcome(  3*d, 3*d+1, a)
            self.user_set_outcome(3*d+1, 3*d+2, b)
            self.user_set_outcome(3*d+2,   3*d, c)

    def complete_grid(self, verb=False) -> bool:

        print("\nThe initial grid is setup to be")
        print(self.grid_states[0].to_short_str())

        print("Completing the grid...")

        return self.recursive_complete_grid(0,verb)

    def recursive_complete_grid(self,current_depth: int,verb) -> bool:
        # if time.time() - self.time >= 10.0:
        #     return True # TIMEOUT

        cur_grid = self.grid_states[current_depth]

        if self.progress_verb and current_depth % 3 == 0:
            s = "-" * (current_depth)
            s += str(current_depth)
            s = f"{s: <150}"
            s += "\r"
            print(s,end="",flush=True)

        if self.detailed_verb:
            wprint("Entering recursive: " + cur_grid.to_short_str())

        # If we trust everything...
        if cur_grid.ltrust.all():
            print("\n:D")
            self.solution_grid = cur_grid
            print("\nFound a solution grid:")
            wprint(self.solution_grid.to_short_str())
            return True

        # Assume rubbish
        child_grid = self.grid_states[current_depth+1]

        k_to_update = cur_grid.k_to_update
        for o in self.sp.x_to_psb[cur_grid.xs[k_to_update]]:
            
            if self.progress_verb and current_depth < 3:
                s = ""
                s += " " * (current_depth*12)
                s += f"{self.sp.k_to_str[k_to_update]} = {color_outcome(o+1)}"
                print(s)

            child_grid.copy_from(cur_grid)

            if current_depth > 5:
                child_grid.use_entropy = False
            else:
                child_grid.use_entropy = False

            ok_so_far = child_grid.set_outcome(k_to_update, o)

            if ok_so_far:
                completable = \
                    self.recursive_complete_grid(current_depth+1,verb)
                if completable:
                    return True
        
        return False

            



