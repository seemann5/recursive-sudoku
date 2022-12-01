

import itertools
from typing import List,Tuple,Iterator
import numpy as np
import time, sys

# --------------------------------------------
# ------------ SudokuProblem Class -----------
# --------------------------------------------
    
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
        # Allows to deduce the third outcome of a cycle givne the first two
        self.ab_to_c = dict()
        # Allows to simply & readably check whether a==b==c or a!=b!=c!=a
        self.valid_events = set()
        for a in (1,2,3):
            self.ab_to_c[(a,a)] = a
            self.valid_events.add((a,a,a))
        for perm in itertools.permutations((1,2,3)):
            a, b, c = perm
            self.ab_to_c[(a,b)] = c
            self.valid_events.add((a,b,c))

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
        
    def __str__(self) -> str:
        return f"--- Inflation: {self.order} sources, " \
                + f"{self.n_alices} Alices, " \
                + f"{self.n_cycles} triangle-isomorphic subgraphs ---"

# -------------------------------------------
# ---------------- FillState ---------------
# -------------------------------------------

class FillState:
    """Stores together a filling and the corresponding cinfo"""

    def __init__(self, sp: SudokuProblem):
        self.sp = sp
        # A sequence of True=filled/False=not filled for alices (k-indexed)
        self.filling = np.full((sp.n_alices), False, dtype=bool)
        # companion of filling. A sequence of 0/1/2/3 for each cycle (l-index)
        self.cinfo = np.full((sp.n_cycles), 0, dtype=int)
        # indicating how many alices are filled in the cycle.

    def copy_from(self, other: "FillState"):
        np.copyto(self.filling, other.filling)
        np.copyto(self.cinfo, other.cinfo)

    def copy(self):
        ret = FillState(self.sp)
        ret.copy_from(self)
        return ret
    
    def get_k_to_update(self, l: int) -> int:
        """This methods assumes that self.cinfo[l] == 2, such that
        there is only one unfilled Alice in the cycle l"""
        return next(k for k in self.sp.l_to_ks[l] if not self.filling[k])
    
    def fill_and_append_to(
                self,
                k_to_update: int,        
                cycles_to_check: List[int]
            ):
        """Partially updates self and appends to cycles_to_check"""
        self.filling[k_to_update] = True

        for involved_l in self.sp.k_to_ls[k_to_update]:
            self.cinfo[involved_l] += 1
            if self.cinfo[involved_l] == 2:
                cycles_to_check.append(involved_l)

    def is_full(self) -> bool:
        return np.all(self.filling)

    def __str__(self) -> str:
        ret = "\n"
        for i in range(self.sp.order):
            ret += "  "
            if i > 0:
                for j in range(self.sp.order):
                    ret += "    "
                ret += "\n  "
            for j in range(self.sp.order):
                if j > 0:
                    ret += "   "

                if i == j:
                    ret += f"{i}"
                else:
                    if self.filling[self.sp.ij_to_k[(i,j)]]:
                        ret += "F"
                    else:
                        ret += "."
            ret += "\n"
        return ret

# --------------------------------------------
# -------------- FillingStep -----------------
# --------------------------------------------

class FillingStep:
    """
    This class contains an initial position to be filled,
    a list of positions that can be directly deduced,
    and a list of cycles to be checked for consistency.
    A FillingStep object is only sensible in the context of a parent
    Filling object to which it refers.
    """
    def __init__(self, sp: SudokuProblem, initial_k: int):
        self.sp = sp
        self.initial_k = initial_k
        # Will be a list of Tuple(k-position to update,
        #                         l-cycle to use for update)
        self.update_rules = []
        # Will be a list of cycles to check
        self.check_rules = []

    def variable_str(self, lim: int = None) -> str:
        """lim = max number of check/update rules displayed. 
        None: everything"""
        ret = f"Update {self.sp.k_to_str[self.initial_k]}"
        for r, (update_k, k1, k2) in enumerate(self.update_rules):
            if r == 0:
                ret += " then\n"
            elif lim is not None and r >= lim:
                ret += f"\n  ... (total of {len(self.update_rules)}" \
                     + " update rules)"
                break
            else:
                ret += "\n"
            ret += f"  update {self.sp.k_to_str[update_k]}" \
                 + f" based on {self.sp.k_to_str[k1]}" \
                 + f" and {self.sp.k_to_str[k2]}"
        for r, (k1,k2,k3) in enumerate(self.check_rules):
            if r == 0:
                ret += "\n"
            elif lim is not None and r >= lim:
                ret += f"\n  ... (total of {len(self.check_rules)}" \
                         + " check rules)"
                break
            else:
                ret += "\n"
            ret += "  check cycle" \
                + f" {self.sp.k_to_str[k1]}" \
                + f" - {self.sp.k_to_str[k2]}" \
                + f" - {self.sp.k_to_str[k3]}"
        return ret

    def __str__(self) -> str:
        return self.variable_str()
    
# -----------------------------------------
# ------------ GridState Class ------------
# -----------------------------------------

class GridState:
    """Stores a numbered filling of a grid"""

    def __init__(self, sp: SudokuProblem):
        self.sp = sp

        self.event = np.full((sp.n_alices), 0, dtype=int)

    def update_based_on(self, fill_step: FillingStep, outcome: int) -> bool:
        """Returns True if sensible update, False otherwise"""
        self.event[fill_step.initial_k] = outcome

        # Update things, no possibility of inconsistency here
        for k_update, k1, k2 in fill_step.update_rules:
            self.event[k_update] = \
                self.sp.ab_to_c[(self.event[k1],self.event[k2])]
        
        # Check for inconsistency
        for k1,k2,k3 in fill_step.check_rules:
            if not (self.event[k1],self.event[k2],self.event[k3]) \
                    in self.sp.valid_events:
                return False

        return True

    def copy_from(self, other):
        np.copyto(self.event, other.event)

    def __str__(self) -> str:
        ret = "\n"
        for i in range(self.sp.order):
            ret += "  "
            if i > 0:
                for j in range(self.sp.order):
                    ret += "    "
                ret += "\n  "
            for j in range(self.sp.order):
                if j > 0:
                    ret += "   "
                    
                if i == j:
                    ret += "+"
                else:
                    outcome = self.event[self.sp.ij_to_k[(i,j)]]
                    if outcome == 0:
                        ret += " "
                    else:
                        ret += str(outcome)
                
            ret += "\n"
        return ret

# --------------------------------------------
# ------------ SudokuSolver Class ------------
# --------------------------------------------
    
class SudokuSolver:
    """Stores a SudokuProblem, a filling order, and 
    is the maestro of the solvign process."""

    def __init__(self, order):
        # General problem info 
        self.sp = SudokuProblem(order)
        print(self.sp)

        # The user-supplied initial filling_steps (related to d-sep)
        self.initial_filling_steps = []
        # The optimized filling_steps
        self.final_filling_steps = []
        
    # --------- String methods ---------
        
    def filling_steps_to_str(self, filling_steps: List["FillingStep"]) -> str:
        ret = ""
        for f, filling_step in enumerate(filling_steps):
            ret += self.sp.k_to_str[filling_step.initial_k]
            if f < len(filling_steps)-1:
                ret += " - "
        return ret
    
    def filling_steps_to_algo(self, filling_steps: List[FillingStep]) -> str:
        ret = ""
        for f, filling_step in enumerate(filling_steps):
            ret += str(filling_step)
            if f < len(filling_steps)-1:
                ret += "\n\n"
        return ret

    def get_resulting_depth(self) -> int:
        return len(self.final_filling_steps)-1
        
    # -------- Fill order setup --------
    
    def set_filling_steps_starting_with(
            self, 
            list_ij: List[Tuple[int,int]],
            verb=False,
            satisfying_depth=0):
        """This lets the user supply a choice of initial filling location,
        in the format (i,j).
        Errors will be thrown if the list_ij is "not minimal", e.g., contains
        twice the same location, and if some i == j.
        Then, the FillOptimizer will be called to determine a depth-optimal
        sequence of filling_steps to fill in completely the grid.
        These are then processed to obtain the full filling_steps.
        """

        # The intermediate fillobject after the initial filling_steps
        fill_state = FillState(self.sp)
        
        self.initial_filling_steps = []

        for (i,j) in list_ij:
            try:
                k = self.sp.ij_to_k[(i,j)]
            except KeyError:
                print(f"Error: the indices ({i},{j}) are invalid",
                    "(out of range or equal to each other)")
                sys.exit()

            fill_step = update_fill_state_and_get_fill_step(fill_state, k)

            self.initial_filling_steps.append(fill_step)
            
        if verb:
            print("Initially, we fill in as:",
                self.filling_steps_to_str(self.initial_filling_steps))
            print("This results in the fill_state")
            print(fill_state)

        self.final_filling_steps = []

        fill_optimizer = FillOptimizer(self.sp, fill_state, satisfying_depth)

        print("Optimizing...", flush=True, end="")
        t_i = time.time()
        best_final_ks = fill_optimizer.get_depth_optimal_filling_order()
        t_f = time.time()
        print(f" done [{np.round(t_f-t_i,3)}s]")

        for k in best_final_ks:
            fill_step = update_fill_state_and_get_fill_step(fill_state, k)

            self.final_filling_steps.append(fill_step)

        if verb:
            print("Completed with:",
                self.filling_steps_to_str(self.final_filling_steps))

            print("In details, the full algo is:")
            print("----------------------------------")
            print(self.filling_steps_to_algo(self.initial_filling_steps))
            print("----------------------------------")
            print(self.filling_steps_to_algo(self.final_filling_steps))
            print("----------------------------------")

    def set_initial_outcomes_and_complete(
        self,
        initial_outcomes: List[int],
        verb=False
        ) -> bool:
        
        base_grid = GridState(self.sp)

        if not len(initial_outcomes) == len(self.initial_filling_steps):
            print(f"Error: you gave {len(initial_outcomes)} outcomes",
                f" but there are {len(self.initial_filling_steps)}",
                "initial filling steps set up.")
            sys.exit()

        for fill_step, outcome \
                in zip(self.initial_filling_steps, initial_outcomes):
            
            if not outcome in (1,2,3):
                print(f"Error: outcome {outcome} out of range.")
                sys.exit()

            all_good = base_grid.update_based_on(fill_step, outcome)

            if not all_good:
                print("Error: there is some self-inconsistency in the",
                    "outcomes provided.")
                sys.exit()
        
        if verb:
            print("The initial grid is now setup to be")
            print(base_grid)

        

# -------------------------------------------
# ------------ FillOptimizer ----------------
# -------------------------------------------
        
class FillOptimizer:
    """This class finds the depth-optimal (unless a satisfying_depth > 0 
    is specified) fill choice"""
    CACHE_SIZE = int(1e5)

    def __init__(self, sp: SudokuProblem,
                       initial_fill_state: FillState,
                       satisfying_depth: int = 0):
        self.sp = sp
        # A hopeful estimate. Might need to tweak this depending on the setup.
        self.max_depth = sp.order 
        # Stop there for efficiency although this may be suboptimal.
        # Value > 0 => sub-optimal solution.
        self.satisfying_depth = satisfying_depth
        
        # A pre-allocated list of fill states. 
        # self.fill_states[d] will be used at depth d.
        # +1 for working purposes.
        self.fill_states = [
                FillState(self.sp) 
                for d in range(self.max_depth+1)
            ]
        self.fill_states[0].copy_from(initial_fill_state)

        # A pre-allocated list of fill orders (i.e., list of k-indices)
        # self.fill_orders[d] will be used at depth d.
        self.fill_orders = [
                np.full((self.max_depth), 1024, dtype=int) 
                for d in range(self.max_depth+1)
            ]
        
        self.optimal_fill_order = []

        # CACHE_SIZE Times for ORDER = 12 with diagonal intiial filling:
            #1e6 -> 4s
            #1e2 -> 45s
            #1e3 -> 43s
            #1e4 -> 14.7s
            #1e5 -> 4s
            # For ORDER = 15 with diagonal fillings:
            # 1e9 -> More than one minute, no patience left.
            
        # The cache dictionary.
        # keys = hash of (filling position, filling)
        # vals = output fo, to be used with np.copyto(..., output fo)
        self.cache_dict = dict()
        # Initially, the cache is not full.
        # Then, when the cache is full, implement a basic LRU
        self.cache_full = False
        self.hash_encounters = [None for i in range(FillOptimizer.CACHE_SIZE)]
        self.lru_index = -1

    def get_depth_optimal_filling_order(self):
        """A wrapper around the recursive function."""
        if len(self.optimal_fill_order) > 0:
            return self.optimal_fill_order

        # The quick recursive magic
        optimal_depth = self.recursive_set_best_fill_order()

        self.optimal_fill_order = [
                self.fill_orders[0][d] for d in range(optimal_depth+1)
            ]
        
        return self.optimal_fill_order

    def recursive_set_best_fill_order(
            self,
            previous_k: int = -1,
            current_depth: int = 0
        ) -> int:
        """The return value is the best depth we found, or None"""

        # Assume of the form [k_0, ..., k_{d-1}, INF, ... INF]
        cur_fill_order = self.fill_orders[current_depth]
        # Assume well initalized
        cur_fill_state = self.fill_states[current_depth]

        # Assume these are currently rubbish
        child_fill_order = self.fill_orders[current_depth+1]
        child_fill_state = self.fill_states[current_depth+1]

        # The future best depth we found. Can be None if we don't find
        # anything.
        best_depth = None

        # Start at previous_k+1, because filling is a commutative operation
        # as far as depth is conserved (in the sense that if a fill_order
        # fills in the grid, then a permutation thereof also does)
        for k in range(previous_k+1, self.sp.n_alices):
            # Cannot fill an already filled Alice
            # This shouldn't be moved outside the loop: the loop will fill in
            # some new Alices during its execution
            if cur_fill_state.filling[k]:
                continue

            # -------------
            # LRU cache of the child_fill_state update
            the_hash = hash(
                    (k, tuple(cur_fill_state.filling))
                )
            
            cached_fill_state = self.cache_dict.get(the_hash, None)

            if cached_fill_state is not None:
                child_fill_state.copy_from(cached_fill_state)
            else:
                # Modulo incrementation, lru_index is not the value to use
                self.lru_index += 1
                if self.lru_index == FillOptimizer.CACHE_SIZE:
                    self.cache_full = True
                    self.lru_index = 0

                if self.cache_full:
                    self.cache_dict.pop(self.hash_encounters[self.lru_index])
                    
                self.hash_encounters[self.lru_index] = the_hash
                
                # Initialize to sensible value
                child_fill_state.copy_from(cur_fill_state)
                # This updates in-place the child_filling&cinfo
                update_fill_state(child_fill_state, k)
                # Update the dict:
                self.cache_dict[the_hash] = child_fill_state.copy()
            # -------------
            
            if child_fill_state.is_full():
                # The child is completely full! Can't get better than that.
                cur_fill_order[current_depth] = k

                # Here we should not test for current_depth < max_depth:
                # if current_depth == max_depth then the depth-cut mechanism
                # broke somewhere.
                #assert current_depth < self.max_depth

                # print(f"  found a completion with {current_depth+1}",
                    # "extra steps")
                
                # -1 because want to find a strictly smaller branch
                self.max_depth = current_depth-1

                # The stopping criterion if we're going for something
                # sub-optimal
                if current_depth <= self.satisfying_depth:
                    self.max_depth = 0

                return current_depth
            
            # Else, the child is not full. Need to go on.
            # Only keep trying if we're not at max_depth.
            if current_depth < self.max_depth:
                # Setup the child_fill_order
                np.copyto(child_fill_order, cur_fill_order)
                child_fill_order[current_depth] = k
                # The best we can do with this child_fill_state:
                attempt_depth = self.recursive_set_best_fill_order(
                        k, current_depth+1
                    )
                # NB: we don't need to check that attempt_depth is less than
                # self.max_depth, this will be taken care of by the above test.
                if attempt_depth is not None:
                    best_depth = attempt_depth
                    np.copyto(cur_fill_order,child_fill_order)
        
        return best_depth

# ---------------------------------------- 
# --------- Auxiliary methods ------------
# ---------------------------------------- 

def update_fill_state(
            fill_state: FillState,
            initial_k: int,
        ):
    """This method updates the fill_state based on an initial position"""
    cycles_to_check = []
    fill_state.fill_and_append_to(initial_k, cycles_to_check)

    while len(cycles_to_check) > 0:
        l = cycles_to_check.pop()

        if fill_state.cinfo[l] == 2:
            k_to_update = fill_state.get_k_to_update(l)
            fill_state.fill_and_append_to(k_to_update, cycles_to_check)

def update_fill_state_and_get_fill_step(
            fill_state: FillState,
            initial_k: int
        ) -> FillingStep:
    """Expanded version of the above.
    This method updates the filling (fo) based on an initial position,
    and returns an description of the steps to use for this purpose
    (a list of now-known other filled positions, as well as a list of
    checks), aggregated in a FillingStep object."""
    # This method assumes that the initial_k is not filled yet,
    # and moreover that it could not have been filled by some cycles.
    # This means that the filling of the involved cycles is at most 1.
    if fill_state.filling[initial_k]:
        print(f"Error, the Alice {fill_state.sp.k_to_str[initial_k]}",
            "is already filled.")
        sys.exit()

    # Setup the initially filled position.
    filling_step = FillingStep(fill_state.sp, initial_k)

    cycles_to_check = []
    # This fills up the involved cycles
    # None of them should be trusted because they're all at most 2 now.
    fill_state.fill_and_append_to(initial_k, cycles_to_check)

    # Indicates whether a given cycle is to be trusted (True), or whether
    # it should maybe be checked for consistency in the future (False).
    # NB: re-allocating memory all the time here, but this method is called
    # a few times only (compared to the above one)
    # The cycles that are already filled should be trusted at this point
    ctrusted = [cycle_filling == 3 for cycle_filling in fill_state.cinfo]

    # STEP 1 
    # We keep looping through all the cycles to see whether or not 
    # There are updatable alices
    while len(cycles_to_check) > 0:
        l = cycles_to_check.pop()

        # If cinfo[l] == 3, this cycle is already full, which means it cannot
        # lead to an updatable Alice. Still, the cycle may not be trusted,
        # and this will be checked in the following STEP 2.
        if fill_state.cinfo[l] == 2:
            # We found an Alice to fill!
            k_to_update = fill_state.get_k_to_update(l)

            # Account for the fact that k is now fixed in the Filling
            # NB: some in there might be 3, but are nonetheless 
            # not trusted, they will go in the checks
            fill_state.fill_and_append_to(k_to_update, cycles_to_check)
                
            # We trust this l because it's the one we used/will use
            # to fill k_to_update
            ctrusted[l] = True

            # Need to update the filling step:
            cycle_ks = fill_state.sp.l_to_ks[l]
            pos_in_cycle = cycle_ks.index(k_to_update)
            k1 = cycle_ks[(pos_in_cycle + 1) % 3]
            k2 = cycle_ks[(pos_in_cycle + 2) % 3]
            filling_step.update_rules.append((k_to_update,k1,k2))
                
    # STEP 2
    # Now we have a bunch of Alices filled, but some cycles are now
    # completely filled: this gives some checks, i.e., consistency
    # requirements.
    filling_step.check_rules = [
            fill_state.sp.l_to_ks[l] 
            for l in range(fill_state.sp.n_cycles)
                if (not ctrusted[l] and fill_state.cinfo[l] == 3)
        ]
    
    return filling_step
