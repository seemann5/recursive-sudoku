# recursive-sudoku

## The problem

The distribution `p = ([111]+[222]+[333])/3 + ([123]+[132]+[213]+[231]+[312]+[321])/6` is feasible in the classical triangle network: it suffices that Bob and Charlie ignore their parent source, and that the two other sources distribute 1,2,3 at random. Bob and Charlie directly output this number, while Alice output the number she receives from either source if the two values she receives agree, or else output the third outcome if they disagree.

However, is this distibution feasible in the classical triangle network if the three agents use the same strategy and the three sources are identical? It turns out it is!

## The approach

Use possibilistic inflation. Thanks to the support (any event of the form 112, 223, etc., is forbidden in the original scenario),
this looks a lot like filling in a sudoku with custom rules.


## An answer for the three-outcome case

Prerequisites: python3 and colored ANSI output in a shell.
(May want to edit `PY` variable of `three_outcomes/makefile`.)

After `cd three_outcomes`, one can run the following:

- `make test` to run a sample of unit tests,

- `make example_algo` to see the sort of internal algorithm that the 
solver uses,

- `make example_fill` and `make example_fill_alt` to see what filling a small
inflation looks like,

- `make` (or `make default`) to run the most interesting case: a guided
example of a large inflation that allows to read off a strategy that does
work out for the target distribution.

## Outlook

The three-outcome distribution we looked at turns out the be symmetrically
achievable in the triangle network. 

We can still define other candidate distributions, such as the four-outcome
analog with only `[111]` (all eq) and `[123]` (all neq).

This will (probably) result in a new four_outcome folder. The solver
will have to be less efficient, because the depth impact of a guess
now depends on more than the guess position - indeed, a `[12x]` cycle can be
completed into `[123]` or `[124]`.
