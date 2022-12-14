# recursive-sudoku

## The problem

### An asymmetric implementation of a symmetric distribution in the classical triangle network

For all `N = 3, 4, ...`, 
let `p_{eq,N} = (1/N) * \sum_{a=1}^N [aaa]` (uniform support over all outcomes
being equal) 
and `p_{neq,N} = 1/(N*(N-1)*(N-2)) * \sum_{a,b,c=1, a!=b!=c!=a}^N [abc]` 
(uniform support over no pair of outcomes being equal to each other).

The distribution `p_N = (1/N) * p_{eq,N} + (1-1/N) * p_{neq,N}` is feasible in
the classical triangle network: 
it suffices that Bob and Charlie ignore their common parent source, and that 
the two other sources distribute `1,2,...,N` uniformly. 
Bob and Charlie directly output this number, while Alice outputs the number
she receives from either source if the two values she receives agree, or else
outputs at random a number distinct from both values she receives.

This distribution `p_N` is symmetric under a `S_3` symmetry group corresponding
to exchanging all outcomes. 
However, the suggested implementation is visibly asymetric: Alice plays a very
distinct role compared to Bob and Charlie.

### Is there a corresponding symmetric implementation of this distribution?

I.e., is there a probabilistic response function `\xi(a|x,y)` where
`x, y \in [0,1]` and `a = 1,...,N` such that

`p(a,b,c) = \int\dd{x}\dd{y}\dd{z} \xi(a|x,y) * \xi(b|y,z) * \xi(c|z,x)`?

Here, we took the three classical latent sources to be uniformly distributed
over the continuous interval `[0,1]`.

## Results

### An anomaly for the three-outcome case

It turns out that the case `N = 3` is special: there, there does exist a
symmetric response function. 
Consider the three classical latent sources to be distributing a number
in `1,2,3`, and let Alice, Bob and Charlie all output the outcome that 
corresponds to `1 + (R - L mod 3)`, where `R` (`L`) is the number received
from the right (left) source. Investigating the possible source value triplets,
one finds that the resulting distribution is indeed `p_3`.

### The four-outcome case

The case `N = 4` does a better justice to our intuition: we find that there
does not exist a symmetric implementation of `p_4`.

More specifically, we find that any four-outcome distribution `p` that has
support *at least* on the events
`e1 = \{[111],[222],[123],[124],[134], [234]\}`, but *no support* on the events
`e2 = \{[112],[113], ..., [433]\}` does not admit a symmetric implementation in
the classical triangle network.

Another way to view this statement is that any symmetric strategy in the
classical triangle network that has support on the all of the events of `e1`
will necessarily have a non-trivial support on some of the events of `e2`.

This statement has been independently verified 
(here)[https://github.com/eliewolfe/mDAG-analysis/blob/main/supports_with_causal_sym.py].

## Methods

We make use of fanout inflation, and more specifically, its possibilistic
version, developed originally in the article 
(A Combinatorial Solution to Causal Compatibility)[https://doi.org/10.1515/jci-2019-0013].
The idea of inflation is to consider the thought experiment where we would
have access to multiple copies of the original sources and agents, wired in a
specific way.
The idea of the possibilistic version is to only consider the events as
being possible or impossible, but discarding the actual probabilities of
possible events.

However, the *no-support-on-`e2`* condition significantly constrains the events
that may occur in the distribution over the outcome of the agents in the
inflation. For any subgraph of the inflation that is isomorphic to the original
triangle network, we have that the corresponding three agents may only output
outcomes that jointly avoid any event in `e2`.

The *d*-separation relations of the inflation graph furthermore let us fill in
some events freely: in particular, for any two independent copies of the
original triangle network, we may plug in any event appearing in the target
distribution in one copy, and any other such event in the second copy.

#### The Sudoku analogy

Think of the inflation graph as a grid of sudoku. 
The *d*-separation relations lets us put initial values, akin to the initial
filling of a [Sudoku](https://en.wikipedia.org/wiki/Sudoku) grid.
Every copy of the original triangle network appearing in the inflation graph
is akin to a square, a row or a column of a Sudoku grid: these are subsets of
squares that are tied together by consistency relations.
The *no-support-on-`e2`* condition is the analog of the rule of the Sudoku that
states that some subsets of 9 entries must contain every number from 1 to 9.

#### The solver

The solver that we use is fairly basic. Starting from an initial outcome
filling of an inflation graph, it updates as much as possible the rest of the
entries by looking at all copies of the original triangle network. It then
picks a square in the grid, fills it with an outcome `o`, updates as much as
possible, and repeats this procedure.
If at any point an inconsistency is found, it reverts back to put `o+1`
instead of `o`, etc.

Features of the solver:

- Deterministic outcome fillings: e.g., an event `1,1,.`, with `.` meaning any
outcome in `1,2,3,4`, must be completed to `1,1,1`.
- Partial outcome fillings: an event `1,2,.` is completed to `1,2,(34)`, where
`(34)` means "`3` or `4`". Thus, internally, the solver fills the grid not with
deterministic outcomes, but rather with possibilities of outcomes.
- Greedy filling: at any step, before branching out, the solver attempts to
update as much as possible the entries of the grid.
- Branching strategy: overall, the number of steps taken to fill a grid looks
a bit like `b^d`, where `b` is the "branching factor", i.e., the number of
branches that pop up when trying to fill a partially unknown outcome in the
grid, and  `d` is the depth, i.e., the number of guesses the solver must make
to complete a grid. Minimizing `d` is obviously the way to go, and the greedy
filling attempts to get there, but when nothing obvious can be updated, the
solver goes for the smallest `b` in the grid. In other words, the solver will
guess the outcome of a square that has only two possible outcomes consistent
with the immediate filling of the grid (such a square typically exists).

#### Comparison with a SAT-solver based approach

In a (related repository)[https://github.com/eliewolfe/mDAG-analysis] that
has a greater scope than the present one, the same result has been
independently verified using a SAT solver (see 
(this file)[https://github.com/eliewolfe/mDAG-analysis/blob/main/supports_with_causal_sym.py]).
Note that the SAT solver performs much better than the present hand-crafted
solver (it would be surprising otherwise).

## Code examples

Prerequisites: `python3` (may need to edit the `PY` variable of `./makefile`),
`make`, colored ANSI output (may need to modify `src/coloring.py` in case the
escape codes do not work properly).

The following examples are available:

### `make rules` 

This example show how the "Sudoku" works at the level of a single copy of the
triangle network.
Introduction to the allowed events:
<p align="center">
  <img src="https://github.com/seemann5/recursive-sudoku/blob/main/out/out_rules.png" />
</p>
Examples of rules:
<p align="center">
  <img src="https://github.com/seemann5/recursive-sudoku/blob/main/out/out_rules_2.png" />
</p>

### `make example_filling` 

A walk-through of an explicit example of filling up an inflation (in the feasible case).
Introduction to the inflation at hand:
<p align="center">
  <img src="https://github.com/seemann5/recursive-sudoku/blob/main/out/out_example_filling.png" />
</p>
A sample of the filling process: we see that the cycle
`A(0,3) - A(3,2) - A(2,0)` is currently filled with `1,.,3`, which means that
`A(3,2)` is now only allowed to output either `2` or `4`.
<p align="center">
  <img src="https://github.com/seemann5/recursive-sudoku/blob/main/out/out_example_filling_2.png" />
</p>

### `make infeasible_grid`

To reproduce the infeasibility result:
<p align="center">
  <img src="https://github.com/seemann5/recursive-sudoku/blob/main/out/out_infeasible_grid.png" />
</p>

## Outlook

The distribution `p_4` is compatible with non-fanout inflation, even in the
presence of symmetric causal mechanisms (indeed, the maximally mixed behavior
over any ring inflation of order `4,5,...` satifies the desired cyclic
symmetry, *d*-separation, and has the appropriate two-party marginals
which are maximally mixed in `p_4`, too).

Thus, `p_4` is a symmetric distribution that:
- does not admit classical symmetric causal mechanisms in the triangle,
- yet is compatible with all the constraints of non-fanout inflation of the 
triangle.

This raises the question of the feasibility of `p_4` using symmetric
*quantum* causal mechanisms in the triangle network.

A next step in checking this would be to use the
[Inflation package](https://github.com/ecboghiu/inflation), extend it to
allow for symmetric quantum causal mechanisms, and run the relevant SDP
relaxation.
