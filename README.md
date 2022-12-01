# recursive-sudoku

## The problem

The distribution `p = ([111]+[222]+[333])/3 + ([123]+[132]+[213]+[231]+[312]+[321])/6` is feasible in the classical triangle network: it suffices that Bob and Charlie ignore their parent source, and that the two other sources distribute 1,2,3 at random. Bob and Charlie directly output this number, while Alice output the number she receives from either source if the two values she receives agree, or else output the third outcome if they disagree.

However, is this distibution feasible in the classical triangle network if the three agents use the same strategy and the three sources are identical?

## The approach

Use possibilistic inflation. Thanks to the support (any event of the form 112, 223, etc., is forbidden in the original scenario),
this looks a lot like filling in a sudoku with custom rules.

## Examples

- Run `make test` to run a sample of unit tests

- Run `make example_algo` to see what a filling algorithm looks like

- and more to come!
