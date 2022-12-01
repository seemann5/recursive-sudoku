# recursive-sudoku

## The problem

Is the distribution
![equation](https://latex.codecogs.com/svg.image?p&space;=&space;\frac{1}{3}([111]&space;&plus;&space;[222]&space;&plus;&space;[333])&space;&plus;&space;\frac{1}{6}([123]&space;&plus;&space;[132]&space;&plus;&space;[213]&space;&plus;&space;[231]&space;&plus;&space;[312]&space;&plus;&space;[321]))
feasible in the classical triangle network if the three agents use the same strategy and the three sources are identical?

## The approach

Use possibilistic inflation. Thanks to the support (any event of the form 112, 223, etc., is forbidden in the original scenario),
this looks a lot like filling in a sudoku with custom rules.

## Examples

- Run `make test` to run a sample of unit tests

- Run `make example_algo` to see what a filling algorithm looks like

- and more to come
