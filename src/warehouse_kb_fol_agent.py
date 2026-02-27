from z3 import DeclareSort, Function, BoolSort, Const, ForAll, Exists, Distinct, Bool, Bools, Or, And, Not, Solver, unsat

Location = DeclareSort('Location')
Damaged_fn  = Function('Damaged',  Location, BoolSort())
Forklift_fn = Function('Forklift', Location, BoolSort())
Creaking_fn = Function('Creaking', Location, BoolSort())
Rumbling_fn = Function('Rumbling', Location, BoolSort())
Safe_fn     = Function('Safe',     Location, BoolSort())
Adjacent_fn = Function('Adjacent', Location, Location, BoolSort())

# Task 1: Test
P  = Function('P',  Location, BoolSort())
L1 = Const('L1', Location)
ForAll(L1, P(L1))
solver = Solver()
solver.add(ForAll(L1, P(L1)))
print(solver.check())