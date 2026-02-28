from z3 import (DeclareSort, Function, BoolSort, Const, ForAll, Exists, 
                Distinct, Bool, Bools, Or, And, Not, Solver, unsat)
from warehouse_kb_agent import get_adjacent, z3_entails

"""
### Task 1: FOL domain setup
Location = DeclareSort('Location')
P  = Function('P',  Location, BoolSort())
L1 = Const('L1', Location)
ForAll(L1, P(L1))
solver = Solver()
solver.add(ForAll(L1, P(L1)))
solver.check() # Prints sat

# Continue with the guide
Location = DeclareSort('Location')
Damaged_fn  = Function('Damaged',  Location, BoolSort())
Forklift_fn = Function('Forklift', Location, BoolSort())
Creaking_fn = Function('Creaking', Location, BoolSort())
Rumbling_fn = Function('Rumbling', Location, BoolSort())
Safe_fn     = Function('Safe',     Location, BoolSort())
Adjacent_fn = Function('Adjacent', Location, Location, BoolSort())

width = 1 # Placeholder, I am assuming it is in the full implementation
height = 1
loc = {}
for x in range(1, width + 1):
    for y in range(1, height + 1):
        loc[(x, y)] = Const(f'L_{x}_{y}', Location)

### Task 2: Quantified physics rules
# Creaking rule
L  = Const('L',  Location)
Lp = Const('Lp', Location)
solver.add(ForAll(L,
    Creaking_fn(L) == Exists(Lp, And(Adjacent_fn(L, Lp), Damaged_fn(Lp)))
))

# Rumbling and safety rule
solver.add(ForAll(L,
    Rumbling_fn(L) == Exists(Lp, And(Adjacent_fn(L, Lp), Forklift_fn(Lp)))
))
solver.add(ForAll(L,
    Safe_fn(L) == And(Not(Damaged_fn(L)), Not(Forklift_fn(L)))
))

# Adjacency
for x in range(1, width + 1):
    for y in range(1, height + 1):
        adj_set = set(get_adjacent(x, y, width, height))
        for x2 in range(1, width + 1):
            for y2 in range(1, height + 1):
                if (x2, y2) in adj_set:
                    solver.add(Adjacent_fn(loc[(x, y)], loc[(x2, y2)]))
                else:
                    solver.add(Not(Adjacent_fn(loc[(x, y)], loc[(x2, y2)])))

# Domain closure 
solver.add(ForAll(L,
    Or([L == loc[(x, y)]
        for x in range(1, width + 1)
        for y in range(1, height + 1)])
))
solver.add(Distinct(list(loc.values())))
"""

# Full implementation (what we just went through)
def build_warehouse_kb_fol(width=4, height=4):
    Location = DeclareSort('Location')
    Damaged_fn  = Function('Damaged',  Location, BoolSort())
    Forklift_fn = Function('Forklift', Location, BoolSort())
    Creaking_fn = Function('Creaking', Location, BoolSort())
    Rumbling_fn = Function('Rumbling', Location, BoolSort())
    Safe_fn     = Function('Safe',     Location, BoolSort())
    Adjacent_fn = Function('Adjacent', Location, Location, BoolSort())
    loc = {}
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            loc[(x, y)] = Const(f'L_{x}_{y}', Location)
    solver = Solver()

    # Domain closure
    L = Const('L', Location)
    solver.add(ForAll(L,
        Or([L == loc[(x, y)]
            for x in range(1, width + 1)
            for y in range(1, height + 1)])
    ))
    solver.add(Distinct(list(loc.values())))

    # Adjacency (closed-world)
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            adj_set = set(get_adjacent(x, y, width, height))
            for x2 in range(1, width + 1):
                for y2 in range(1, height + 1):
                    if (x2, y2) in adj_set:
                        solver.add(Adjacent_fn(loc[(x, y)], loc[(x2, y2)]))
                    else:
                        solver.add(Not(Adjacent_fn(loc[(x, y)], loc[(x2, y2)])))
    # Quantified physics rules
    Lp = Const('Lp', Location)
    solver.add(ForAll(L,
        Creaking_fn(L) == Exists(Lp, And(Adjacent_fn(L, Lp), Damaged_fn(Lp)))
    ))
    solver.add(ForAll(L,
        Rumbling_fn(L) == Exists(Lp, And(Adjacent_fn(L, Lp), Forklift_fn(Lp)))
    ))
    solver.add(ForAll(L,
        Safe_fn(L) == And(Not(Damaged_fn(L)), Not(Forklift_fn(L)))
    ))
    # Initial knowledge
    solver.add(Safe_fn(loc[(1, 1)]))
    predicates = {
        'Creaking': Creaking_fn, 'Rumbling': Rumbling_fn,
        'Safe': Safe_fn, 'Damaged': Damaged_fn,
        'Forklift': Forklift_fn, 'Adjacent': Adjacent_fn,
    }
    return solver, loc, predicates

### Task 3: Manual reasoning
solver, loc, preds = build_warehouse_kb_fol()

# No creaking or rumbling at (1, 1)
solver.add(Not(preds['Creaking'](loc[(1, 1)])))
solver.add(Not(preds['Rumbling'](loc[(1, 1)])))
print(z3_entails(solver, preds['Safe'](loc[(1, 2)]))) 
print(z3_entails(solver, preds['Safe'](loc[(2, 1)]))) 

# Creaking but no rumbling at (1, 2)
solver.add(preds['Creaking'](loc[(1, 2)]))
solver.add(Not(preds['Rumbling'](loc[(1, 2)])))
print(z3_entails(solver, preds['Safe'](loc[(3, 1)])))
print(z3_entails(solver, preds['Safe'](loc[(2, 2)])))

# Rumbling but no creaking at (2, 1)
solver.add(preds['Rumbling'](loc[(2, 1)]))
solver.add(Not(preds['Creaking'](loc[(2, 1)])))
print(z3_entails(solver, preds['Safe'](loc[(2, 2)])))
print(z3_entails(solver, preds['Safe'](loc[(3, 1)])))
print(z3_entails(solver, preds['Safe'](loc[(1, 3)])))
# print(z3_entails(solver, preds['Safe'](loc[(0, 2)])))

# These match the conclusions that the propositional agent came to

### Task 4 in full_fol_agent.py

### Task 5: Domain closure investigation
"""
 - Results with domain closure
True
True
False
False
True  (Safe)
False (Not Safe)
False (Not Safe)

 - Results without domain closure
True
True
False
False
True
False
False

The results are the same, but the system has issues with not
being able to figure out where the forklift and damaged floor are.
It is now considering locations outside of the bounds of the 
warehouse. So, this time the false results are from uncertainty 
rather than knowing the tiles are unsafe. It still knowas that 
(2, 2) is safe since it only perceives creaking at (1, 2) and 
rumbling at (2, 1).

Removing domain closure to check (0, 2), a location outside of the
warehouse, the solver returns False because it doesn't know whether
rumbling is from (0, 2) or (1, 3).

z3 is able to do this because of the general rules that it uses.
Without domain closure, it considers any square that is adjacent, 
even those outside of the warehouse. 
"""