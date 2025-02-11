import numpy as np
import gurobipy as gp

list_of_Alices = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 0), (1, 3), (3, 1)]
print(list_of_Alices)

#Discover symmetry
canonical_order = {pair: i for i, pair in enumerate(list_of_Alices)}
under_cylic_symmetry = [tuple([1, 2, 3, 0][p] for p in pair) for pair in list_of_Alices]
print(under_cylic_symmetry)
new_order=tuple(canonical_order[pair] for pair in under_cylic_symmetry)
print(new_order)


nof_Alices = len(list_of_Alices)
d=3
inflation_shape = nof_Alices*(d,)

m=gp.Model()
#Internal mVar, 12 Alices A_12...A_43
Q_infl=m.addMVar(shape=inflation_shape, lb=0)
for indices in np.ndindex(*inflation_shape):
    new_indices = tuple(indices[p] for p in new_order)
    m.addConstr(Q_infl[indices] == Q_infl[new_indices])
m.optimize()