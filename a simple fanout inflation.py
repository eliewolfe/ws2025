from __future__ import print_function
import numpy as np
import gurobipy as gp
import itertools 
from tqdm import tqdm
from sys import stderr
from orbits import identify_orbits

def eprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)


list_of_Alices = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 0), (1, 3), (3, 1)]
eprint("List of Alices:", list_of_Alices)

#Discover symmetry
canonical_order = {pair: i for i, pair in enumerate(list_of_Alices)}
under_cylic_symmetry = [tuple([1, 2, 3, 0][p] for p in pair) for pair in list_of_Alices]
# print(under_cylic_symmetry)
new_order=tuple(canonical_order[pair] for pair in under_cylic_symmetry)
# print(new_order)
nof_Alices = len(list_of_Alices)

def marginal_on(p:np.ndarray, indices: tuple) -> np.ndarray:
    set3 = set(range(3))
    assert set3.issuperset(indices), "indices must be in the range 0-2"
    to_sum_over = set(range(3)).difference(indices)
    return np.asarray(p).sum(axis=tuple(to_sum_over))

def test_distribution_with_symmetric_fanout(p_obs: np.ndarray, verbose=2) -> str:
    p = np.asarray(p_obs)
    d = p.shape[0]
    assert p.ndim == 3, "p_obs must be a tripartite probability distibution"
    assert np.array_equiv(p.shape, d), "all parties must have the same cardinality"

    inflation_shape = nof_Alices*(d,)

    m=gp.Model()

    #Internal mVar, 8 Alices 
    # IMPOSE SYMMETRY
    if verbose:
        eprint("Imposing symmetries...")
    Q_infl = np.empty(shape=inflation_shape, dtype=object)
    orbits = identify_orbits(inflation_shape, new_order)
    Q_infl_raw = m.addMVar(shape=(len(orbits),), lb=0)
    for (var, orbit) in zip(Q_infl_raw.tolist(), orbits):
        Q_infl.flat[orbit] = var
    m.update()
    Q_infl = gp.MVar.fromlist(Q_infl)

    # total_nof_vars = d**nof_Alices
    # orbit_template = np.zeros((total_nof_vars,), dtype=int)
    # orbit_instance = 1
    # for i in range(total_nof_vars):
    #     if orbit_template[i] == 0:
    #         #time for a new orbit!
    #         orbit_template[i] == orbit_instance
    #
    #         orbit_template[i] = orbit_instance
    #         orbit_instance += 1
    #         for sym in np.array(symmetries, dtype=int):
    #             new_indices = np.array(i, dtype=int)[sym]
    #             new_indices = tuple(new_indices.flat)
    #             orbit_template[new_indices] = orbit_template[i]
    # replacement_indices = np.arange(total_nof_vars).reshape(inflation_shape)
    # alternative_indices = replacement_indices.transpose(new_order).ravel()
    # Q_inf_flat = Q_infl.reshape((total_nof_vars))
    # m.addConstr(Q_inf_flat == Q_inf_flat[alternative_indices])
    #
    def marginal_on_internal(indices: tuple) -> gp._matrixapi.MVar:
        temp_mvar = m.addMVar(shape=(d,)*len(indices))
        to_sum_over = set(range(nof_Alices)).difference(indices)
        m.addConstr(temp_mvar == Q_infl.sum(axis=tuple(to_sum_over)))
        return temp_mvar

    # factorization
    mA01 = marginal_on_internal([0])
    mA23 = marginal_on_internal([2])
    mA0123 = marginal_on_internal([0, 2])
    m.addConstr(mA01.reshape((d,1))*mA23.reshape((1,d)) == mA0123)

    mA02and20 = marginal_on_internal([4,5])
    mA13and31 = marginal_on_internal([6,7])
    mA02and20and13and31 = marginal_on_internal([4,5,6,7])
    m.addConstr(mA02and20.reshape((d,d,1,1))*mA13and31.reshape((1,1,d,d)) == mA02and20and13and31)

    # injectable sets
    all_injectable_sets_as_Alices = [
        [(0,1), (1,2), (2,0)],
        [(0,1), (1,3), (3,0)]
    ]
    for injectable_set_Alices in tqdm(all_injectable_sets_as_Alices):
        injectable_set_indices = tuple(canonical_order[pair] for pair in injectable_set_Alices)
        m_injectable = marginal_on_internal(injectable_set_indices)
        m.addConstr(m_injectable == p)

    m.optimize()
    print("Model status:", m.status)
    

def prob_agree_or_disagree(n: int) -> np.ndarray:
    prob = np.zeros((n, n, n))
    agree_events = n
    disagree_events = n*(n-1)*(n-2)
    n_events = agree_events + disagree_events

    for i in range(n):
        prob[i, i, i] = 1/n_events

    for (i,j,k) in itertools.permutations(range(n), 3):
        prob[i, j, k] = 1/n_events

    assert prob.sum() == 1, "probabilities must sum to 1"
    return prob


test_distribution_with_symmetric_fanout(prob_agree_or_disagree(4))