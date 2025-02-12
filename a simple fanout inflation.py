from __future__ import print_function
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import itertools 
from tqdm import tqdm
from sys import stderr
from orbits import identify_orbits
import utils

def eprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)

def marginal_on(p:np.ndarray, indices: tuple) -> np.ndarray:
    set3 = set(range(p.ndim))
    assert set3.issuperset(indices), "indices must be in the range 0-2"
    to_sum_over = set3.difference(indices)
    temp_arr = np.asarray(p).sum(axis=tuple(to_sum_over))
    order = tuple(np.argsort(indices))
    # print("Extra re-arranging:", order
    return temp_arr.transpose(order)

def test_distribution_with_symmetric_fanout(p_obs: np.ndarray, n:int, verbose=2) -> str:
    list_of_Alices = list(itertools.permutations(range(n), 2))
    for i in range(n):
        list_of_Alices.remove((i, (i-1)%n))
    if verbose:
        eprint("List of Alices:", list_of_Alices)

    #Discover symmetry
    canonical_order = {pair: i for i, pair in enumerate(list_of_Alices)}
    under_cylic_symmetry = [tuple(range(n)[(p+1)%n] for p in pair) for pair in list_of_Alices]
    # print(under_cylic_symmetry)
    new_order=tuple(canonical_order[pair] for pair in under_cylic_symmetry)
    # print(new_order)
    nof_Alices = len(list_of_Alices)

    p = np.asarray(p_obs)
    d = p.shape[0]
    assert p.ndim == 3, "p_obs must be a tripartite probability distibution"
    assert np.array_equiv(p.shape, d), "all parties must have the same cardinality"

    inflation_shape = nof_Alices*(d,)

    m=gp.Model()

    #Internal mVar, 8 Alices 
    # IMPOSE SYMMETRY
    if verbose:
        eprint("Discovering symmetries of inflation graph probabilities...")
    Q_infl = np.empty(shape=inflation_shape, dtype=object)
    orbits = identify_orbits(inflation_shape, new_order)
    Q_infl_raw = m.addMVar(shape=(len(orbits),), lb=0)
    if verbose:
        eprint("Constructing symmetric MVar...")
    for (var, orbit) in zip(Q_infl_raw.tolist(), orbits):
        Q_infl.flat[orbit] = var
    m.update()
    Q_infl = gp.MVar.fromlist(Q_infl)
    Q_infl.__name__ = "Q_infl"

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
    def _marginal_on(indices: tuple) -> gp._matrixapi.MVar:
        """
        This returns a marginal on the given indices, and respects the order of the indices
        """
        temp_mvar = m.addMVar(shape=(d,)*len(indices))
        all_indices = set(range(nof_Alices))
        assert all_indices.issuperset(indices), "indices must be in the range 0-2"
        to_sum_over = all_indices.difference(indices)
        m.addConstr(temp_mvar == Q_infl.sum(axis=tuple(to_sum_over)))
        order = np.argsort(indices)
        as_ndarray = np.array(temp_mvar.tolist(), dtype=object)
        return gp.MVar.fromlist(as_ndarray.transpose(order))

    # factorization
    if verbose:
        eprint("Imposing factorization constraints...")
    # TODO: use symmetries to reduce constraints
    for pair in tqdm(utils.maximal_factorizing_pairs(list_of_Alices)):
        indices1 = sorted(pair[0])
        indices2 = sorted(pair[1])

        m1 = _marginal_on(indices1)
        m2 = _marginal_on(indices2)

        m_total = _marginal_on(indices1 + indices2)
        m1_r = m1.reshape((d,)*len(indices1) + (1,)*len(indices2))
        m2_r = m2.reshape((1,)*len(indices1) + (d,)*len(indices2))
        
        m.addConstr(m1_r * m2_r == m_total)


    # injectable sets
    if verbose:
        eprint("Imposing injectable set marginal equalities...")
    # TODO: use symmetries to reduce constraints
    maximal = utils.maximal_injectable_sets(list_of_Alices)
    for clique in tqdm(maximal):
        m_injectable = _marginal_on(clique)
        if len(clique) == 3:
            p_marg = p
        else:
            p_marg = marginal_on(p, tuple(range(len(clique))))
        m.addConstr(m_injectable == p_marg)

    if verbose:
        eprint("Initiating optimization of the model...")
    m.optimize()

    # Dictionary to translate status codes
    status_dict = {
        gp.GRB.OPTIMAL: "Optimal solution found",
        gp.GRB.UNBOUNDED: "Model is unbounded",
        gp.GRB.INFEASIBLE: "Model is infeasible",
        gp.GRB.INF_OR_UNBD: "Model is infeasible or unbounded",
        gp.GRB.INTERRUPTED: "Optimization was interrupted",
        gp.GRB.TIME_LIMIT: "Time limit reached",
        gp.GRB.SUBOPTIMAL: "Suboptimal solution found",
        gp.GRB.USER_OBJ_LIMIT: "User objective limit reached",
        gp.GRB.NUMERIC: "Numerical issues",
    }

    # Print model status
    status_message = status_dict.get(m.status, f"Unknown status ({m.status})")
    print(f"Model status: {m.status} - {status_message}")

    if m.status == GRB.OPTIMAL:
        print("\nOptimal solution:")
        sol = np.asarray(Q_infl.x)
        for i in orbits[:,0]:
            val = sol.flat[i]
            if val > 1e-6:
                print(f"Q_infl[{tuple(np.unravel_index(i, inflation_shape))}]: {val}")
        #for v in m.getVars():
            #if v.x > 0:
            #    print(f"{v.varName}: {v.x}")
        print(f"Objective value: {m.objVal}")
    elif m.status == gp.GRB.INFEASIBLE:
        """
            Addition to obtain more information about the infeasibility
        """
        print("\nThe model is infeasible. Computing IIS...")
        m.computeIIS()

        print("\n--- IIS Report ---")
        
        # Print constraints that are part of the IIS
        print("Conflicting constraints:")
        for constr in m.getConstrs():
            if constr.IISConstr:  # True if this constraint is in the IIS
                print(f"  - {constr.ConstrName}")

        # Print variables that are part of the IIS
        print("\nConflicting variables:")
        for var in m.getVars():
            if var.IISLB or var.IISUB:  # True if this variable is in the IIS
                print(f"  - {var.varName} (Lower Bound: {var.IISLB}, Upper Bound: {var.IISUB})")

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

if __name__ == "__main__":
    import sys
    outcomes = 4 # int(sys.argv[1])
    inflation = 4

    test_distribution_with_symmetric_fanout(prob_agree_or_disagree(outcomes), inflation)

    # from sympy import Symbol
    # p_test = np.empty((2,2,2,2), dtype=object)
    # for indices in np.ndindex(*p_test.shape):
    #     p_test[indices] = Symbol(str(indices))

    # print(marginal_on(p_test, (0,2,1)