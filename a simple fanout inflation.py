from typing import List, Tuple, Set
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import itertools 
from tqdm import tqdm
from sys import stderr
from orbits import identify_orbits
import utils
# from nonfanout_inflation_general import prob_twosame
from utils import eprint


def marginal_on(p:np.ndarray, indices: tuple) -> np.ndarray:
    set3 = set(range(p.ndim))
    assert set3.issuperset(indices), "indices must be in the range 0-2"
    to_sum_over = set3.difference(indices)
    temp_arr = np.asarray(p).sum(axis=tuple(to_sum_over))
    order = tuple(np.argsort(indices))
    # print("Extra re-arranging:", order
    return temp_arr.transpose(order)

def list_of_Alices(n: int, verbose=2) -> List[Tuple[int,int]]:
    alices = list(itertools.permutations(range(n), 2))
    for i in range(n):
        alices.remove((i, (i-1)%n))
    if verbose:
        eprint("List of Alices:", alices)
    return alices

def test_distribution_with_symmetric_fanout(
    p_obs: np.ndarray, 
    alices:List[Tuple[int,int]], 
    verbose=2,
    maximize_visibility=False,
    visibility_bounds=(0,1)) -> str:

    #Discover symmetry
    dimensional_symmetry = utils.discover_symmetries(alices)

    p_ideal = np.asarray(p_obs)
    assert p_ideal.ndim == 3, "p_obs must be a tripartite probability distibution"
    d = p_ideal.shape[0]
    assert np.array_equiv(p_ideal.shape, d), "all parties must have the same cardinality"

    nof_Alices = len(alices)
    inflation_shape = nof_Alices*(d,)
    inflation_flat_shape = d**nof_Alices
    # n = max(max(pair) for pair in alices) + 1


    with gp.Env(empty=False, params={'OutputFlag': verbose}) as env, gp.Model(env=env) as m:


        v = m.addVar(lb=visibility_bounds[0], ub=visibility_bounds[1], name="v")
        noise = np.ones_like(p_ideal)/inflation_flat_shape
        if maximize_visibility:
            m.setObjective(v, sense=gp.GRB.MAXIMIZE)
            p = p_ideal * v + noise * (1-v)
        else:
            p = p_ideal

        # IMPOSE SYMMETRY
        if len(dimensional_symmetry) == 1:
            # if the only permutation is the identity, then there is no need to impose symmetry
            Q_infl = m.addMVar(shape=inflation_shape, lb=0)
        else:
            if verbose:
                eprint("Discovering symmetries of inflation graph probabilities...")
            Q_infl = np.empty(shape=inflation_shape, dtype=object)
            orbits = identify_orbits(inflation_shape, dimensional_symmetry, verbose=verbose)
            Q_infl_raw = m.addMVar(shape=(len(orbits),), lb=0)
            if verbose:
                eprint("Constructing symmetric MVar...")
            for (var, orbit) in tqdm(zip(Q_infl_raw.tolist(), orbits), total=len(orbits), disable=not verbose):
                Q_infl.flat[orbit] = var
            m.update()
            Q_infl = gp.MVar.fromlist(Q_infl)
            Q_infl.__name__ = "Q_infl"

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
        for pair in tqdm(utils.maximal_factorizing_pairs(alices), disable=not verbose):
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
        maximal = utils.maximal_injectable_sets(alices)
        for clique in tqdm(maximal, disable=not verbose):
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
            GRB.OPTIMAL: "Optimal solution found",
            GRB.UNBOUNDED: "Model is unbounded",
            GRB.INFEASIBLE: "Model is infeasible",
            GRB.INF_OR_UNBD: "Model is infeasible or unbounded",
            GRB.INTERRUPTED: "Optimization was interrupted",
            GRB.TIME_LIMIT: "Time limit reached",
            GRB.SUBOPTIMAL: "Suboptimal solution found",
            GRB.USER_OBJ_LIMIT: "User objective limit reached",
            GRB.NUMERIC: "Numerical issues",
        }

        # Print model status
        status_message = status_dict.get(m.status, f"Unknown status ({m.status})")
        print(f"Model status: {m.status} - {status_message}")

        if maximize_visibility:
            try:
                obj = m.getObjective()
                return obj.getValue()
            except AttributeError:
                print("No objective found!")
                return m.status
        else:
            return m.status

        # if m.status == GRB.OPTIMAL:
        #     print("\nOptimal solution:")
        #     sol = np.asarray(Q_infl.x)
        #     for i in orbits[:,0]:
        #         val = sol.flat[i]
        #         if val > 1e-6:
        #             print(f"Q_infl[{tuple(np.unravel_index(i, inflation_shape))}]: {val}")
        #     #for v in m.getVars():
        #         #if v.x > 0:
        #         #    print(f"{v.varName}: {v.x}")
        #     print(f"Objective value: {m.objVal}")
        # elif m.status == gp.GRB.INFEASIBLE:
        #     """
        #         Addition to obtain more information about the infeasibility
        #     """
        #     print("\nThe model is infeasible. Computing IIS...")
        #     m.computeIIS()

        #     print("\n--- IIS Report ---")

        #     # Print constraints that are part of the IIS
        #     print("Conflicting constraints:")
        #     for constr in m.getConstrs():
        #         if constr.IISConstr:  # True if this constraint is in the IIS
        #             print(f"  - {constr.ConstrName}")

        #     # Print variables that are part of the IIS
        #     print("\nConflicting variables:")
        #     for var in m.getVars():
        #         if var.IISLB or var.IISUB:  # True if this variable is in the IIS
        #             print(f"  - {var.varName} (Lower Bound: {var.IISLB}, Upper Bound: {var.IISUB})")



if __name__ == "__main__":
    from distlib import prob_agree
    distribution_for_vis_analysis = prob_agree(2)
    inflation = 5

    
    # print("\n ITERATIONS:")
    # print(find_solution(outcomes, inflation, 0.01, [0.4, 0.5]))
    val = test_distribution_with_symmetric_fanout(
        p_obs=distribution_for_vis_analysis,
        alices=list_of_Alices(inflation), 
        verbose=0, 
        maximize_visibility=True, 
        visibility_bounds=(0,1))

    # print(f"The optimal visibility is {val}")

    # print(prob_noisy_GHZ(3, 0.5))

    # from sympy import Symbol
    # p_test = np.empty((2,2,2,2), dtype=object)
    # for indices in np.ndindex(*p_test.shape):
    #     p_test[indices] = Symbol(str(indices))

    # print(marginal_on(p_test, (0,2,1)