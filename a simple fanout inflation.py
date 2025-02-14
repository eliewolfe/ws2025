from typing import List, Tuple, Union
import numpy as np
import gurobipy as gp
from tqdm import tqdm
from utils import eprint, discover_symmetries, maximal_injectable_sets_under_symmetry, maximal_factorizing_pairs_under_symmetry
from gphelpers import create_arbitrary_symmetric_mVar, impose_factorization, gp_project_on, status_dict


def marginal_on(p:np.ndarray, indices: tuple) -> np.ndarray:
    set3 = set(range(p.ndim))
    assert set3.issuperset(indices), "indices must be in the range 0-2"
    to_sum_over = set3.difference(indices)
    temp_arr = np.asarray(p).sum(axis=tuple(to_sum_over))
    order = tuple(np.argsort(indices))
    # print("Extra re-arranging:", order
    return temp_arr.transpose(order)

def test_distribution_with_symmetric_fanout(
    p_obs: np.ndarray, 
    alices:List[Tuple[int,int]], 
    verbose=2,
    maximize_visibility=False,
    visibility_bounds=(0,1)) -> Union[str, float]:

    p_ideal = np.asarray(p_obs)
    assert p_ideal.ndim == 3, "p_obs must be a tripartite probability distibution"
    d = p_ideal.shape[0]
    assert np.array_equiv(p_ideal.shape, d), "all parties must have the same cardinality"

    with gp.Env(empty=False, params={'OutputFlag': bool(verbose)}) as env, gp.Model(env=env) as m:
        # Preparing for possible optimization problem
        v = m.addVar(lb=visibility_bounds[0], ub=visibility_bounds[1], name="v")
        noise = np.ones_like(p_ideal)/d**3
        assert noise.sum() == 1, "Noise must sum to 1"
        if maximize_visibility:
            m.setObjective(v, sense=gp.GRB.MAXIMIZE)
            p = p_ideal * v + noise * (1-v)
        else:
            p = p_ideal


        # IMPOSE SYMMETRY
        dimensional_symmetry = discover_symmetries(alices)
        if verbose:
            eprint("Symmetry group of order: ", len(dimensional_symmetry))
        Q_infl = create_arbitrary_symmetric_mVar(m, d, dimensional_symmetry, verbose=verbose)
        Q_infl.__name__ = "Q_infl"


        # IMPOSE FACTORIZATION
        if verbose:
            eprint("Discovering quadratic factorization relations...")
        factorizations = maximal_factorizing_pairs_under_symmetry(alices, dimensional_symmetry)
        if verbose:
            eprint("Imposing quadratic factorization constraints...")
        for (indices1, indices2) in factorizations:
            interpretation = [tuple(alices[i] for i in indices1), tuple(alices[i] for i in indices2)]
            try:
                impose_factorization(m, Q_infl, indices1, indices2)
                if verbose>=2:
                    eprint(f"Factorization {[indices1, indices2]} corresponding to {interpretation}")
            except AssertionError:
                eprint(f"!! Failed to impose factorization {[indices1, indices2]} corresponding to {interpretation}")


        # IMPOSE injectable sets
        if verbose:
            eprint("Imposing injectable set marginal equalities...")
        # TODO: use symmetries to reduce constraints
        maximal = maximal_injectable_sets_under_symmetry(alices, dimensional_symmetry)
        for clique in tqdm(maximal, disable=not verbose):
            interpretation = tuple(alices[i] for i in clique)
            try:
                m_injectable = gp_project_on(Q_infl, clique)
                if verbose>=2:
                    eprint(f"Imposing injectable set {clique} corresponding to {interpretation}")
                if len(clique) == 3:
                    p_marg = p
                else:
                    p_marg = marginal_on(p, tuple(range(len(clique))))
                m.addConstr(m_injectable == p_marg)

            except AssertionError:
                eprint(f"!! Failed to impose injectable set {clique} corresponding to {interpretation}")


        if verbose:
            eprint("Initiating optimization of the model...")
        m.optimize()

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
    from distlib import prob_agree, prob_all_disagree
    distribution_for_vis_analysis = prob_agree(2)
    from infgraphs import gen_fanout_inflation as list_of_Alices

    alices=list_of_Alices(5)
    val = test_distribution_with_symmetric_fanout(
        p_obs=distribution_for_vis_analysis,
        alices=alices,
        verbose=2,
        maximize_visibility=True,
        visibility_bounds=(0,1))
    print(f"The optimal visibility is {val}")

    # alices=list_of_Alices(4)
    # test_distribution_with_symmetric_fanout(
    #     p_obs=prob_all_disagree(4),
    #     alices=alices,
    #     verbose=2,
    #     maximize_visibility=False)