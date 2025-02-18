from typing import List, Tuple, Union
import numpy as np
import gurobipy as gp
from tqdm import tqdm
from utils import eprint
from gphelpers import create_arbitrary_symmetric_mVar, impose_factorization, gp_project_on, status_dict
from infgraphs import InfGraph, Alices
from distlib import prob_noise
from enum import Enum

class IGO(Enum):
    MAXIMIZE_DIFFERENCE = 1
    MAXIMIZE_MINIMUM = 2
    
def marginal_on(p:np.ndarray, indices: tuple) -> np.ndarray:
    set3 = set(range(p.ndim))
    assert set3.issuperset(indices), "indices must be in the range 0-2"
    to_sum_over = set3.difference(indices)
    temp_arr = np.asarray(p).sum(axis=tuple(to_sum_over))
    order = tuple(np.argsort(indices))
    # print("Extra re-arranging:", order
    return temp_arr.transpose(order)

class InfGraphOptimizer(InfGraph):
    def __init__(self, alices: Alices,
                 d: int,
                 verbose=2,
                 go_nonlinear=True):
        super().__init__(alices)
        self.verbose = verbose
        self.d = d
        self.lp = not go_nonlinear
        self.env = gp.Env(empty=False, params={'OutputFlag': bool(verbose)})
        self.m = gp.Model(env=self.env)

        #Initiate dummy distribution and no-optimization status
        self.p = prob_noise(self.d)
        self.status = 0
        self.status_message = "No optimization performed yet."

        #Initiate symmetry MVar
        self.Q_infl = create_arbitrary_symmetric_mVar(self.m,
                                                      self.d,
                                                      self.symmetry_group,
                                                      verbose=verbose)
        self.Q_infl.__name__ = "Q_infl"

        # IMPOSE FACTORIZATION
        if verbose:
            eprint("Discovering quadratic factorization relations...")
        factorizations = self.maximal_factorizing_pairs_under_symmetry
        if verbose:
            eprint("Imposing quadratic factorization constraints...")
        for (indices1, indices2) in factorizations:
            interpretation = [tuple(alices[i] for i in indices1), tuple(alices[i] for i in indices2)]
            try:
                impose_factorization(self.m, self.Q_infl, indices1, indices2)
                if verbose>=2:
                    eprint(f"Factorization {[indices1, indices2]} corresponding to {interpretation}")
            except AssertionError:
                eprint(f"!! Failed to impose factorization {[indices1, indices2]} corresponding to {interpretation}")


    def test_distribution(self, *p_ideal: np.ndarray,
                          lbs=0,
                          ubs=1,
                          objective=IGO.MAXIMIZE_MINIMUM,
                          probe_support=False,
                          maximize_visibility=False,
                          visibility_bounds=(0,1)) -> Union[str, float]:
        for i in range(dlen):
            assert p_ideal[i].ndim == 3, f"{i}th dist: p_obs must be a tripartite probability distibution"
            assert np.array_equiv(p_ideal[i].shape, self.d), f"{i}th dist: All parties must have cardinality {self.d}"
        
        self.p = p_ideal[0]
        dlen = dlen
        if dlen > 1:
            try:
                assert len(lbs) == dlen, "The number of lower bounds should match the number of given probability distributions"
            except TypeError:
                continue 
            
            try:
                assert len(ubs) == dlen, "The number of lower bounds should match the number of given probability distributions"
            except TypeError:
                continue 
            
            # initialize weights (normalized)
            # TODO: print this at the end
            w = self.m.addMVar(shape=(dlen,), lb=lbs, ub=ubs, name="weights")
            m.addConstr(w.sum() == 1)
            
            wlist = w.tolist()
            match objective:
                case IGO.MAXIMIZE_DIFFERENCE:
                    def find_min_diff(mv: gp._matrixapi.MVar) -> float:
                        mv_sorted = np.sort(mv)
                        diffs = list(map(lambda i: mv_sorted[i+1]-mv_sorted[i], range(dlen-1)))
                        return np.min(diffs)
                    
                    self.m.setObjective(find_min_diff(w), sense=gp.GRB.MAXIMIZE)
                case IGO.MAXIMIZE_MINIMUM:
                    self.m.setObjective(np.min(wlist), sense=gp.GRB.MAXIMIZE)

            self.p = np.sum(wlist[i] * p_ideal[i] for i in range(dlen))
        if maximize_visibility:
            v = self.m.addVar(lb=visibility_bounds[0], ub=visibility_bounds[1], name="v")
            noise = np.ones_like(self.p)/self.d**3
            self.m.setObjective(v, sense=gp.GRB.MAXIMIZE)
            self.p = self.p * v + noise * (1-v)

        # IMPOSE injectable sets
        if self.verbose:
            eprint("Imposing injectable set marginal equalities...")
        # TODO: use symmetries to reduce constraints
        maximal = self.maximal_injectable_sets_under_symmetry
        for clique in tqdm(maximal, disable=not self.verbose):
            interpretation = tuple(alices[i] for i in clique)
            try:
                m_injectable = gp_project_on(self.Q_infl, clique)
                if self.verbose>=2:
                    eprint(f"Imposing injectable set {clique} corresponding to {interpretation}")
                if len(clique) == 3:
                    p_marg = self.p
                else:
                    p_marg = marginal_on(self.p, tuple(range(len(clique))))
                self.m.addConstr(m_injectable == p_marg)
            except AssertionError:
                eprint(f"!! Failed to impose injectable set {clique} corresponding to {interpretation}")

        if self.verbose:
            eprint("Initiating optimization of the model...")
        self.m.optimize()

        # Print model status
        self.status = self.m.status
        self.status_message = status_dict.get(self.status, f"Unknown status ({self.status})")
        print(f"Model status: {self.status} - {self.status_message}")

        if maximize_visibility:
            try:
                obj = self.m.getObjective()
                if dlen > 1:
                    return obj.getValue(), np.asarray(w.x)
                return obj.getValue()
            except AttributeError:
                print("No objective found!")
                return self.status
        else:
            return self.status

    def close(self):
        self.__del__()

    def __del__(self):
        self.m.dispose()
        self.env.dispose()
#
#
#
# def test_distribution_with_symmetric_fanout(
#     p_obs: np.ndarray,
#     alices:List[Tuple[int,int]],
#     verbose=2,
#     maximize_visibility=False,
#     visibility_bounds=(0,1)) -> Union[str, float]:
#
#     p_ideal = np.asarray(p_obs)
#     assert p_ideal.ndim == 3, "p_obs must be a tripartite probability distibution"
#     d = p_ideal.shape[0]
#     assert np.array_equiv(p_ideal.shape, d), "all parties must have the same cardinality"
#
#     with gp.Env(empty=False, params={'OutputFlag': bool(verbose)}) as env, gp.Model(env=env) as m:
#         # Preparing for possible optimization problem
#         v = m.addVar(lb=visibility_bounds[0], ub=visibility_bounds[1], name="v")
#         noise = np.ones_like(p_ideal)/d**3
#         assert noise.sum() == 1, "Noise must sum to 1"
#         if maximize_visibility:
#             m.setObjective(v, sense=gp.GRB.MAXIMIZE)
#             p = p_ideal * v + noise * (1-v)
#         else:
#             p = p_ideal
#
#
#         # IMPOSE SYMMETRY
#         dimensional_symmetry = discover_symmetries(alices)
#         if verbose:
#             eprint("Symmetry group of order: ", len(dimensional_symmetry))
#         Q_infl = create_arbitrary_symmetric_mVar(m, d, dimensional_symmetry, verbose=verbose)
#         Q_infl.__name__ = "Q_infl"
#
#
#         # IMPOSE FACTORIZATION
#         if verbose:
#             eprint("Discovering quadratic factorization relations...")
#         factorizations = maximal_factorizing_pairs_under_symmetry(alices, dimensional_symmetry)
#         if verbose:
#             eprint("Imposing quadratic factorization constraints...")
#         for (indices1, indices2) in factorizations:
#             interpretation = [tuple(alices[i] for i in indices1), tuple(alices[i] for i in indices2)]
#             try:
#                 impose_factorization(m, Q_infl, indices1, indices2)
#                 if verbose>=2:
#                     eprint(f"Factorization {[indices1, indices2]} corresponding to {interpretation}")
#             except AssertionError:
#                 eprint(f"!! Failed to impose factorization {[indices1, indices2]} corresponding to {interpretation}")
#
#
#         # IMPOSE injectable sets
#         if verbose:
#             eprint("Imposing injectable set marginal equalities...")
#         # TODO: use symmetries to reduce constraints
#         maximal = maximal_injectable_sets_under_symmetry(alices, dimensional_symmetry)
#         for clique in tqdm(maximal, disable=not verbose):
#             interpretation = tuple(alices[i] for i in clique)
#             try:
#                 m_injectable = gp_project_on(Q_infl, clique)
#                 if verbose>=2:
#                     eprint(f"Imposing injectable set {clique} corresponding to {interpretation}")
#                 if len(clique) == 3:
#                     p_marg = p
#                 else:
#                     p_marg = marginal_on(p, tuple(range(len(clique))))
#                 m.addConstr(m_injectable == p_marg)
#
#             except AssertionError:
#                 eprint(f"!! Failed to impose injectable set {clique} corresponding to {interpretation}")
#
#
#         if verbose:
#             eprint("Initiating optimization of the model...")
#         m.optimize()
#
#         # Print model status
#         status_message = status_dict.get(m.status, f"Unknown status ({m.status})")
#         print(f"Model status: {m.status} - {status_message}")
#
#         if maximize_visibility:
#             try:
#                 obj = m.getObjective()
#                 return obj.getValue()
#             except AttributeError:
#                 print("No objective found!")
#                 return m.status
#         else:
#             return m.status
#
#         # if m.status == GRB.OPTIMAL:
#         #     print("\nOptimal solution:")
#         #     sol = np.asarray(Q_infl.x)
#         #     for i in orbits[:,0]:
#         #         val = sol.flat[i]
#         #         if val > 1e-6:
#         #             print(f"Q_infl[{tuple(np.unravel_index(i, inflation_shape))}]: {val}")
#         #     #for v in m.getVars():
#         #         #if v.x > 0:
#         #         #    print(f"{v.varName}: {v.x}")
#         #     print(f"Objective value: {m.objVal}")
#         # elif m.status == gp.GRB.INFEASIBLE:
#         #     """
#         #         Addition to obtain more information about the infeasibility
#         #     """
#         #     print("\nThe model is infeasible. Computing IIS...")
#         #     m.computeIIS()
#
#         #     print("\n--- IIS Report ---")
#
#         #     # Print constraints that are part of the IIS
#         #     print("Conflicting constraints:")
#         #     for constr in m.getConstrs():
#         #         if constr.IISConstr:  # True if this constraint is in the IIS
#         #             print(f"  - {constr.ConstrName}")
#
#         #     # Print variables that are part of the IIS
#         #     print("\nConflicting variables:")
#         #     for var in m.getVars():
#         #         if var.IISLB or var.IISUB:  # True if this variable is in the IIS
#         #             print(f"  - {var.varName} (Lower Bound: {var.IISLB}, Upper Bound: {var.IISUB})")
#
#

if __name__ == "__main__":
    from distlib import prob_agree, prob_all_disagree
    distribution_for_vis_analysis = prob_agree(2)
    from infgraphs import gen_fanout_inflation

    alices=gen_fanout_inflation(5)
    InfGraph52 = InfGraphOptimizer(alices, d=2, verbose=2)
    optimal_vsi = InfGraph52.test_distribution(prob_agree(2),
                                 maximize_visibility=True)
    print(f"The optimal visibility is {optimal_vsi}")
    InfGraph52.close()

    # alices=list_of_Alices(5)
    # val = test_distribution_with_symmetric_fanout(
    #     p_obs=distribution_for_vis_analysis,
    #     alices=alices,
    #     verbose=2,
    #     maximize_visibility=True,
    #     visibility_bounds=(0,1))
    # print(f"The optimal visibility is {val}")

    # alices=list_of_Alices(5)
    # InfGraph54 = InfGraphOptimizer(alices, d=4, verbose=2)
    # InfGraph54.test_distribution(prob_all_disagree(4))