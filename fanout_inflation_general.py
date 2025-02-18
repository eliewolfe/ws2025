from typing import List, Tuple, Union
import numpy as np
import gurobipy as gp
from tqdm import tqdm
from itertools import combinations
from utils import eprint
from gphelpers import (create_arbitrary_symmetric_mVar,
                       impose_factorization,
                       gp_project_on,
                       status_dict,
                       marginal_on,
                       impose_semi_factorization)
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
        if self.lp:
            self.m.Params.NonConvex = 0
        else:
            self.m.Params.NonConvex = -1

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
        factorizations = self.maximal_non_semiexpressible_pairs_under_symmetry
        if not self.lp:
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
        dlen = len(p_ideal)
        for i in range(dlen):
            assert p_ideal[i].ndim == 3, f"{i}th dist: p_obs must be a tripartite probability distibution"
            assert np.array_equiv(p_ideal[i].shape, self.d), f"{i}th dist: All parties must have cardinality {self.d}"
        
        self.p = p_ideal[0]
        if dlen > 1:
            try:
                assert len(lbs) == dlen, "The number of lower bounds should match the number of given probability distributions"
                min_lb = np.min(lbs)
            except TypeError:
                min_lb = lbs
            
            try:
                assert len(ubs) == dlen, "The number of lower bounds should match the number of given probability distributions"
                min_ub = np.min(ubs)
            except TypeError:
                min_ub = ubs
            
            # initialize weights (normalized)
            # TODO: print this at the end
            w = self.m.addMVar(shape=(dlen,), lb=lbs, ub=ubs, name="weights")
            self.m.addConstr(w.sum() == 1)
            
            wlist = w.tolist()
            wmin = self.m.addVar(lb=min_lb, ub=min_ub)
            self.m.addGenConstrMin(wmin, wlist)
            match objective:
                case IGO.MAXIMIZE_DIFFERENCE:
                    wdiffs = self.m.addMVar(shape=(dlen*(dlen-1)//2,), lb=0)
                    wabsdiffs = self.m.addMVar(shape=(dlen*(dlen-1)//2,), lb=0)
                    combos = combinations(range(dlen), 2)
                    for i, (v1, v2) in zip(range(dlen), combos):
                         self.m.addConstr(wdiffs[i]==w[v1] - w[v2])
                         self.m.addGenConstrAbs(wabsdiffs[i],wdiffs[i])
                    
                    wdiffmin = self.m.addVar(lb=0, ub=2)
                    self.m.addGenConstrMin(wdiffmin, wabsdiffs.tolist())

                    self.m.setObjective(wdiffmin, sense=gp.GRB.MAXIMIZE)
                case IGO.MAXIMIZE_MINIMUM:
                    self.m.setObjective(wmin, sense=gp.GRB.MAXIMIZE)
                case _:
                    if verbose:
                        eprint("No objective selected, proceeding by maximizing the minimum weight.")
                    self.m.setObjective(wmin, sense=gp.GRB.MAXIMIZE)

            self.p = np.sum(wlist[i] * p_ideal[i] for i in range(dlen))
        if maximize_visibility:
            assert not self.lp, "Nonlinear optimization is required for visibility maximization."
            v = self.m.addVar(lb=visibility_bounds[0], ub=visibility_bounds[1], name="v")
            noise = np.ones_like(p_ideal[0])/self.d**3
            self.m.setObjective(v, sense=gp.GRB.MAXIMIZE)
            self.p = p_ideal[0] * v + noise * (1-v)



        # IMPOSE injectable sets
        if self.verbose:
            eprint("Imposing injectable set marginal equalities...")
        maximal = self.maximal_injectable_but_not_semi_expressible_under_symmetry
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

        #IMPOSE semi-expressible sets
        if self.verbose:
            eprint("Imposing semi-expressible set factorization equalities...")
        semiexpressible_sets = self.maximal_semiexpressible_sets_under_symmetry
        for (indices1, indices2) in semiexpressible_sets:
            interpretation = [tuple(alices[i] for i in indices1), tuple(alices[i] for i in indices2)]
            is_expressible = (indices2 in self.all_injectable_sets)
            impose_semi_factorization(self.m, self.Q_infl, self.p, indices1, indices2, expressible=is_expressible)
            if self.verbose >= 2:
                if is_expressible:
                    eprint(f"Expressible set {[indices1, indices2]} corresponding to {interpretation}")
                else:
                    eprint(f"Semi-expressible set {[indices1, indices2]} corresponding to {interpretation}")


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

if __name__ == "__main__":
    from distlib import prob_agree, prob_all_disagree
    distribution_for_vis_analysis = prob_agree(2)
    from infgraphs import gen_fanout_inflation

    # alices=gen_fanout_inflation(5)
    # InfGraph52 = InfGraphOptimizer(alices, d=2, verbose=2, go_nonlinear=False)
    # optimal_vsi = InfGraph52.test_distribution(prob_agree(2),
    #                              maximize_visibility=True)
    # print(f"The optimal visibility is {optimal_vsi}")
    # InfGraph52.close()

    # alices=list_of_Alices(5)
    # val = test_distribution_with_symmetric_fanout(
    #     p_obs=distribution_for_vis_analysi
    #     alices=alices,
    #     verbose=2,
    #     maximize_visibility=True,
    #     visibility_bounds=(0,1))
    # print(f"The optimal visibility is {val}")

    alices=gen_fanout_inflation(3)
    InfGraph54 = InfGraphOptimizer(alices, d=2, verbose=2, go_nonlinear=False)
    InfGraph54.test_distribution(prob_agree(2), prob_all_disagree(2), objective=IGO.MAXIMIZE_DIFFERENCE)
    InfGraph54.close()