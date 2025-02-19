# from collections.abc import Callable
from typing import Protocol, Union
from functools import partial
import numpy as np
import gurobipy as gp
from collections import defaultdict
from infgraphs import InfGraph
from tqdm import tqdm
from utils import eprint
from orbits import identify_orbits
from gc import collect
from gphelpers import (create_arbitrary_symmetric_mVar,
                       impose_factorization,
                       gp_project_on,
                       status_dict,
                       marginal_on,
                       impose_semi_factorization)

class SupportTester(Protocol):
    def __call__(fakeSelf,
                 p: np.ndarray, *,
                 verbose: int = 2,
                 go_nonlinear: bool = False) -> Union[str, np.ndarray]:
        pass

def support_tester(g: InfGraph)-> SupportTester:
    return partial(_test_support, g)

def _test_support(g: InfGraph,
                  p: np.ndarray,
                  verbose=2,
                  go_nonlinear=True) -> Union[str, np.ndarray]:
    """
    This function tests whether a given support is compatible with the given inflation graph.
    :param g: The InfGraph
    :param p: The support (as a distribution)
    :return: The result of the Gurobi optimization
    """
    p_shape = p.shape
    assert len(p_shape) == 3, "Support must be a 3-tensor."
    d = p_shape[0]
    assert np.array_equiv(p_shape, d), "Support must be have consistent dimensions."

    lp = not go_nonlinear
    env = gp.Env(empty=False, params={'OutputFlag': bool(verbose)})
    m = gp.Model(env=env)

    if lp:
        m.Params.NonConvex = 0
    else:
        m.Params.NonConvex = -1


    forbidden_bipartite_events = set([idxs for (idxs, val) in np.ndenumerate(p.sum(axis=0)) if not val])
    forbidden_tripartite_events = set([idxs for (idxs, val) in np.ndenumerate(p) if not val])
    forbidden_tripartite_events = [event for event in forbidden_tripartite_events if event[1:] not in forbidden_bipartite_events]

    mvar_shape = (d,) * g.nof_alices
    orbits = identify_orbits(mvar_shape, g.symmetry_group, verbose=verbose)
    if verbose:
        eprint(f"Length of orbits: {len(orbits)}")
    unravelled_orbits = np.stack(np.unravel_index(orbits, mvar_shape), axis=-1)
    del orbits
    collect(generation=2)

    previously_forbidden_events = set([])
    max_inj_sets_dict = defaultdict(list)
    for inj_set in g.maximal_injectable_sets_under_symmetry:
        max_inj_sets_dict[len(inj_set)].append(inj_set)
    for n in sorted(max_inj_sets_dict.keys()):
        inj_sets = max_inj_sets_dict[n]
        p_marg = marginal_on(p, tuple(range(n)))
        forbidden_events = set([idxs for (idxs, val) in np.ndenumerate(p_marg) if not val])
        forbidden_events = set(filter(lambda event: event[1:] not in previously_forbidden_events, forbidden_events))
        previously_forbidden_events.update(forbidden_events)
        picklist = np.ones(len(unravelled_orbits), dtype=bool)
        sub_orbits = unravelled_orbits[:, :, inj_sets]
        assert sub_orbits.ndim == 4, "The sub_orbits must be 4-dimensional."
        old_shape = sub_orbits.shape
        new_shape = (old_shape[0], old_shape[1]*old_shape[2], old_shape[3])
        sub_orbits = sub_orbits.reshape(new_shape)
        for i, sub_orbit in enumerate(sub_orbits):
            if not forbidden_events.isdisjoint(map(tuple, sub_orbit)):
                picklist[i] = False
        unravelled_orbits = unravelled_orbits[picklist]
        if verbose:
            eprint(f"Length of orbits after accounting for forbidden length-{n} events: {len(unravelled_orbits)}")

    nof_nonzero_orbits = len(unravelled_orbits)
    raw_MVar = m.addMVar(shape=(len(nof_nonzero_orbits),), lb=0)
    mVar_template = np.zeros(shape=mvar_shape, dtype=object)
    for (var, unravelled_orbit) in tqdm(zip(raw_MVar.tolist(), unravelled_orbits), total=len(unravelled_orbits), disable=not verbose):
        for unravelled_event in unravelled_orbit:
            mVar_template[tuple(unravelled_event)] = var
    m.update()
    q_inf = gp.MVar.fromlist(mVar_template)



