from typing import List, Tuple
import numpy as np
import gurobipy as gp
from utils import eprint
from orbits import identify_orbits
from tqdm import tqdm

def create_arbitrary_symmetric_mVar(m: gp._model.Model,
                                    d: int,
                                    symmetry_group: np.ndarray,
                                    verbose=2) -> gp._matrixapi.MVar:
    """
    This function creates an MVar that is symmetric under the given permutation group.
    :param m: The gurobi model
    :param d: The cardinality of each Alice's input+output combined
    :param symmetry_group: The permutation group that defines the symmetry
    :param verbose: The verbosity level
    :return: An MVar that is symmetric under the given permutation group
    """
    assert len(symmetry_group) >= 1, "Symmetry group must contain at least the identity permutation."
    assert symmetry_group.ndim == 2, "Symmetry group must be a 2D numpy array."
    nof_Alices = len(symmetry_group[0])
    mvar_shape = (d,)*nof_Alices
    flat_shape = d**nof_Alices

    if len(symmetry_group) == 1:
        # if the only permutation is the identity, then there is no need to impose symmetry
        this_MVar = m.addMVar(shape=flat_shape, lb=0)
    else:
        if verbose:
            eprint("Discovering symmetries of inflation graph probabilities...")
        this_MVar_as_ndarray = np.empty(shape=flat_shape, dtype=object)
        orbits = identify_orbits(mvar_shape, symmetry_group, verbose=verbose)
        raw_MVar = m.addMVar(shape=(len(orbits),), lb=0)
        if verbose:
            eprint("Constructing symmetric MVar...")
        for (var, orbit) in tqdm(zip(raw_MVar.tolist(), orbits), total=len(orbits), disable=not verbose):
            this_MVar_as_ndarray[orbit] = var
        m.update()
        this_MVar = gp.MVar.fromlist(this_MVar_as_ndarray.reshape(mvar_shape))
    return this_MVar

def _cyclic_symmetry_group(n: int) -> np.ndarray:
    return np.array([np.roll(np.arange(n), i) for i in range(n)], dtype=int)

def create_cyclic_symmetric_mVar(m: gp._model.Model,
                                 d: int,
                                 n: int,
                                 verbose=2) -> gp._matrixapi.MVar:
    """
    This function creates an MVar that is symmetric under the given permutation group.
    :param m: The gurobi model
    :param d: The cardinality of each Alice's input+output combined
    :param n: The degree of the cyclic symmetry group
    :param verbose: The verbosity level
    :return: An MVar that is symmetric under the given permutation group
    """
    return create_arbitrary_symmetric_mVar(m,
                                           d,
                                           _cyclic_symmetry_group(n),
                                           verbose=verbose)



def gp_project_on(q: gp._matrixapi.MVar,
                  indices: Tuple[int,...]) -> gp._matrixapi.MLinExpr:
    """
    This returns an MLinExpr on the given indices, assuming that the indices are sorted.
    """
    all_indices = set(range(q.ndim))
    assert all_indices.issuperset(indices), "indices outside of MVar dimensions"
    assert len(set(indices)) == len(indices), "indices must be unique"
    assert all(indices[i] < indices[i+1] for i in range(len(indices)-1)), "indices must be sorted"
    to_sum_over = all_indices.difference(indices)
    return q.sum(axis=tuple(to_sum_over))

def gp_marginal_on(m: gp._model.Model,
                   q: gp._matrixapi.MVar,
                   indices: Tuple[int,...]) -> gp._matrixapi.MVar:
    """
    This returns an MVar associated with the marginal on the given indices.
    The indices need not be sorted.
    """
    sorted_indices = tuple(sorted(indices))
    projection = gp_project_on(q, sorted_indices)
    temp_MVar = m.addMVar(shape=tuple(q.shape[i] for i in sorted_indices))
    m.addConstr(temp_MVar == projection)
    if np.array_equal(sorted_indices, indices):
        return temp_MVar
    else:
        order = np.argsort(indices)
        temp_MVar_as_ndarray = np.array(temp_MVar.tolist(), dtype=object)
        return gp.MVar.fromlist(temp_MVar_as_ndarray.transpose(order))


def impose_factorization(m: gp._model.Model,
                         q: gp._matrixapi.MVar,
                         indices1: Tuple[int,...],
                         indices2: Tuple[int,...]) -> None:
    """
    This function imposes factorization constraints on the given MVar.
    :param m: The gurobi model
    :param q: The MVar to impose factorization constraints on
    :param indices1: The indices associated with one of the factors
    :param indices2: The indices associated with the other factor
    :return: None
    """
    m1 = gp_marginal_on(m, q, indices1)
    m1 = m1.reshape(m1.shape + (1,) * len(indices2))
    m2 = gp_marginal_on(m, q, indices2)
    m2 = m2.reshape((1,) * len(indices1) + m2.shape)
    m_total = gp_marginal_on(m, q, indices1 + indices2)
    m.addConstr(m1 * m2 == m_total)
    return None


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

if __name__ == "__main__":
    print(_cyclic_symmetry_group(5))