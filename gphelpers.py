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


def _test_sorted(indices: Tuple[int,...]) -> bool:
    return all(indices[i] < indices[i+1] for i in range(len(indices)-1))


def gp_project_on(q: gp._matrixapi.MVar,
                  indices: Tuple[int,...]) -> gp._matrixapi.MLinExpr:
    """
    This returns an MLinExpr on the given indices, assuming that the indices are sorted.
    """
    all_indices = set(range(q.ndim))
    assert all_indices.issuperset(indices), "indices outside of MVar dimensions"
    assert _test_sorted(indices), "indices must be sorted"
    to_sum_over = all_indices.difference(indices)
    return q.sum(axis=tuple(to_sum_over))

def marginal_on(p: np.ndarray, indices: tuple) -> np.ndarray:
    set3 = set(range(p.ndim))
    assert set3.issuperset(indices), "indices must be in the range 0-2"
    to_sum_over = set3.difference(indices)
    temp_arr = np.asarray(p).sum(axis=tuple(to_sum_over))
    order = tuple(np.argsort(indices))
    # print("Extra re-arranging:", order
    return temp_arr.transpose(order)


def shapes_for_factorization(initial_shape: Tuple[int,...],
                             indices1: Tuple[int,...],
                             indices2: Tuple[int,...]) -> Tuple[Tuple[int,...], Tuple[int,...], Tuple[int,...]]:
    """
    This function returns the shapes of the factorized MVars.
    :param initial_shape: The shape of the initial MVar
    :param indices1: The indices associated with one of the factors
    :param indices2: The indices associated with the other factor
    :return: The shapes of the factorized MVars
    """
    combined_indices = tuple(sorted(indices1+indices2))
    where1 = np.isin(combined_indices, indices1)
    where2 = np.logical_not(where1)
    assert np.array_equal(where2, np.isin(combined_indices, indices2)), "Sanity check failed"
    template1 = np.ones(len(combined_indices), dtype=int)
    template2 = template1.copy()
    natural_shape_1 = tuple(np.take(initial_shape, indices1).flat)
    natural_shape_2 = tuple(np.take(initial_shape, indices2).flat)
    template1[where1] = natural_shape_1
    template2[where2] = natural_shape_2
    shape_1 = tuple(template1.flat)
    shape_2 = tuple(template2.flat)
    return (shape_1, shape_2, combined_indices)

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
    shape_1, shape_2, combined_indices = shapes_for_factorization(q.shape, indices1, indices2)
    natural_shape_1 = tuple(np.take(q.shape, indices1).flat)
    natural_shape_2 = tuple(np.take(q.shape, indices2).flat)
    mv1 = m.addMVar(shape=shape_1)
    m.addConstr(mv1.reshape(natural_shape_1) == gp_project_on(q, indices1))
    mv2 = m.addMVar(shape=shape_2)
    m.addConstr(mv2.reshape(natural_shape_2) == gp_project_on(q, indices2))
    m_combined = gp_project_on(q, combined_indices)
    m.addConstr(mv1 * mv2 == m_combined)
    return None


def impose_semi_factorization(m: gp._model.Model,
                              q: gp._matrixapi.MVar,
                              p: np.ndarray,
                              indices1: Tuple[int, ...],
                              indices2: Tuple[int, ...],
                              expressible=False) -> None:
    """
    This function imposes factorization constraints on the given MVar.
    :param m: The gurobi model
    :param q: The MVar to impose factorization constraints on
    :param indices1: The indices associated with the injectable factors
    :param indices2: The indices associated with the other factor
    :return: None
    """
    shape_1, shape_2, combined_indices = shapes_for_factorization(q.shape, indices1, indices2)
    if len(indices1) == 3:
        mv1 = p
    else:
        mv1 = marginal_on(p, tuple(range(len(indices1))))
        mv1 = mv1.reshape(shape_1)
    if expressible:
        if len(indices2) == 3:
            mv2 = p
        else:
            mv2= marginal_on(p, tuple(range(len(indices2))))
            mv2 = mv2.reshape(shape_2)
    else:
        natural_shape_2 = tuple(np.take(q.shape, indices2).flat)
        mv2 = m.addMVar(shape=shape_2)
        m.addConstr(mv2.reshape(natural_shape_2) == gp_project_on(q, indices2))
    m_combined = gp_project_on(q, combined_indices)
    LHS = mv1 * mv2
    print(f"Debugging: Is expressible? {expressible}, {type(LHS)}, {LHS.shape}, {type(m_combined)}, {m_combined.shape}")
    m.addConstr(LHS == m_combined)
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