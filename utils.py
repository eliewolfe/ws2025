from __future__ import print_function
from typing import List, Tuple, Set
import itertools
from collections import deque
import numpy as np
from more_itertools import powerset
from sys import stderr
from operator import itemgetter
### PUBLIC FUNCTIONS ###

def eprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)


def maximal_injectable_sets(alices: List[Tuple[int, int]]) -> List[Tuple[int,...]]:
  return _all_and_maximal_cliques(_injection_graph(alices), isolate_maximal=True)[1]

def maximal_injectable_sets_under_symmetry(alices: List[Tuple[int, int]],
                                           symmetry_group: np.ndarray) -> List[Tuple[int,...]]:
  return _unique_vecs_under_symmetry(
    maximal_injectable_sets(alices),
    symmetry_group)


def maximal_factorizing_pairs(alices: List[Tuple[int, int]]) -> List[Tuple[Tuple[int,...], Tuple[int,...]]]:
  #First let's obtain all pairs of factorizing sets based on sources, then we'll filter for maximality.
  factorizing_source_pairs = partitions_with_min_size(alices)
  canonical_dict = {alice: i for i, alice in enumerate(alices)}
  factorizing_pairs = []
  for source_partition in factorizing_source_pairs:
    source_set1 = set(source_partition[0])
    source_set2 = set(source_partition[1])
    first_alices_idxs = sorted(canonical_dict[alice] for alice in alices if source_set1.issuperset(alice))
    second_alices_idxs = sorted(canonical_dict[alice] for alice in alices if source_set2.issuperset(alice))
    factorizing_pairs.append((tuple(first_alices_idxs), tuple(second_alices_idxs)))
  return list(_hypergraph_full_cleanup(factorizing_pairs))


def maximal_factorizing_pairs_under_symmetry(alices: List[Tuple[int, int]],
                                           symmetry_group: np.ndarray) -> List[Tuple[Tuple[int,...], Tuple[int,...]]]:
  return _unique_factorizations_under_symmetry(
    maximal_factorizing_pairs(alices),
    symmetry_group)


def discover_symmetries(alices: List[Tuple[int, int]]) -> np.ndarray:
  canonical_order = {pair: i for i, pair in enumerate(alices)}
  flat_indices = tuple(itertools.chain.from_iterable(alices))
  nof_flat_indices = len(flat_indices)
  all_copy_indices = sorted(set(itertools.chain.from_iterable(alices)))
  assert np.array_equal(all_copy_indices, np.arange(len(all_copy_indices))), "Copy indices must be contiguous."
  discovered_symmetries = []
  for perm_candidate in itertools.permutations(all_copy_indices):
    permuted_flat_indices = itemgetter(*flat_indices)(perm_candidate)
    relabelled_alices = [permuted_flat_indices[i:i+2] for i in range(0, nof_flat_indices, 2)]
    try:
      discovered_symmetries.append([canonical_order[pair] for pair in relabelled_alices])
    except KeyError:
      continue
  assert len(discovered_symmetries) >= 1, "At least the identity permutation should be found."
  return np.array(discovered_symmetries, dtype=int)


### PRIVATE FUNCTIONS ###

def partitions_with_min_size(alices: List[Tuple[int, int]], min_size=2):
  """
  Obtain all orderless partitions where both partitions have at least `min_size` elements.
  WLOG, the first partition will be the one with the zero element.
  """
  range_set = set(itertools.chain.from_iterable(alices))
  first_element = range_set.pop()
  pairs = []
  for size1 in range(min_size-1, len(range_set) - min_size + 1):
    for subset1 in itertools.combinations(range_set, size1):
      subset1 = set(subset1)
      subset1.add(first_element)
      remaining_elements = range_set.difference(subset1)
      pairs.append((subset1, remaining_elements))
  return pairs


def _unique_vecs_under_symmetry(vecs: List[Tuple[int,...]], symmetry_group: np.ndarray) -> List[Tuple[int,...]]:
  pending_list = set(map(tuple, vecs))
  clean_list = set([])
  while pending_list:
    new_vec = pending_list.pop()
    clean_list.add(new_vec)
    variants = set(tuple(sorted(variant)) for variant in symmetry_group[:,list(new_vec)].tolist())
    pending_list.difference_update(variants)
  return list(clean_list)

def _unique_factorizations_under_symmetry(pairs: List[Tuple[Tuple[int,...],Tuple[int,...]]], symmetry_group: np.ndarray) -> List[Tuple[Tuple[int,...],Tuple[int,...]]]:
  pending_list = set(pairs.copy())
  clean_list = set([])
  while pending_list:
    new_pair = pending_list.pop()
    clean_list.add(new_pair)
    variants = set([])
    vec1_as_list = list(new_pair[0])
    vec2_as_list = list(new_pair[1])
    for perm in symmetry_group:
      new_vec1 = tuple(sorted(perm[vec1_as_list].tolist()))
      new_vec2 = tuple(sorted(perm[vec2_as_list].tolist()))
      variants.add((new_vec1, new_vec2))
      variants.add((new_vec2, new_vec1))
    pending_list.difference_update(variants)
  return list(clean_list)


def _all_and_maximal_cliques(adjmat: np.ndarray,
              max_n=0,
              isolate_maximal=True) -> (List[List[int]], List[List[int]]):
  """Based on NetworkX's `enumerate_all_cliques`.
  This version uses native Python sets instead of numpy arrays.
  (Performance comparison needed.)

  Parameters
  ----------
  adjmat : numpy.ndarray
    A boolean numpy array representing the adjacency matrix of an undirected graph.
  max_n : int, optional
    A cutoff for clique size reporting. Default 0, meaning no cutoff.
  isolate_maximal : bool, optional
    A flag to disable filtering for maximality, which can increase performance. True by default.

  Returns
  -------
  Tuple[List, List]
    A list of all cliques as well as a list of maximal cliques. The maximal cliques list will be empty if the
    `isolate_maximal` flag is set to False.
  """
  all_cliques = [[]]
  maximal_cliques = []
  verts = tuple(range(adjmat.shape[0]))
  initial_cliques = [[u] for u in verts]
  nbrs_mat = np.triu(adjmat, k=1)
  initial_cnbrs = [np.flatnonzero(nbrs_mat[u]).tolist() for u in verts]
  nbrs = list(map(set, initial_cnbrs))
  queue = deque(zip(initial_cliques, initial_cnbrs))
  there_is_a_cutoff = (max_n <= 0)
  while queue:
    base, cnbrs = queue.popleft()
    all_cliques.append(base)
    if isolate_maximal and not len(cnbrs):
      base_as_set = set(base)
      if not any(base_as_set.issubset(superbase) for (superbase, _) in queue):
        maximal_cliques.append(base)
    elif there_is_a_cutoff or len(base) < max_n:
      for i, u in enumerate(cnbrs):
        new_base = base.copy()
        new_base.append(u)
        new_cnbrs = list(filter(nbrs[u].__contains__, cnbrs[i+1:]))
        queue.append((new_base, new_cnbrs))
  return all_cliques, maximal_cliques

def _hypergraph_full_cleanup(hypergraph: List[Tuple[Tuple[int,...], Tuple[int,...]]]) -> Set[Tuple[Tuple[int,...], Tuple[int,...]]]:
  hypergraph_copy = set(hypergraph)
  cleaned_hypergraph_copy = hypergraph_copy.copy()
  for dominating_hyperedge in hypergraph_copy:
    if dominating_hyperedge in cleaned_hypergraph_copy:
      dominated_hyperedges = []
      for dominated_hyperedge in cleaned_hypergraph_copy:
        if not _factorization_subset(dominating_hyperedge, dominated_hyperedge):
          continue
        if dominated_hyperedge == dominating_hyperedge:
          continue
        if tuple(reversed(dominated_hyperedge)) == dominating_hyperedge:
          continue
        dominated_hyperedges.append(dominated_hyperedge)
      cleaned_hypergraph_copy.difference_update(dominated_hyperedges)
  return cleaned_hypergraph_copy

def _injection_graph(alices: List[Tuple[int, int]]) -> np.ndarray:
  # Caution: injectable sets learned from this graph have an orientation which must be accounted for!
  n = len(alices)
  adjacency = np.zeros((n,n), dtype=bool)
  for i, alice in enumerate(alices):
    for j, bob in enumerate(alices):
      if alice[1] == bob[0] and alice[0] != bob[1]:
        adjacency[i,j] = True
        adjacency[j,i] = True
  return adjacency

def _factorization_test(set0: List[Tuple[int, int]], set1: List[Tuple[int, int]]) -> bool:
  """
  Returns True iff the intersection of the indices of the two sets is empty.
  """
  indices0 = set(itertools.chain.from_iterable(set0))
  indices1 = set(itertools.chain.from_iterable(set1))
  return indices0.isdisjoint(indices1)


def _factorization_subset(
  set1:Tuple[Tuple[int,...], Tuple[int,...]],
  set2:Tuple[Tuple[int,...], Tuple[int,...]]) -> bool:
  """
  Check if the pair set2 has elements all of which are subsets of elements of set1.
  """
  set1 = tuple(map(set, set1))
  set2 = tuple(map(set, set2))
  return (
    (set1[0].issuperset(set2[0]) and set1[1].issuperset(set2[1])) 
    or 
    (set1[0].issuperset(set2[1]) and set1[1].issuperset(set2[0]))
    )

if __name__ == "__main__":
  list_of_Alices = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 0), (1, 3), (3, 1)]
  array_of_Alices = np.array(list_of_Alices, dtype=object)
  
  for factor_pair in maximal_factorizing_pairs(list_of_Alices):
    interpretation = [list(map(tuple, array_of_Alices[list(factor)])) for factor in factor_pair]
    print(f"Factorization {list(factor_pair)} corresponding to {interpretation}")

  for max_inj in maximal_injectable_sets(list_of_Alices):
    print(f"Injection {max_inj} corresponding to {list(map(tuple,array_of_Alices[max_inj]))}")
