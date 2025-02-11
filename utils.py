from typing import List, Tuple, Set
import itertools
from collections import deque
import numpy as np
from more_itertools import powerset

def maximal_injectable_sets(alices: List[Tuple[int, int]]) -> List:
  return all_and_maximal_cliques(injection_graph(alices), isolate_maximal=True)[1]

def all_and_maximal_cliques(adjmat: np.ndarray,
              max_n=0,
              isolate_maximal=True) -> (List, List):
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

def hypergraph_full_cleanup(hypergraph: Set[Tuple[Set, Set]]) -> Set[Tuple[Set, Set]]:
  hypergraph_copy = set(hypergraph)
  cleaned_hypergraph_copy = hypergraph_copy.copy()
  for dominating_hyperedge in hypergraph_copy:
    if dominating_hyperedge in cleaned_hypergraph_copy:
      dominated_hyperedges = []
      for dominated_hyperedge in cleaned_hypergraph_copy:
        if not factorization_subset(dominating_hyperedge, dominated_hyperedge):
          continue
        if dominated_hyperedge == dominating_hyperedge:
          continue
        if tuple(reversed(dominated_hyperedge)) == dominating_hyperedge:
          continue
        dominated_hyperedges.append(dominated_hyperedge)
      cleaned_hypergraph_copy.difference_update(dominated_hyperedges)
  return cleaned_hypergraph_copy

def injection_graph(alices: List[Tuple[int, int]]) -> np.ndarray:
  # Caution: injectable sets learned from this graph have an orientation which must be accounted for!
  n = len(alices)
  adjacency = np.zeros((n,n), dtype=bool)
  for i, alice in enumerate(alices):
    for j, bob in enumerate(alices):
      if alice[1] == bob[0] and alice[0] != bob[1]:
        adjacency[i,j] = True
        adjacency[j,i] = True
  return adjacency

def factorization_test(set0: Set[Tuple[int, int]], set1: Set[Tuple[int, int]]) -> bool:
  """
  Returns True iff the intersection of the indices of the two sets is empty.
  """
  indices0 = set([x for xs in set0 for x in xs])
  indices1 = set([x for xs in set1 for x in xs])
  return indices0.isdisjoint(indices1)

def maximal_factorizing_pairs(alices: List[Tuple[int, int]]) -> List[Tuple[List, List]]:
  #First let's obtain all pairs of factorizing sets, then we'll filter for maximality.
  factorizing_pairs = []
  explored_sets = set()
  alice_indices = set(range(len(alices)))
  for first_set in powerset(sorted(alice_indices)):
    explored_sets.add(frozenset(first_set))
    if not first_set:
      continue
    first_alices = [alices[i] for i in first_set]
    for second_set in powerset(sorted(alice_indices.difference(first_set))):
      if frozenset(second_set) in explored_sets:
        continue 
      second_alices = [alices[i] for i in second_set]
      if factorization_test(first_alices, second_alices):
        factorizing_pairs.append((first_set, second_set))
  return hypergraph_full_cleanup(factorizing_pairs)


# def factorizing_pairs_from_injectable_sets(injectable_sets: List[Set[Tuple[int, int]]]) -> List[Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]]:
#   return list(filter(factorization_test, itertools.permutations(injectable_sets, 2)))

# def factorizing_pairs_from_alices(alices: List[Tuple[int, int]]) -> np.ndarray:
#   injectable_sets, maximal_injectable_sets = all_and_maximal_cliques(injection_graph(alices), isolate_maximal=False)
#   return factorizing_pairs_from_injectable_sets(injectable_sets)

def factorization_subset(
  set1:Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]], 
  set2:Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]) -> bool:
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
