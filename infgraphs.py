from typing import List, Tuple, TypeAlias
import itertools
import numpy as np
from functools import cached_property
from operator import itemgetter
from utils import (_all_and_maximal_cliques,
                   _unique_vecs_under_symmetry,
                   _unique_factorizations_under_symmetry,
                   partitions_with_min_size,
                   _hypergraph_full_cleanup)

Alices: TypeAlias = List[Tuple[int,int]]
Indices: TypeAlias = Tuple[int, ...]

def gen_fanout_inflation(n: int) -> Alices:
    return [pair for pair in itertools.permutations(range(n), 2) if pair[0] != (pair[1]+1)%n]

def gen_nonfanout_inflation(n: int) -> Alices:
    return [(i,i+1) for i in range(n-1)] + [(n-1,0)]

def gen_fanout_inflation_alt(n: int) -> Alices:
    return list(itertools.permutations(range(n), 2))

class InfGraph:
    def __init__(self, alices: Alices):
        self.alices = alices
        self.nof_alices = len(alices)
        self.canonical_order = {pair: i for i, pair in enumerate(alices)}
        self.flat_indices = tuple(itertools.chain.from_iterable(alices))
        self.nof_flat_indices = len(self.flat_indices)
        self.all_copy_indices = sorted(set(itertools.chain.from_iterable(alices)))
        self.nof_sources = len(self.all_copy_indices)
        assert np.array_equal(self.all_copy_indices, np.arange(self.nof_sources)), "Copy indices must be contiguous."
        (self.all_injectable_sets, self.maximal_injectable_sets) = _all_and_maximal_cliques(
            self.injection_graph,
            isolate_maximal=True)
        self.maximal_injectable_sets_under_symmetry = _unique_vecs_under_symmetry(self.maximal_injectable_sets,
                                                                                  self.symmetry_group)

    def __str__(self):
        return f"{self.alices}"

    def __repr__(self):
        return self.__str__()

    @cached_property
    def injection_graph(self) -> np.ndarray:
        adjacency = np.zeros((self.nof_alices, self.nof_alices), dtype=bool)
        for i, alice in enumerate(self.alices):
            for j, bob in enumerate(self.alices):
                if alice[1] == bob[0] and alice[0] != bob[1]:
                    adjacency[i, j] = True
                    adjacency[j, i] = True
        return adjacency

    @cached_property
    def symmetry_group(self) -> np.ndarray:
        discovered_symmetries = []
        for perm_candidate in itertools.permutations(self.all_copy_indices):
            permuted_flat_indices = itemgetter(*self.flat_indices)(perm_candidate)
            relabelled_alices = [permuted_flat_indices[i:i+2] for i in range(0, self.nof_flat_indices, 2)]
            try:
                discovered_symmetries.append([self.canonical_order[pair] for pair in relabelled_alices])
            except KeyError:
                continue
        assert len(discovered_symmetries) >= 1, "At least the identity permutation should be found."
        return np.array(discovered_symmetries, dtype=int)

    @cached_property
    def maximal_factorizing_pairs(self) -> List[Tuple[Indices, Indices]]:
        # First let's obtain all pairs of factorizing sets based on sources, then we'll filter for maximality.
        factorizing_source_pairs = partitions_with_min_size(self.alices)
        factorizing_pairs = []
        for source_partition in factorizing_source_pairs:
            source_set1 = set(source_partition[0])
            source_set2 = set(source_partition[1])
            first_alices_idxs = sorted(self.canonical_order[alice] for alice in self.alices if source_set1.issuperset(alice))
            second_alices_idxs = sorted(self.canonical_order[alice] for alice in self.alices if source_set2.issuperset(alice))
            factorizing_pairs.append((tuple(first_alices_idxs), tuple(second_alices_idxs)))
        return list(_hypergraph_full_cleanup(factorizing_pairs))

    @cached_property
    def maximal_factorizing_pairs_under_symmetry(self) -> List[Tuple[Indices, Indices]]:
        return _unique_factorizations_under_symmetry(self.maximal_factorizing_pairs,
                                                     self.symmetry_group)


if __name__ == "__main__":
    print(gen_fanout_inflation(4))

    print(gen_nonfanout_inflation(5))