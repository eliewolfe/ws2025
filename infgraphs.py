from typing import List, Tuple, TypeAlias
import itertools
import numpy as np
from functools import cached_property
from operator import itemgetter
from utils import (_all_and_maximal_cliques,
                   _unique_vecs_under_symmetry,
                   _unique_factorizations_under_symmetry,
                   partitions_with_min_size,
                   _factorizations_full_cleanup,
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
        self.all_injectable_sets = list(map(tuple, self.all_injectable_sets))
        self.maximal_injectable_sets = list(map(tuple, self.maximal_injectable_sets))
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
        factorizing_source_pairs = partitions_with_min_size(self.alices, min_size=2)
        factorizing_pairs = []
        for (lhs, rhs) in factorizing_source_pairs:
            source_set1 = set(lhs)
            source_set2 = set(rhs)
            first_alices_idxs = sorted(self.canonical_order[alice] for alice in self.alices if source_set1.issuperset(alice))
            second_alices_idxs = sorted(self.canonical_order[alice] for alice in self.alices if source_set2.issuperset(alice))
            factorizing_pairs.append((tuple(first_alices_idxs), tuple(second_alices_idxs)))
        return list(_factorizations_full_cleanup(factorizing_pairs))

    @cached_property
    def maximal_factorizing_pairs_under_symmetry(self) -> List[Tuple[Indices, Indices]]:
        return _unique_factorizations_under_symmetry(self.maximal_factorizing_pairs,
                                                     self.symmetry_group)

    @cached_property
    def maximal_semiexpressible_sets(self) -> List[Tuple[Indices, Indices]]:
        semiexpressible_pairs = []
        for (lhs, rhs) in self.maximal_factorizing_pairs:
            lhs_set = set(lhs)
            injectable_components = [injectable_set for injectable_set in self.all_injectable_sets if lhs_set.issuperset(injectable_set)]
            injectable_components = _hypergraph_full_cleanup(injectable_components)
            for new_lhs in injectable_components:
                semiexpressible_pairs.append((new_lhs, rhs))
            rhs_set = set(rhs)
            injectable_components = [injectable_set for injectable_set in self.all_injectable_sets if rhs_set.issuperset(injectable_set)]
            injectable_components = _hypergraph_full_cleanup(injectable_components)
            for new_rhs in injectable_components:
                semiexpressible_pairs.append((new_rhs, lhs))
        return list(_factorizations_full_cleanup(semiexpressible_pairs))

    @cached_property
    def maximal_semiexpressible_sets_under_symmetry(self) -> List[Tuple[Indices, Indices]]:
        return _unique_factorizations_under_symmetry(self.maximal_semiexpressible_sets,
                                                     self.symmetry_group)

    @cached_property
    def maximal_non_semiexpressible_pairs(self) -> List[Tuple[Indices, Indices]]:
        to_filter_out = set(self.maximal_semiexpressible_sets)
        to_filter_out.update([tuple(reversed(pair)) for pair in self.maximal_semiexpressible_sets])
        return list(set(self.maximal_factorizing_pairs).difference(to_filter_out))

    @cached_property
    def maximal_non_semiexpressible_pairs_under_symmetry(self) -> List[Tuple[Indices, Indices]]:
        return _unique_factorizations_under_symmetry(self.maximal_non_semiexpressible_pairs,
                                                     self.symmetry_group)

    @cached_property
    def maximal_injectable_but_not_semi_expressible(self) -> List[Indices]:
        to_keep = set(self.maximal_injectable_sets)
        to_filter_out = set(semiexpressible_set[0] for semiexpressible_set in self.maximal_semiexpressible_sets)
        to_keep.difference_update(to_filter_out)
        return list(to_keep)

    @cached_property
    def maximal_injectable_but_not_semi_expressible_under_symmetry(self) -> List[Indices]:
        return _unique_vecs_under_symmetry(self.maximal_injectable_but_not_semi_expressible,
                                           self.symmetry_group)




if __name__ == "__main__":
    print(gen_fanout_inflation(4))

    print(gen_nonfanout_inflation(5))