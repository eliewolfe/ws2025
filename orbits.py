import numpy as np
from tqdm import tqdm
# from gc import collect

# def identify_orbits(tensor_shape, permutation):
#     """
#     Identify orbits from the action of a symmetry generated by a permutation on the indices of a tensor.
#     Slower but lower memory usage than the alternative implementation.
#     """
#     total_elements = np.prod(tensor_shape)
#     paradigm = np.arange(total_elements, dtype=np.min_scalar_type(total_elements)).reshape(tensor_shape)
#     orbits_found = []
#     discovered_yet = np.zeros(total_elements, dtype=bool)
#     shifted = paradigm.transpose(permutation).flat
#     perm_iterator = tuple(range(len(permutation)))
#     for i in tqdm(paradigm.flat, total=total_elements):
#         if discovered_yet[i]:
#             continue
#         current_orbit = []
#         x=i
#         for _ in perm_iterator:
#             discovered_yet[x] = True
#             current_orbit.append(x)
#             x = shifted[x]
#         orbits_found.append(current_orbit)
#     return np.array(orbits_found, dtype=int)

def identify_orbits(tensor_shape, permutation):
    """
    Identify orbits from the action of a symmetry generated by a permutation on the indices of a tensor.
    Faster but higher memory usage than the alternative implementation.
    """
    current_perm = np.array(permutation, dtype=int)
    all_perms = []
    for _ in range(len(permutation)-1):
        current_perm = np.take(current_perm, permutation)
        all_perms.append(current_perm)
    all_perms = np.unique(all_perms, axis=0)
    n_nontrivial_perms = len(all_perms)
    total_elements = np.prod(tensor_shape)
    np_dtype = np.min_scalar_type(total_elements)
    paradigm = np.arange(total_elements, dtype=np_dtype).reshape(tensor_shape)
    alternatives = np.empty(shape=(n_nontrivial_perms, total_elements), dtype=np_dtype)
    print("Now exploring the consequences of the permutations")
    for i, perm in tqdm(enumerate(all_perms), total=n_nontrivial_perms):
        alternatives[i] = paradigm.transpose(tuple(perm)).reshape(-1)
    # del paradigm
    # collect(generation=2)
    # Picklist computation assumes that the permutation is a derangement
    picklist = np.all(paradigm.reshape(-1) <= alternatives, axis=0)
    print("Picklist identified, now compressing and stacking.")
    nof_orbits = np.count_nonzero(picklist)
    orbits = np.empty(shape=(n_nontrivial_perms+1, nof_orbits), dtype=np_dtype)
    orbits[0] = paradigm.reshape(-1)[picklist]
    for i, alternative in tqdm(enumerate(alternatives[:, picklist]), total=n_nontrivial_perms):
        orbits[i+1] = alternative
    # representatives = paradigm.reshape(-1)[picklist,np.newaxis]
    # nonrepresentatives = alternatives.T[picklist]
    return orbits.T

if __name__ == "__main__":
    # Example usage
    tensor_shape = (3, 3, 3)
    permutation = [2, 0, 1]
    orbits = identify_orbits(tensor_shape, permutation)
    print(orbits)