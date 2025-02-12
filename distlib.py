import numpy as np

def prob_agree_or_disagree(n: int) -> np.ndarray:
    prob = np.zeros((n, n, n))
    agree_events = n
    disagree_events = n*(n-1)*(n-2)
    n_events = agree_events + disagree_events
    for i in range(n):
        prob[i, i, i] = 1/n_events
    for (i,j,k) in itertools.permutations(range(n), 3):
        prob[i, j, k] = 1/n_events
    return prob

def prob_agree(n:int) -> np.ndarray:
    prob = np.zeros((n, n, n))
    agree_events = n
    for i in range(n):
        prob[i,i,i] += 1/agree_events
    return prob

def prob_twosame(n: int) -> np.ndarray:
    prob = np.zeros((n, n, n))

    prob_value=1/(3*(n-1)*n)
    for i in range(n):
        for j in range (n):
            if i!=j:
                prob[i,i,j]=prob_value
                prob[i,j,i]=prob_value
                prob[j,i,i]=prob_value
    return prob

def prob_all_disagree(n:int) -> np.ndarray:
    prob = np.zeros((n, n, n))
    disagree_events = n*(n-1)*(n-2)
    for (i,j,k) in itertools.permutations(range(n), 3):
        prob[i, j, k] = 1/disagree_events
    return prob