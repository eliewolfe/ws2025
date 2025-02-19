import numpy as np
import itertools

def prob_noise(n: int) -> np.ndarray:
    prob = np.ones((n, n, n))/n**3
    return prob

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

def prob_all_disagree(n: int) -> np.ndarray:
    prob = np.zeros((n, n, n))
    disagree_events = n*(n-1)*(n-2)
    for (i,j,k) in itertools.permutations(range(n), 3):
        prob[i, j, k] = 1/disagree_events
    return prob

def prob_disagree_cyclic() -> np.ndarray:
    prob0 = np.zeros((3,3,3))
    prob1 = np.zeros((3,3,3))
    disagree_events = 3
    for i in range(3):
        prob0[i, (i+1)%3, (i+2)%3] = 1/disagree_events
        prob1[i, (i-1)%3, (i-2)%3] = 1/disagree_events
    return prob0, prob1
