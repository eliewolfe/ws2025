import gurobipy as gp
from gurobipy import GRB
from itertools import product, repeat, chain
import numpy as np
import itertools
from gphelpers import create_cyclic_symmetric_mVar

time_limit = GRB.INFINITY
tol = 1e-5
return_dist = True
print_model = True
    

def generate_tuples(n, inp, out):
    # Define ranges for first n entries and second n entries
    first_range = range(inp)  # Values: 0 to inp-1
    second_range = range(out) # Values: 0 to out-1

    # Generate all n-length tuples for each half
    first_half = list(itertools.product(first_range, repeat=n))
    second_half = list(itertools.product(second_range, repeat=n))

    # Combine every first_half tuple with every second_half tuple
    all_tuples = [first + second for first, second in itertools.product(first_half, second_half)]
    
    return all_tuples


def nonfanout_inflation(p_ABCXYZ:np.ndarray, nc:int):
    (cardA, cardB, cardC, cardX, cardY, cardZ) = p_ABCXYZ.shape 
    assert cardA == cardB == cardC, "The input probability distribution is not valid, the marginals do not match"
    assert cardX == cardY == cardZ, "The input probability distribution is not valid, the marginals do not match"

    with (gp.Env(empty=True) as env):
        # env.setParam('OutputFlag', 0) # To supress output
        env.start()
        with gp.Model("qcp", env=env) as m:
            def cycle_variables(nc, cardout, cardinp):
                # Defining the cyclic-symmetric robabilities
                combined_d = cardout * cardinp
                Q_as_ndarray = np.array(create_cyclic_symmetric_mVar(m, combined_d, nc).tolist(), dtype=object)
                initial_shape = (cardout, cardinp)*nc
                transposition = tuple(np.argsort(tuple(chain.from_iterable(zip(range(nc), range(nc, 2*nc))))).flat)
                Q_as_ndarray = Q_as_ndarray.reshape(initial_shape).transpose(transposition)
                Q_ = gp.MVar.fromlist(Q_as_ndarray)
                Q_.__name__ = f"Q_{nc}"

                cardinalities = tuple(repeat(cardout, times=nc)) + tuple(repeat(cardinp, times=nc))
                assert np.array_equal(Q_.shape, cardinalities), f"The shape of the MVar {Q_.shape} does not match the expected shape {cardinalities}"

                # Independencies
                tuplenc = tuple(range(nc))
                tuple_0 = tuple(np.delete(tuplenc,0))
                for outs in itertools.product(range(cardout), repeat=nc-2):
                    for inps in itertools.product(range(cardinp), repeat=nc):
                        m.addConstr(Q_.sum(axis=(1, nc-1))[outs+inps]==Q_.sum(axis=(tuple_0))[(outs[0],)+inps]*Q_.sum(axis=(0,1,nc-1))[tuple(outs[1:(nc-1)])+inps], "Independences")

                # No signalling
                for inp in range(cardinp-1):
                    m.addConstr(Q_.sum(axis=nc)[...,inp]==Q_.sum(axis=nc)[...,inp+1], f"No-signalling {inp} vs {inp+1}")


                return Q_
            
            # Define the probabilities
            Qs = [cycle_variables(nc=4+l, cardout=cardA, cardinp=cardX) for l in range(nc-3)]


            # Injectable sets
            p_ABXY = p_ABCXYZ.sum(axis=(2))[:,:,:,:,0]
            # index_tuple_0 = (slice(None),) * 2*(0+2) + (0,) * (4-(0+2))
            m.addConstr(Qs[0].sum(axis=(2, 3))[:,:,:,:,0,0] == p_ABXY, name="Injectable sets")        
            
            if nc>=5:
                for l in range(1,(nc-3)):
                    l_1=l-1
                    l_2=l
                    index_tuple_1 = (slice(None),) * 2*(l+2) + (0,) * ((l_1+4)-(l+2))
                    index_tuple_2 = (slice(None),) * 2*(l+2) + (0,) * ((l_2+4)-(l+2))
                    m.addConstr(Qs[l_1].sum(axis=tuple(range(l+2,l_1+4)))[index_tuple_1]==Qs[l_2].sum(axis=tuple(range(l+2,l_2+4)))[index_tuple_2], "Compatibility")
            # this could be simplified because we're always just suming over 1 index or 2 indeces so we don't need the range
            
            m.setObjective(0.0, GRB.MAXIMIZE)
            m.Params.NonConvex = 2

            try:
                m.optimize()
                m.getAttr('x')  # To trigger an error in case it is not solved, because I made it silent
                return 1
            except:
                return 0
            
# # Checks
prob = np.zeros((2, 2, 2, 1, 1, 1))
prob[0,0,0,0,0,0] = 1/2
prob[1,1,1,0,0,0] = 1/2

def prob_postquantum(E1, E2, E3):
    p = np.zeros((2, 2, 2, 1, 1, 1))
    for a, b, c in np.ndindex(2, 2, 2):
        ap = 2*a-1
        bp = 2*b-1
        cp = 2*c-1
        p[a, b, c] = (1/8) * (1 
                                 + (ap + bp + cp)*E1 
                                 + (ap*bp + bp*cp + cp*ap)*E2 
                                 + ap*bp*cp*E3)
    return p

p_post = np.zeros((2, 2, 2, 1, 1, 1))
for a, b, c in np.ndindex(2, 2, 2):
    p_post[a,b,c,0,0,0] = prob_postquantum(0,0,np.sqrt(2))[a,b,c]

def prob_agree(n: int) -> np.ndarray:
    prob = np.zeros((n, n, n, 1, 1, 1))
    for i in range(n):
        prob[i, i, i,0,0,0] = 1/n
    return prob

from distlib import prob_all_disagree as prob_all_disagree_no_inputs
from distlib import prob_twosame as prob_two_agree_no_inputs

def prob_all_disagree(n: int) -> np.ndarray:
    return prob_all_disagree_no_inputs(n).reshape((n, n, n, 1, 1, 1))

def prob_twosame(n: int) -> np.ndarray:
    return prob_two_agree_no_inputs(n).reshape((n, n, n, 1, 1, 1))


print("\n\nChecking the feasibility of the all-disagree distribution 3-outcomes")
print(nonfanout_inflation(prob_all_disagree(3),4))

print("\n\nChecking the feasibility of the all-disagree distribution 4-outcomes")
print(nonfanout_inflation(prob_all_disagree(4),5))

#
# print("\n\nChecking the feasibility of the 2 agree")
# print(nonfanout_inflation(prob_twosame(3), 7))


# print("\n\nChecking the feasibility of the noisy-GHZ distribution")
# print(nonfanout_inflation(p_post, 8))


# print("Checking the infeasibility of the always-agree distribution")
# print(nonfanout_inflation(prob,4))

# p_zeros = np.zeros((2, 2, 2, 1, 1, 1))
# p_zeros[0,0,0,0,0,0] = 1
# print("\n\nChecking the feasibility of the noisy-GHZ distribution")
# print(nonfanout_inflation(p_zeros, 10))

    