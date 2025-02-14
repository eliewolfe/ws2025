import gurobipy as gp
from gurobipy import GRB
from itertools import product, repeat
import numpy as np
import qutip as qt

time_limit = GRB.INFINITY
tol = 1e-5
return_dist = True
print_model = True

def square_inflation(p_ABCXYZ: np.ndarray):
    nc = 4 # we are doing the square inflation
    (cardA, cardB, cardC, cardX, cardY, cardZ) = p_ABCXYZ.shape 
    assert cardA == cardB == cardC, "The input probability distribution is not valid, the marginals do not match"
    assert cardX == cardY == cardZ, "The input probability distribution is not valid, the marginals do not match"
    cardinalities = tuple(repeat(cardA, times=nc)) + tuple(repeat(cardX, times=nc)) # we are imposing symmetries so the 4 parties will have the same number of outputs and inputs
    cardinalities_sum = tuple(repeat(cardA, times=(nc-1))) + tuple(repeat(cardX, times=nc))
    
    with (gp.Env(empty=True) as env):
        # env.setParam('OutputFlag', 0) # To supress output
        env.start()
        with gp.Model("qcp", env=env) as m:
            m.params.NonConvex = 2  # Using quadratic equality constraints.
            
            # Defining the probabilities
            Q_ABCDXYZW = m.addMVar(cardinalities, lb=0, name="Q_ABCDXYZW")
            for x, y, z, w in np.ndindex(cardX, cardX, cardX, cardX):
                m.addConstr(Q_ABCDXYZW[...,x,y,z,w].sum()==1, name="Normalization")
            
            # NS constraints
            Q_ABCXYZW = m.addMVar(cardinalities_sum, lb=0, name="Q_ABCXYZW")
            Q_ABDXYZW = m.addMVar(cardinalities_sum, lb=0, name="Q_ABDXYZW")
            Q_ACDXYZW = m.addMVar(cardinalities_sum, lb=0, name="Q_ACDXYZW")
            Q_BCDXYZW = m.addMVar(cardinalities_sum, lb=0, name="Q_BCDXYZW")
            m.addConstr(Q_ABCXYZW==Q_ABCDXYZW.sum(axis=3), name="Q_ABCXYZW from Q_ABCDXYZW")
            m.addConstr(Q_ABDXYZW==Q_ABCDXYZW.sum(axis=2), name="Q_ABDXYZW from Q_ABCDXYZW")
            m.addConstr(Q_ACDXYZW==Q_ABCDXYZW.sum(axis=1), name="Q_ACDXYZW from Q_ABCDXYZW")
            m.addConstr(Q_BCDXYZW==Q_ABCDXYZW.sum(axis=0), name="Q_BCDXYZW from Q_ABCDXYZW")
            
            for inp in tuple(range(cardX-1)):
                m.addConstr(Q_ABCXYZW[...,inp]==Q_ABCXYZW[...,inp+1], name="NS for w")
                m.addConstr(Q_ABDXYZW[...,inp,:]==Q_ABDXYZW[...,inp+1,:], name="NS for z")
                m.addConstr(Q_ACDXYZW[...,inp,:,:]==Q_ACDXYZW[...,inp+1,:,:], name="NS for y")
                m.addConstr(Q_BCDXYZW[...,inp,:,:,:]==Q_BCDXYZW[...,inp+1,:,:,:], name="NS for x")
                
            # Cyclic symmetry
            # See https://numpy.org/doc/2.1/reference/generated/numpy.permute_dims.html
            # m.addConstr(Q_ABCDXYZW == np.permute_dims(Q_ABCDXYZW, axes=[1,2,3,0,5,6,7,4]))
            for (a,b,c,d,x,y,z,w) in np.ndindex(*Q_ABCDXYZW.shape):
                m.addConstr(Q_ABCDXYZW[a,b,c,d,x,y,z,w] == Q_ABCDXYZW[b,c,d,a,y,z,w,x])
            
            # Injectable sets
            p_ABXY=p_ABCXYZ.sum(axis=(2))[:,:,:,:,0]
            m.addConstr(p_ABXY==Q_ABCDXYZW.sum(axis=(2,3))[:,:,:,:,0,0], name="Q_ABXY from Q_ABCDXYZW")        
            
            # Independencies
            Q_ACXZ = m.addMVar((cardA, cardA, cardX, cardX), lb=0, name="Q_ACXZ")
            Q_AX = m.addMVar((cardA, cardX), lb=0, name="Q_AX")
            Q_CZ = m.addMVar((cardA, cardX), lb=0, name="Q_CZ")
            m.addConstr(Q_ACXZ==Q_ABCDXYZW.sum(axis=(1,3))[:,:,:,0,:,0], name="Q_ACXZ from Q_ABCDXYZW")
            m.addConstr(Q_AX==Q_ACXZ.sum(axis=(1))[:,:,0], name="Q_AX from Q_ACXZ")
            m.addConstr(Q_CZ==Q_ACXZ.sum(axis=(0))[:,0,:], name="Q_CZ from Q_ACXZ")

            for o1, o2, i1, i2 in np.ndindex(*Q_ACXZ.shape):
                m.addConstr(Q_ACXZ[o1,o2,i1,i2]==Q_AX[o1,i1]*Q_CZ[o2,i2], name="indep 1")
            
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


prob_disagree = np.zeros((3, 3, 3, 1, 1, 1))
for i in range(3):
    prob_disagree[i, (i+1)%3, (i+2)%3,0,0,0] = 1/6
    prob_disagree[i, (i+2)%3, (i+1)%3,0,0,0] = 1/6

print("Checking the infeasibility of the always-agree distribution")
print(square_inflation(prob))

print("\n\nChecking the feasibility of the noisy-GHZ distribution")
print(square_inflation(p_post))

print("\n\nChecking the feasibility of the all-disagree distribution")
print(square_inflation(prob_disagree))

