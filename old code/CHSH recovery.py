# Part 1 -- quantum probability calculations.
import numpy as np
import einops
ket0 = np.array([1,0], dtype=int)[np.newaxis]
ket1 = np.array([0,1], dtype=int)[np.newaxis]

ket00 = einops.rearrange(
    einops.einsum(ket0,
                  ket0,
                  "m1 n1, m2 n2 -> m1 m2 n1 n2"),
    "m1 m2 n1 n2 -> (m1 m2) (n1 n2)")

ket11 = einops.rearrange(
    einops.einsum(ket1,
                  ket1,
                  "m1 n1, m2 n2 -> m1 m2 n1 n2"),
    "m1 m2 n1 n2 -> (m1 m2) (n1 n2)")

ket01 = einops.rearrange(
    einops.einsum(ket0,
                  ket1,
                  "m1 n1, m2 n2 -> m1 m2 n1 n2"),
    "m1 m2 n1 n2 -> (m1 m2) (n1 n2)")

ket10 = einops.rearrange(
    einops.einsum(ket1,
                  ket0,
                  "m1 n1, m2 n2 -> m1 m2 n1 n2"),
    "m1 m2 n1 n2 -> (m1 m2) (n1 n2)")

psi=np.divide(ket00+ket11, np.sqrt(2))
rho = einops.rearrange(
    einops.einsum(psi,
                  np.conj(psi).T,
                  "m1 n1, m2 n2 -> m1 m2 n1 n2"),
    "m1 m2 n1 n2 -> (m1 m2) (n1 n2)")
print(rho)

s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])

def A(x: int, a: int) -> np.ndarray:
    sign = (+1, -1)[a]
    basis = (s3, s1)[x]
    return (np.eye(2)+sign*basis)/2

def B(y: int, b: int) -> np.ndarray:
    sign = (+1, -1)[b]
    basis = ((s3+s1)/np.sqrt(2), (s3-s1)/np.sqrt(2))[y]
    return (np.eye(2)+sign*basis)/2

def compute_prob(x: int, y: int, a: int, b: int) -> float:
    AB_op = einops.rearrange(
        einops.einsum(A(x, a),
                      B(y, b),
                      "m1 n1, m2 n2 -> m1 m2 n1 n2"),
        "m1 m2 n1 n2 -> (m1 m2) (n1 n2)")
    Pxy = einops.einsum(rho,
                        AB_op,

                        "m n, m n -> ")
    return Pxy


Pquantum = np.zeros((2,2,2,2), dtype=float)
for a,b,x,y in np.ndindex(2,2,2,2):
    Pquantum[x,y,a,b] = compute_prob(x,y,a,b)
for x,y in np.ndindex(2,2):
    print(Pquantum[x,y])



# from ortools.linear_solver import pywraplp
# solver = pywraplp.Solver.CreateSolver("GLOP")

from ortools.math_opt.python import mathopt
model = mathopt.Model(name="LHVM")
variables_dict = dict()
variables_list = []
constraints_for_dual = []
for x,y,a,b in np.ndindex(2,2,2,2):
    var_name = f"P(A={a},B={b}|X={x},Y={y})"
    variable = model.add_variable(lb=0.0, name=var_name)
    variables_dict[var_name] = variable
    constraint = model.add_linear_constraint(variable == Pquantum[x,y,a,b])
    constraints_for_dual.append(constraint)
    variables_list.append(var_name)
slack = model.add_variable(name="slack")
for y,a1,a2,b in np.ndindex(2,2,2,2):
    var_name = f"P(A_0={a1}, A_1={a2}, B={b}|Y={y})"
    variable = model.add_variable(name=var_name)
    variables_dict[var_name] = variable
    model.add_linear_constraint(variable >= -slack)
#Imposing consistency
for y,a,b in np.ndindex(2,2,2):
    model.add_linear_constraint(
        variables_dict[f"P(A={a},B={b}|X={0},Y={y})"] == sum(
            variables_dict[f"P(A_0={a}, A_1={a2}, B={b}|Y={y})"] for a2 in range(2)))
    model.add_linear_constraint(
        variables_dict[f"P(A={a},B={b}|X={1},Y={y})"] == sum(
            variables_dict[f"P(A_0={a2}, A_1={a}, B={b}|Y={y})"] for a2 in range(2)))
#Imposing no-signalling on the partially unpacked graph
for a1, a2 in np.ndindex(2,2):
    model.add_linear_constraint(
        sum(
            variables_dict[f"P(A_0={a1}, A_1={a2}, B={b}|Y={0})"] for b in range(2)
        ) == sum(
        variables_dict[f"P(A_0={a1}, A_1={a2}, B={b}|Y={1})"] for b in range(2)
    )
    )
#Quantum probabilities
model.set_objective(objective=slack, is_maximize=False)
solution = mathopt.solve(model, mathopt.SolverType.GLOP,
              params = mathopt.SolveParameters(enable_output=True))
print("Solutions:", solution.solutions)

print("Dual values:", solution.dual_values(constraints_for_dual))
for i, v in enumerate(solution.dual_values(constraints_for_dual)):
    if not np.isclose(v, 0):
        print(f"{v}*{variables_list[i]}")