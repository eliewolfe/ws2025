from inflation.inflation import InflationProblem, InflationLP
import numpy as np

triangle = InflationProblem(dag={"rho_AB": ["A", "B"],
                                    "rho_BC": ["B", "C"],
                                    "rho_AC": ["A", "C"]},
                                    classical_sources=None,
                                    outcomes_per_party=(3, 3, 3),
                                    settings_per_party=(1, 1, 1),
                                    inflation_level_per_source=(2, 2, 2), 
                                    order=["A", "B", "C"])
InfLP = InflationLP(triangle, include_all_outcomes=False, verbose=2)

prob_disagree = np.zeros((3, 3, 3, 1, 1, 1))
for i in range(3):
    prob_disagree[i, (i+1)%3, (i+2)%3,0,0,0] = 1/6
    prob_disagree[i, (i+2)%3, (i+1)%3,0,0,0] = 1/6
    
InfLP.set_distribution(prob_disagree, use_lpi_constraints=False)
InfLP.solve()
for k, v in InfLP.certificate_as_dict().items():
    if k.n_factors == 3:
        print(f"{k} : {v}")
print(InfLP.certificate_as_string())

