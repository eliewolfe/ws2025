from fanout_inflation_general import InfGraphOptimizer, IGO
from distlib import prob_agree, prob_all_disagree
distribution_for_vis_analysis = prob_agree(2)
from infgraphs import gen_fanout_inflation


alices=gen_fanout_inflation(5)
InfGraph52 = InfGraphOptimizer(alices, d=2, verbose=2)
optimal_vsi = InfGraph52.test_distribution(prob_agree(2),
                                maximize_visibility=True)
print(f"The optimal visibility is {optimal_vsi}")
InfGraph52.close()