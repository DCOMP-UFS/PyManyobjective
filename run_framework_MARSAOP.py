import json
import warnings
from run import run

warnings.simplefilter(action='ignore', category=FutureWarning)

args = {}
args["Gmax"] = 10
args["variance"] = 0.2
args["prob"] = 1.0
args["wv"] = 0.1
args["thr1"] = 12
args["thr2"] = 14

bestPrecision = 0

for wv in range(1, 101, 5):
    for thr1 in range(1, 10):
        for thr2 in range(1, 10):
            print(wv, thr1, thr2)

            args["wv"] = wv / 100.0
            args["thr1"] = thr1
            args["thr2"] = thr2

            with open("resources/args_samples/framework_args_MARSAOP.json", "w") as args_file:
                args_file.write(json.dumps(args))

            run("MARSAOP", "SVM_hyperparameters", "GA", "SBX", "Polynomial", "Binary", "CrowdingDistance",
                5,
                framework_args_file="resources/args_samples/framework_args_MARSAOP.json",
                moea_args_file="resources/args_samples/MEMO_args.json",
                pareto_front_file="tests/zdt1_front.txt",
                problem_args_file="resources/args_samples/problem_args_ZDT.json",
                mutation_args_file="resources/args_samples/Polynomial_args.json",
                crossover_args_file="resources/args_samples/SBX_args.json")
            
            ac = 0
            for i in range(5):
                with open("results/MARSAOP_SVM_hyperparameters_GA/bestPrecision" + str(i), "r") as result_file:
                    ac += float(result_file.readline())
            ac /= 5
            bestPrecision = max(bestPrecision, ac)

print(bestPrecision)

                