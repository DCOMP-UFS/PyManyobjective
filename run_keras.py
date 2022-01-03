import json

from src.MOEAs.NSGAII import *
from src.MOEAs.MRGA import *

from src.MOEAs.mutations.PolynomialMutation import PolynomialMutation

from src.MOEAs.crossovers.SBXCrossover import SBXCrossover

from src.MOEAs.sparsities.CrowdingDistance import CrowdingDistance

from src.BinaryTournament import BinaryTournament

from src.frameworks.M1 import M1
from src.frameworks.M3 import M3
from src.frameworks.M6 import M6
from src.frameworks.SMO import SMO
from src.frameworks.SMB import SMB

from src.problems.ZDT import *
from src.problems.DTLZ import *
from pymoo.problems.multi.zdt import ZDT1 as ZDT1_pymoo
from pymoo.problems.multi.zdt import ZDT3 as ZDT3_pymoo
from pymoo.problems.multi.zdt import ZDT6 as ZDT6_pymoo
from pymoo.problems.many.dtlz import DTLZ1 as DTLZ1_pymoo
from pymoo.problems.many.dtlz import DTLZ2 as DTLZ2_pymoo
from pymoo.problems.many.dtlz import DTLZ3 as DTLZ3_pymoo
from pymoo.problems.many.dtlz import DTLZ4 as DTLZ4_pymoo
from pymoo.problems.many.dtlz import DTLZ5 as DTLZ5_pymoo
from pymoo.problems.many.dtlz import DTLZ6 as DTLZ6_pymoo
from pymoo.problems.many.dtlz import DTLZ7 as DTLZ7_pymoo

from keras.datasets import mnist

from src.problems.keras_nn import Keras_NN

from keras.utils import np_utils

from sklearn import datasets
from sklearn.model_selection import train_test_split

from pymoo.factory import get_performance_indicator
from pymoo.performance_indicator.kktpm import KKTPM

from pathlib import Path

def get_sparsity(sparsity):
    if sparsity == "CrowdingDistance":
        return CrowdingDistance()

    print("unknown crowding")
    return None

def get_selection(selection):
    if selection == "Binary":
        return BinaryTournament()

    print("unknown selection")
    return None

def get_mutation(mutation, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)

    if mutation == "Polynomial":
        return PolynomialMutation(mutationProbability=args["probability"], distributionIndex=args["distribution"])

    print("unknown mutation")
    return None

def get_crossover(crossover, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)

    if crossover == "SBX":
        return SBXCrossover(distributionIndex=args["distribution"], crossoverProbability=args["probability"])

    print("unknown crossover")
    return None

def get_problem_pymoo(problem, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)

    if problem == "ZDT1":
       return ZDT1_pymoo(args["n"])
    if problem == "ZDT3":
       return ZDT3_pymoo(args["n"])
    if problem == "ZDT6":
       return ZDT6_pymoo(args["n"])

    args["n"] = args["m"] + args["k"] - 1

    if problem == "DTLZ1":
        return DTLZ1_pymoo(args["n"], args["m"])
    if problem == "DTLZ2":
        return DTLZ2_pymoo(args["n"], args["m"])
    if problem == "DTLZ3":
        return DTLZ3_pymoo(args["n"], args["m"])
    if problem == "DTLZ4":
        return DTLZ4_pymoo(args["n"], args["m"])
    if problem == "DTLZ5":
        return DTLZ5_pymoo(args["n"], args["m"])
    if problem == "DTLZ6":
        return DTLZ6_pymoo(args["n"], args["m"])
    if problem == "DTLZ7":
        return DTLZ7_pymoo(args["n"], args["m"])

    print("unknown problem")

def get_framework(framework, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)

    if framework == "M1": 
        return M1(None, None, args["sample_size"], args["tau"], args["SEmax"])
    if framework == "M3":
        return M3(None, None, args["sample_size"], args["SEmax"], args["k"], args["alfa"], args["R"])
    if framework == "M6":
        return M6(None, None, None, args["sample_size"], args["SEmax"], args["R"], KKTPM())

    if framework == "SMO":
        return SMO(None, None, args["sample_size"], args["tmax"], args["tparada"])
    if framework == "SMB":
        return SMB(None, None, args["sample_size"], args["tmax"], args["tparada"])

    print("unknown framework")

def get_MOEA(MOEA, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)

    if MOEA == "NSGAII": 
        return NSGAII(None, args["maxEvaluations"], args["populationSize"], args["offspringSize"], None, None, None, None)
    if MOEA == "MRGA": 
        return MRGA(None, args["maxEvaluations"], args["populationSize"], args["offspringSize"], None, None, None, None, R=None)

    print("unknown problem")
    return None

def get_problem(problem, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)

    # ZDT problems
    if problem == "ZDT1":
        return ZDT1(args["n"])
    if problem == "ZDT3": 
        return ZDT3(args["n"])
    if problem == "ZDT6": 
        return ZDT6(args["n"])

    if problem == "DTLZ1": 
        return DTLZ1(args["m"], args["k"])
    if problem == "DTLZ2": 
        return DTLZ2(args["m"], args["k"])
    if problem == "DTLZ3": 
        return DTLZ3(args["m"], args["k"])
    if problem == "DTLZ4": 
        return DTLZ4(args["m"], args["k"])
    if problem == "DTLZ5": 
        return DTLZ5(args["m"], args["k"])
    if problem == "DTLZ6": 
        return DTLZ6(args["m"], args["k"])
    if problem == "DTLZ7": 
        return DTLZ7(args["m"], args["k"])

    if problem == "NN_MNIST":
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))

        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.5, shuffle=False)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_test /= 255

        nb_classes = 10
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        return Keras_NN(X_train, y_train, X_test, y_test, 64, 10)

    print("unknown problem")
    return None

def save(name, i, P, F, igd):
    Path("results/" + name).mkdir(parents=True, exist_ok=True)
    with open("results/" + name + "/P" + i, 'w+') as P_file:
        P_file.write(json.dumps(P))
    with open("results/" + name + "/F" + i, 'w+') as F_file:
        F_file.write(json.dumps(F))
    with open("results/" + name + "/igd" + i, 'w+') as igd_file:
        igd_file.write(json.dumps(igd))

def run(framework, problem, moea, crossover, mutation, selection, sparsity, n, framework_args_file, problem_args_file, moea_args_file, pareto_front_file, mutation_args_file, crossover_args_file):
    Problem = get_problem(problem, problem_args_file)

    MOEA = get_MOEA(moea, moea_args_file)
    MOEA.problem = Problem
    MOEA.mutation = get_mutation(mutation, mutation_args_file)
    MOEA.crossover = get_crossover(crossover, crossover_args_file)
    MOEA.selection = get_selection(selection)
    MOEA.sparsity = get_sparsity(sparsity)

    Framework   = get_framework(framework, framework_args_file)
    Framework.problem = Problem
    if framework == "M6":
        Framework.problem_np = get_problem_pymoo(problem, problem_args_file)
        MOEA.R = Framework.R
    Framework.EMO = MOEA

    for i in range(1):
        P = Framework.run()
        X = []
        F = []
        for p in P:
            X.append(p.decisionVariables)
            F.append(p.objectives)
    
        print(F)