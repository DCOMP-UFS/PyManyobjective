import json
import time

from src.MOEAs.GA import *
from src.MOEAs.NSGAII import *
from src.MOEAs.MRGA import *
from src.MOEAs.GridSearch import *

from src.MOEAs.mutations.PolynomialMutation import PolynomialMutation

from src.MOEAs.crossovers.SBXCrossover import SBXCrossover

from src.MOEAs.sparsities.CrowdingDistance import CrowdingDistance

from src.BinaryTournament import BinaryTournament

from src.frameworks.M1_linear import M1
from src.frameworks.M3 import M3
from src.frameworks.M6 import M6
from src.frameworks.SMO import SMO
from src.frameworks.SMB import SMB
from src.frameworks.MARSAOP import MARSAOP

from tensorflow.keras.datasets import mnist

from src.problems.keras_nn import Keras_NN
from src.problems.SVM_hyperparameters import SVM_hyperparameters

from tensorflow.keras import utils

from sklearn import datasets
from sklearn.model_selection import train_test_split

from pymoo.factory import get_performance_indicator
from pymoo.indicators.kktpm import KKTPM

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

    if framework == "M1_linear": 
        return M1_linear(None, None, args["sample_size"], args["tau"], args["SEmax"])
    
    if framework == "MARSAOP":
        return MARSAOP(None, args["Gmax"], args["prob"], args["variance"], args["wv"], args["thr1"], args["thr2"])

    print("unknown framework")

def get_MOEA(MOEA, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)

    if MOEA == "GA":
        return GA(None, args["maxEvaluations"], args["populationSize"], args["offspringSize"], None, None, None, None)
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
        y_train = utils.to_categorical(y_train, nb_classes)
        y_test = utils.to_categorical(y_test, nb_classes)
        return Keras_NN(X_train, y_train, X_test, y_test, 64, 10)

    if problem == "SVM_hyperparameters":
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.45, shuffle=False)
        return SVM_hyperparameters(X_train, y_train, X_test, y_test)

    print("unknown problem")
    return None

def save(name, i, P, F, T, bestPrecision):
    Path("results/" + name).mkdir(parents=True, exist_ok=True)
    with open("results/" + name + "/P" + i, 'w+') as P_file:
        P_file.write(json.dumps(P))
    with open("results/" + name + "/F" + i, 'w+') as F_file:
        F_file.write(json.dumps(F))
    with open("results/" + name + "/T" + i, 'w+') as T_file:
        T_file.write(json.dumps(T))
    with open("results/" + name + "/bestPrecision" + i, 'w+') as precision_file:
        precision_file.write(json.dumps(bestPrecision))

def run(framework, problem, moea, crossover, mutation, selection, sparsity, n, framework_args_file, problem_args_file, moea_args_file, pareto_front_file, mutation_args_file, crossover_args_file):
    Problem = get_problem(problem, problem_args_file)

    MOEA = get_MOEA(moea, moea_args_file)
    MOEA.problem = Problem
    MOEA.mutation = get_mutation(mutation, mutation_args_file)
    MOEA.crossover = get_crossover(crossover, crossover_args_file)
    MOEA.selection = get_selection(selection)
    MOEA.sparsity = get_sparsity(sparsity)

    Framework = get_framework(framework, framework_args_file)
    Framework.problem = Problem
    Framework.EMO = MOEA

    for i in range(n):
        t1 = time.process_time()
        P = Framework.run()
        t2 = time.process_time()
        t = t2 - t1

        X = []
        F = []
        T = []
        bestPrecision = 0
        for p in P:
            X.append(p.decisionVariables)
            f = []
            for objective in p.objectives:
                f.append(1.0 - objective)
            F.append(f)
            T.append(t)

            bestPrecision = max(bestPrecision, 1.0 - p.objectives[0])
    
        save(framework + "_" + problem + "_" + moea, str(i), X, F, T, bestPrecision)