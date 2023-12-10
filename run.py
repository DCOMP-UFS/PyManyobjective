import json
import time

from src.MOEAs.GA import *
from src.MOEAs.NSGAII import *
from src.MOEAs.MRGA import *

from src.MOEAs.mutations.PolynomialMutation import PolynomialMutation

from src.MOEAs.crossovers.SBXCrossover import SBXCrossover

from src.MOEAs.sparsities.CrowdingDistance import CrowdingDistance

from src.BinaryTournament import BinaryTournament

from src.frameworks.M1 import M1
from src.frameworks.M1_linear import M1 as M1_linear
from src.frameworks.M3 import M3
from src.frameworks.M6 import M6
from src.frameworks.SMO import SMO
from src.frameworks.SMB import SMB
#from src.frameworks.MARSAOP import MARSAOP
from src.frameworks.Dummy import Dummy

from tensorflow.keras.datasets import mnist

from src.problems.keras_nn import Keras_NN
from src.problems.keras_cnn import Keras_CNN
from src.problems.SVM_hyperparameters import SVM_hyperparameters
from src.problems.SVM_hyperparameters_sen_spe import SVM_hyperparameters_sen_spe
from src.problems.SVM_hyperparameters_sen_spe_2 import SVM_hyperparameters_sen_spe_2
from src.problems.DTLZ import DTLZ2

from tensorflow.keras import utils

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pymoo.performance_indicator.kktpm import KKTPM
from pymoo.factory import get_performance_indicator

from pathlib import Path

def get_sparsity(sparsity):
    if sparsity == "CrowdingDistance":
        return CrowdingDistance()

    print("unknown crowding")
    return None

def get_selection(selection, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)["Selection"]

    if selection == "Binary":
        return BinaryTournament(args["tournament_size"])

    print("unknown selection")
    return None

def get_mutation(mutation, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)["Mutation"]

    if mutation == "Polynomial":
        return PolynomialMutation(mutationProbability=args["probability"], distributionIndex=args["distribution"])

    print("unknown mutation")
    return None

def get_crossover(crossover, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)["Crossover"]

    if crossover == "SBX":
        return SBXCrossover(distributionIndex=args["distribution"], crossoverProbability=args["probability"])

    print("unknown crossover")
    return None

def get_framework(framework, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)["Framework"]

    if framework == "M1": 
        return M1(None, None, args["sample_size"], args["tau"], args["SEmax"])
    if framework == "M3":
        return M3(None, None, args["sample_size"], args["SEmax"], args["k"], args["alfa"], args["R"])
    if framework == "M6":
        return M6(None, None, None, args["sample_size"], args["SEmax"], args["R"], KKTPM())

    if framework == "SMO":
        return SMO(None, args["sample_size"], args["tmax"], args["tparada"], None)
    if framework == "SMB":
        return SMB(None, args["sample_size"], args["tmax"], args["tparada"], None)

    if framework == "M1_linear": 
        return M1_linear(None, None, args["sample_size"], args["tau"], args["SEmax"])

    if framework == "MARSAOP":
        return MARSAOP(None, args["Gmax"], args["prob"], args["variance"], args["wv"], args["thr1"], args["thr2"], args["batch_size"])

    if framework == "Dummy":
        return Dummy(None, None)

    print("unknown framework")

def get_MOEA(MOEA, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)["MEMO"]

    if MOEA == "GA":
        return GA(None, args["maxEvaluations"], args["populationSize"], args["offspringSize"], None, None, None, None)
    if MOEA == "NSGAII": 
        return NSGAII(None, args["maxEvaluations"], args["populationSize"], args["offspringSize"], None, None, None, None)
    if MOEA == "MRGA": 
        return MRGA(None, args["maxEvaluations"], args["populationSize"], args["offspringSize"], None, None, None, None, R=None)

    print("unknown problem")
    return None

def interval_string_to_average_int(s):
    interval = [float(x) for x in s.split('-')]
    return (interval[0] + interval[1]) / 2.0

def get_problem(problem, args_file):
    args = None
    with open(args_file) as fargs:
        args = json.load(fargs)["Problem"]

    if problem == "DTLZ2": 
        return DTLZ2(args["m"], args["k"])
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

    if problem == "CNN_MNIST":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(60000, 28, 28, 1)
        X_test = X_test.reshape(10000, 28, 28, 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        y_train = utils.to_categorical(y_train)
        y_test = utils.to_categorical(y_test)

        return Keras_CNN(X_train, y_train, X_test, y_test, (28, 28, 1), 10)

    if problem == "SVM_hyperparameters":
        d = datasets.load_digits()
        X = d.data
        y = d.target
        n_samples = len(X)
        return SVM_hyperparameters(X, y)
    
    if problem == "SVM_hyperparameters_boston":
        d = datasets.load_boston()
        X = d.data
        y = data.target
        return SVM_hyperparameters(X, y)

    if problem == "SVM_hyperparameters_statlog":
        X_ls = []
        y_ls = []
        with open("datasets/heart.dat", "r") as statlog_dat_file:
            lines = statlog_dat_file.readlines()
            for line in lines:
                row_data = [float(x) for x in line.split()]
                X_ls.append(row_data[:-1])
                y_ls.append(row_data[-1])
        X = np.array(X_ls)
        transformer = MinMaxScaler()
        transformer.fit(X)
        X = transformer.transform(X)
        y = np.array(y_ls)
        return SVM_hyperparameters_sen_spe_2(X, y)

    if problem == "SVM_hyperparameters_breast":
        X_ls = []
        y_ls = []
        with open("datasets/breast-cancer.dat", "r") as breast_dat_file:
            for line in breast_dat_file.readlines():
                row_data = [x for x in line.split(',')]
                print(row_data)
                # 1. Class: no-recurrence-events, recurrence-events
                x = []
                if row_data[0] == "no-recurrence-events":
                    x.append(0)
                else:
                    x.append(1)
                # 2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
                x.append(interval_string_to_average_int(row_data[1]))
                # 3. menopause: lt40, ge40, premeno.
                if row_data[2] == "lt40":
                    x.append(0)
                elif row_data[2] == "ge40":
                    x.append(1)
                else:
                    x.append(2)
                # 4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59.
                x.append(interval_string_to_average_int(row_data[3]))
                # 5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39.
                x.append(interval_string_to_average_int(row_data[4]))
                # 6. node-caps: yes, no.
                if row_data[5] == "no":
                    x.append(0)
                else:
                    x.append(1)
                # 7. deg-malig: 1, 2, 3.
                x.append(float(row_data[6]))
                # 8. breast: left, right.
                if row_data[7] == "left":
                    x.append(0)
                else:
                    x.append(1)
                # 9. breast-quad: left-up, left-low, right-up, right-low, central.
                if row_data[8] == "left_up":
                    x.append(0)
                elif row_data[8] == "left_low":
                    x.append(1)
                elif row_data[8] == "right_up":
                    x.append(2)
                elif row_data[8] == "right_low":
                    x.append(3)
                else:
                    x.append(4)
                print("")
                print(row_data[8])
                print(x[-1])
                print("")
                # 10. irradiat: yes, no.
                if row_data[9][0] == 'n':
                    y_ls.append(0)
                else:
                    y_ls.append(1)
                X_ls.append(x)
            X = np.array(X_ls)
            transformer = MinMaxScaler()
            transformer.fit(X)
            X = transformer.transform(X)
            y = np.array(y_ls)
            return SVM_hyperparameters_sen_spe_2(X, y)

    if problem == "SVM_hyperparameters_haberman":
        X_ls = []
        y_ls = []
        with open("datasets/haberman.dat", "r") as iris_dat_file:
            for line in iris_dat_file.readlines():
                row_data = [x for x in line.split(',')]
                X_ls.append([float(x) for x in row_data[:-1]])
                if row_data[-1] == "1\n":
                    y_ls.append(0)
                else:
                    y_ls.append(1)
        X = np.array(X_ls)
        transformer = MinMaxScaler()
        transformer.fit(X)
        X = transformer.transform(X)
        y = np.array(y_ls)

        return SVM_hyperparameters_sen_spe_2(X, y)

    print("unknown problem")
    return None

def save(name, i, P, F, T):
    Path("results/" + name).mkdir(parents=True, exist_ok=True)
    with open("results/" + name + "/P" + i, 'w+') as P_file:
        P_file.write(json.dumps(P))
    with open("results/" + name + "/F" + i, 'w+') as F_file:
        F_file.write(json.dumps(F))
    with open("results/" + name + "/T" + i, 'w+') as T_file:
        T_file.write(json.dumps(T))

def run(framework, problem, moea, crossover, mutation, selection, sparsity, n, args_file, save_dir=None):
    if save_dir == None:
        save_dir = framework + "_" + problem + "_" + moea
        print("Save directory not provided, saving to:", save_dir)

    Problem = get_problem(problem, args_file)

    MOEA = get_MOEA(moea, args_file)
    MOEA.problem = Problem
    MOEA.mutation = get_mutation(mutation, args_file)
    MOEA.crossover = get_crossover(crossover, args_file)
    MOEA.selection = get_selection(selection, args_file)
    MOEA.sparsity = get_sparsity(sparsity)

    Framework = get_framework(framework, args_file)
    Framework.problem = Problem
    Framework.EMO = MOEA

    for i in range(n):
        print("At {}-th run".format(i + 1))

        t1 = time.process_time()
        P = Framework.run()
        t2 = time.process_time()
        T = t2 - t1

        X = P.decisionVariables.tolist()
        F = P.objectives.tolist()

        save(save_dir, str(i), X, F, T)