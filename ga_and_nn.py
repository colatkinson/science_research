#!/bin/env python2
"""A script to use GA to evolve MLP for financial prediction."""
from __future__ import division, print_function
from pyevolve import (G1DList, GSimpleGA, GAllele, Selectors, Mutators,
                      Initializators)
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer, SigmoidLayer
import pickle
import random
random.seed()
import re
import matplotlib.pyplot as plt
from decimal import Decimal

_PREFIX = "nn_ga_"
_FAST = False


class FinanceNet(object):

    """A class to represent a financial MLP."""

    def parse_data(self, filen, begin=1000, end=1500):
        """Parse data from file name in the format from the Federal Reserve."""
        arr = []
        with open(filen) as data_file:
            lines_old = []
            for line in data_file.read().split("\n")[begin:end]:
                splt = re.split(r"\s+", line.lstrip())
                if len(splt) >= 2:
                    if splt[1] != "ND":  # Remove points with no data
                        if len(lines_old) != 0:
                            if len(arr) >= 1:
                                arr[-1][3] = splt[1]
                            if len(lines_old) >= 5:
                                arr_tmp = [float(lines_old[0][1]),
                                           float(lines_old[2][1]),
                                           float(lines_old[4][1]), 0]
                                arr.append(arr_tmp)
                        lines_old.insert(0, splt)
        return arr

    def make_dataset(self, arr):
        """Generate a dataset from the data."""
        data = SupervisedDataSet(3, 1)
        for i in arr:
            if len(i) == 4:
                data.addSample(i[0:3], i[3])
        return data

    def training(self, data):
        """Train the network on the data."""
        net = buildNetwork(data.indim, self.hidden_nodes, data.outdim,
                           hiddenclass=self.layer_type, bias=self.bias,
                           fast=self.fast)
        net.randomize()
        trnr = BackpropTrainer(net, dataset=data, momentum=self.momentum,
                               verbose=False, weightdecay=0.01)
        trnr.trainUntilConvergence(maxEpochs=self.epochs)

        return (trnr, net)

    def plot(self):
        """Plot the results using matplotlib."""
        y_data = self.y_data
        x_data = range(len(y_data))
        y2_data = self.y2_data

        plt.plot(x_data, y_data, linestyle="dashed", marker="o", color="green",
                 label="Predicted")
        plt.plot(x_data, y2_data, linestyle="solid", marker="o", color="red",
                 label="Actual")
        plt.ylabel("Exchange Rate (USD)")
        plt.xlabel("Day")
        plt.title("German Mark/USD Exchange Rate vs Time")
        plt.legend(loc="lower right")
        plt.show()

    def test(self, trnr, net):
        """Test the net on the data, and return the NMSE."""
        trnr.testOnData(self.tstdata, verbose=False)
        data = self.tstdata
        arr = [net.activate(x) for x, _ in data]

        # Now to plot the data
        y_data = arr
        y2_data = []
        for i in data["input"]:
            y2_data.append(i[0])
        self.y_data = y_data
        self.y2_data = y2_data

        nmse = self.calc_nmse(y_data, y2_data)

        print("NMSE: %s" % nmse)

        return nmse

    def load_data(self):
        """Load data from file and split it into testing and training."""
        alldata = self.make_dataset(self.parse_data("dat96_ge.txt"))
        tstdata, trndata = alldata.splitWithProportion(0.25)

        return (alldata, tstdata, trndata)

    def calc_nmse(self, actual, predicted):
        """Calculate the NMSE given lists of actual and predicted."""
        total = 0
        if len(actual) == len(predicted):
            for i in range(0, len(actual)):
                total += abs(actual[i]-predicted[i])
            try:
                return total/len(actual)
            except ZeroDivisionError:
                return 1000  # arbitrary big number to prevent selection
        else:
            print("Actual and predicted arrays must have the same length")
            return -1.0

    def calc_up_down_acc(self, actual, predicted):
        """Calculate the up/down accuracy given actual and predicted."""
        total = 0
        if len(actual) == len(predicted):
            for i in range(0, len(actual)-1):
                delta_act = actual[i+1] - actual[i]
                delta_pre = predicted[i+1] - predicted[i]
                if delta_act/abs(delta_act) == delta_pre/abs(delta_pre):
                    total += 1
            try:
                return total/(len(actual)-1)*100
            except ZeroDivisionError:
                return 0  # if zero division happens, just give up
        else:
            print("Actual and predicted arrays must have the same length")
            return -1.0

    def run(self):
        """Execute the NN (train and calculate NMSE)."""
        if self.newnet:
            self.trained, self.net = self.training(self.trndata)
        else:
            self.trained = 0
            self.net = 0
            with open(self.fn_prefix+"_network", "r") as data_file:
                self.net = pickle.load(data_file)
            with open(self.fn_prefix+"_trainer", "r") as data_file:
                self.trained = pickle.load(data_file)
        self.nmse = self.test(self.trained, self.net)

    def __init__(self, prefix, newnet=True, fast=False, days=500,
                 hidden_nodes=6, epochs=30, bias=True, momentum=0.1,
                 layer=TanhLayer):
        """Initialize the class."""
        self.alldata, self.tstdata, self.trndata = self.load_data()
        self.fn_prefix = prefix
        self.newnet = newnet
        self.days = days
        self.hidden_nodes = hidden_nodes
        self.epochs = epochs
        self.bias = bias
        self.momentum = momentum
        self.layer_type = layer
        self.fast = fast

        self.trained = 0
        self.y_data = []
        self.y2_data = []
        self.nmse = 0.0
        self.net = 0


def evolve_callback(ga_engine):
    """Callback for each generation. Prints %% progress."""
    generation = ga_engine.getCurrentGeneration()
    if generation % 100 == 0:
        print("Current generation: %d" % (generation,))
        print(ga_engine.getStatistics())
    return False


def gen_net(chromosome, fast=False):
    """Generate a network from the given chromosome."""
    net = FinanceNet(_PREFIX, days=chromosome[0],
                     hidden_nodes=chromosome[1],
                     epochs=chromosome[2], bias=chromosome[3],
                     momentum=chromosome[4], layer=chromosome[5],
                     fast=fast)
    return net


def eval_func(chromosome):
    """Calculate score of chromosome."""
    score = 0.0
    net = gen_net(chromosome, chromosome.getParam("fast"))
    net.run()
    score = 1/net.nmse  # take inverse so lower is better
    return score


def save_class(inst, prefix):
    """Pickle the given neural network instance."""
    with open(prefix+"_class", "w") as data_file:
        pickle.dump(inst, data_file)
    return True


def load_class(prefix):
    """Load a pickled neural network class instance."""
    with open(prefix+"_class") as data_file:
        return pickle.load(data_file)


def run_ga(fast):
    """Initialize and run the genetic algorithms."""
    alleles = GAllele.GAlleles()                    # create list of all alleles
    alleles.add(GAllele.GAlleleRange(1, 1000))    # range for the number of days
    alleles.add(GAllele.GAlleleRange(1, 20))            # range for hidden nodes
    alleles.add(GAllele.GAlleleRange(1, 10))                  # number of epochs
    alleles.add(GAllele.GAlleleList([True, False]))                       # bias
    alleles.add(GAllele.GAlleleList([float(x * Decimal("0.1"))
                                     for x in xrange(0, 10)]))        # momentum
    alleles.add(GAllele.GAlleleList([TanhLayer, SigmoidLayer]))   # tanh/sigmoid

    genome = G1DList.G1DList(len(alleles))
    genome.setParams(allele=alleles, fast=fast)
    genome.evaluator.set(eval_func)

    genome.mutator.set(Mutators.G1DListMutatorAllele)
    genome.initializator.set(Initializators.G1DListInitializatorAllele)

    gen_alg = GSimpleGA.GSimpleGA(genome)
    gen_alg.setMultiProcessing(not fast)
    gen_alg.selector.set(Selectors.GRouletteWheel)
    gen_alg.setGenerations(20)

    gen_alg.evolve(freq_stats=0)
    return gen_alg.bestIndividual()


def multiprocess_run():
    """Run with GA multiprocessing."""
    main(False)


def arac_run():
    """Run with accelerated NN (arac)."""
    main(True)


def main(fast):
    """Main function."""
    if fast:
        print("Running in fast mode (arac)")
    else:
        print("Running in slow mode")

    res = run_ga(fast)
    print(res)
    print(res.genomeList)
    net = gen_net(res.genomeList)
    net.run()
    net.plot()
    save_class(net, _PREFIX)

if __name__ == "__main__":
    main(_FAST)
