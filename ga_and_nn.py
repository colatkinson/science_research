from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import GAllele
from pyevolve import Selectors
from pyevolve import Mutators
from pyevolve import Initializators
from pybrain.datasets            import ClassificationDataSet, SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure           import TanhLayer
from pybrain.structure           import SigmoidLayer
from pybrain.structure           import RecurrentNetwork
from pybrain.structure           import LinearLayer
from pybrain.structure           import FullConnection
import math
import pickle
import random
random.seed()
import sys
import re
import numpy
import matplotlib.pyplot as plt
from decimal import Decimal
import imp
import timeit
"""try:
    imp.find_module("arac")
    _FAST = True
except ImportError:
    _FAST = False"""
_PREFIX = "nn_ga_"
_FAST = False


class FinanceNet:
    def parse_data(self, fn, begin=1000, end=1500):
        arr = []
        with open(fn) as f:
            lines_old = []
            for line in f.read().split("\n")[begin:end]:  #Split into lines
                splt = re.split("\s+", line.lstrip())
                if len(splt) >= 2:
                    if splt[1] != "ND":  #Remove points with no data
                        if len(lines_old) == 0:
                            chg = 0.5
                        else:
                            if float(splt[1]) > float(lines_old[0][1]):
                                chg = 1
                            elif float(splt[1]) < float(lines_old[0][1]):
                                chg = 0
                            else:
                                chg = 0
                            if len(arr) >= 1:
                                #print lines_old[-1]
                                #print lines_old[-1][2]
                                arr[-1][3] = splt[1]
                            if len(lines_old) >= 5:
                                arr_tmp = [float(lines_old[0][1]), float(lines_old[2][1]), float(lines_old[4][1]), 0]
                                arr.append(arr_tmp)
                        lines_old.insert(0, splt)
        #print len(arr)
        return arr

    def make_dataset(self, arr):
        data = SupervisedDataSet(3, 1)
        for i in arr:
            #print len(i)
            if len(i) == 4:
                #print i
                data.addSample(i[0:3], i[3])
        return data

    def training(self, data):
        n = buildNetwork(data.indim, self.hidden_nodes, data.outdim, hiddenclass=self.layer_type, bias=self.bias, fast=self.fast)
        n.randomize()
        t = BackpropTrainer(n, dataset=data, momentum=self.momentum, verbose=False, weightdecay=0.01)
        t.trainUntilConvergence(maxEpochs=self.epochs)
        #t = 0
    
        return (t, n)
    
    def plot(self):
        y = self.y
        x = range(len(y))
        y2 = self.y2
        
        plt.plot(x, y, linestyle="dashed", marker="o", color="green", label="Predicted")
        plt.plot(x, y2, linestyle="solid", marker="o", color="red", label="Actual")
        plt.ylabel("Exchange Rate (USD)")
        plt.xlabel("Day")
        plt.title("German Mark/USD Exchange Rate vs Time")
        plt.legend(loc="lower right")
        plt.show()
    
    def test(self, t, net):
        t.testOnData(self.tstdata, verbose = False)
        d2 = self.make_dataset(self.parse_data("dat96_ge.txt", begin=2000-self.days, end=2000))
        arr = [net.activate(x) for x, _ in d2]
    
        #Now to plot the data
        y = arr
        #x = range(len(arr))
        y2 = []
        for i in d2["input"]:
            y2.append(i[0])
        self.y = y
        self.y2 = y2
        #print d2["input"]
        """plt.plot(x, y, linestyle="dashed", marker="o", color="green", label="Predicted")
        plt.plot(x, y2, linestyle="solid", marker="o", color="red", label="Actual")
        plt.ylabel("Exchange Rate (USD)")
        plt.xlabel("Day")
        plt.title("German Mark/USD Exchange Rate vs Time")
        plt.legend(loc="lower right")"""
        
        #print("NMSE: %s" % self.calc_NMSE(y, y2))
    
        #show plot
        #plt.show()
        return self.calc_NMSE(y, y2)


    def load_data(self):
        alldata = self.make_dataset(self.parse_data("dat96_ge.txt"))
        tstdata, trndata = alldata.splitWithProportion( 0.25 )
        tst_len = int(len(alldata)*0.25)
    
        return (alldata, tstdata, trndata)

    def calc_NMSE(self, actual, predicted):
        t = 0
        if(len(actual) == len(predicted)):
            for i in range(0, len(actual)):
                t += abs(actual[i]-predicted[i])
            try:
                return t/len(actual)
            except ZeroDivisionError:
                return 1000 #arbitrary big number
        else:
            print("Actual and predicted arrays must have the same length")
            return -1.0
    
    def run(self):
        if self.newnet:
            self.trained, self.net = self.training(self.trndata)
            #test(trained, data2)
            """with open(self.fn_prefix+"_network", "w") as f:
                pickle.dump(self.net, f)
            with open(self.fn_prefix+"_trainer", "w") as f:
                pickle.dump(self.trained, f)"""
        else:
            self.trained = 0
            self.net = 0
            with open(self.fn_prefix+"_network", "r") as f:
                self.net = pickle.load(f)
            with open(self.fn_prefix+"_trainer", "r") as f:
                self.trained = pickle.load(f)
        self.nmse = self.test(self.trained, self.net)
    
    def __init__(self, prefix, newnet=True, fast=False, days=500, hidden_nodes=6, epochs=30, bias=True, momentum=0.1, layer=TanhLayer):
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

def evolve_callback(ga_engine):
    generation = ga_engine.getCurrentGeneration()
    if generation % 100 == 0:
        print "Current generation: %d" % (generation,)
        print ga_engine.getStatistics()
    return False

def gen_net(chromosome, fast=False):
    """This is just here so I only have to change the chromosome around
    in one place"""
    n = FinanceNet(_PREFIX, days=chromosome[0],
                    hidden_nodes=chromosome[1],
                    epochs=chromosome[2], bias=chromosome[3],
                    momentum=chromosome[4], layer=chromosome[5],
                    fast=fast)
    return n

def eval_func(chromosome):
    score = 0.0
    #print chromosome
    """net = FinanceNet("nn_ga_", days=chromosome[0],
                     hidden_nodes=chromosome[1],
                     epochs=chromosome[2], bias=chromosome[3],
                     momentum=chromosome[4], layer=chromosome[5])"""
    net = gen_net(chromosome, chromosome.getParam("fast"))
    net.run()
    score = 1/net.nmse   #to counteract the fact that higher score==better
    #print score
    return score

def save_class(inst, prefix):
    with open(prefix+"_class", "w") as f:
        pickle.dump(inst, f)
    return True

def load_class(prefix):
    with open(prefix+"_class") as f:
        return pickle.load(f)

def run_ga(fast):
    alleles = GAllele.GAlleles()    #create list of all alleles
    alleles.add(GAllele.GAlleleRange(1, 1000))  #add the allele range for the number of days, from 0-1000
    alleles.add(GAllele.GAlleleRange(1, 20))   #add the range for number of hidden nodes
    alleles.add(GAllele.GAlleleRange(1, 10))    #number of epochs (maybe raise later, but not tryna have like 1000 gens
    alleles.add(GAllele.GAlleleList([True, False])) #bias
    alleles.add(GAllele.GAlleleList([float(x * Decimal("0.1")) for x in xrange(0, 10)])) #momentum. Doing complicated stuff with Decimal to avoid error
    #alleles.add(GAllele.GAlleleList([float(x * Decimal("0.01")) for x in xrange(0, 100)])) #weight decay
    alleles.add(GAllele.GAlleleList([TanhLayer, SigmoidLayer]))
    
    genome = G1DList.G1DList(len(alleles))
    genome.setParams(allele=alleles, fast=fast)
    genome.evaluator.set(eval_func)
    
    #genome.evaluator.set(eval_func)
    genome.mutator.set(Mutators.G1DListMutatorAllele)
    genome.initializator.set(Initializators.G1DListInitializatorAllele)
    
    ga = GSimpleGA.GSimpleGA(genome)
    #ga.stepCallback.set(evolve_callback)
    #ga.setMultiProcessing(True)
    ga.setMultiProcessing(not fast)
    ga.selector.set(Selectors.GRouletteWheel)
    ga.setGenerations(20)
    
    ga.evolve(freq_stats=0)
    #print ga.bestIndividual()
    #print ga.bestIndividual().genomeList
    return ga.bestIndividual()

def multiprocess_run():
    main(False)

def arac_run():
    main(True)

def main(fast):
    #print _FAST
    if fast:
        print("Running in fast mode (arac)")
    else:
        print("Running in slow mode")
    #raw_input()
    bi = run_ga(fast)
    print bi
    print bi.genomeList
    """net = FinanceNet(_PREFIX, days=chromosome[0],
                     hidden_nodes=chromosome[1],
                     epochs=chromosome[2], bias=chromosome[3],
                     momentum=chromosome[4], layer=chromosome[5])"""
    net = gen_net(bi.genomeList)
    net.run()
    #net.plot()
    save_class(net, _PREFIX)

if __name__ == "__main__":
    #_FAST = False
    #print "Multiprocess"
    #print timeit.Timer(lambda: multiprocess_run(), "gc.enable()").timeit(1)
    #_FAST = True
    #print "arac"
    #print timeit.Timer(lambda: arac_run(), "gc.enable()").timeit(1)
    main(_FAST)
