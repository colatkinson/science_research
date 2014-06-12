from __future__ import division
from ga_and_nn import *

def simul():
    n = load_class("nn_ga_2")
    data = n.tstdata["input"]
    cash_init = 10000
    cash = cash_init
    shares = 0
    savings = 0
    
    for i in xrange(0, len(data)-1):
        v = n.net.activate(data[i])
        delta = v - data[i][0]
        sign = delta/abs(delta)
        #print delta
        if sign > 0:
            if cash - data[i][0] > 0:
                shares += int(cash/data[i][0])
                cash -= shares*data[i][0]
        elif sign < 0:
            cash += shares*data[i][0]
            shares = 0
        if cash > cash_init:
            savings += cash-cash_init
            cash = cash_init
        print cash, shares
    cash += shares*data[-1][0]
    shares = 0
    #print shares
    print cash+savings

if __name__=="__main__":
    simul()
