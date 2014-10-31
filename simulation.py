# simulation.py
"""A script to test existing data with a market simulation."""
from __future__ import division
from ga_and_nn import *


def simul():
    """Run the market simulation."""
    n = load_class("mark2")
    data = n.tstdata["input"]
    cash_init = 10000
    cash = cash_init
    shares = 0
    savings = 0
    bh_cash = 10000
    bh_shares = int(bh_cash/data[0][0])
    bh_cash -= bh_shares*data[0][0]
    print("Buy Hold: %s %s" % (bh_cash, bh_shares))
    stock_val = []
    for i in data:
        stock_val.append(i[0])
    cash_arr = []
    cash_stock = []

    for i in xrange(0, len(data)-1):
        v = n.net.activate(data[i])
        delta = v - data[i][0]
        sign = delta/abs(delta)
        if sign > 0:
            if cash - data[i][0] > 0:
                shares += int(cash/data[i][0])
                cash -= shares*data[i][0]
        elif sign < 0:
            cash += shares*data[i][0]
            shares = 0
        """if cash > cash_init:
            savings += cash-cash_init
            cash = cash_init"""
        print(cash, shares)
        cash_arr.append(cash)
        cash_stock.append(cash+data[i][0]*shares)
    cash += shares*data[-1][0]
    shares = 0
    bh_cash += bh_shares*data[-1][0]
    bh_shares = 0
    print("\nResults")
    print("NN: %s" % (cash+savings))
    print("B/H: %s" % bh_cash)
    plot_data(cash_arr, cash_stock, stock_val)


def plot_data(cash, cash_shares, stock_val):
    """Plot the simulation data."""
    x = xrange(0, len(cash))
    plt.plot(x, cash, linestyle="dashed", marker="o",
             color="green", label="Cash")
    plt.plot(x, cash_shares, linestyle="solid", marker="o",
             color="red", label="Cash+Shares")
    plt.ylabel("Cash")
    plt.xlabel("Day")
    plt.title("German Mark/USD Exchange Rate vs Time")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    simul()
