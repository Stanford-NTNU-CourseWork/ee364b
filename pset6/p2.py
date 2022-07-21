import numpy as np
import matplotlib.pyplot as plt

def a():

    plt.vlines(0, 0, 1)
    plt.hlines(0, -1, 0)
    plt.hlines(1, 0, 1)
    plt.show()

def b():
    fix, ax = plt.subplots()
    lambd = 0.5
    ax.vlines(0, 0, lambd)
    ax.hlines(0, -1, 0)
    ax.hlines(lambd, 0, 1)
    ax.set_title("lambda F(x)")
    ax.set_yticks(ticks=[0, lambd])
    ax.set_yticklabels(labels=["0", "lambda"])
    plt.savefig(ax.title.get_text())
    plt.show()
    fix, ax = plt.subplots()

    ax.vlines(0, 0, lambd)

    xs = np.linspace(0, 1, 10)
    xs2 = np.linspace(-1, 0, 10)
    ax.plot(xs, xs+lambd)
    ax.plot(xs2, xs2+0)
    ax.set_title("x + lambda F(x)")
    ax.set_yticks(ticks=[0, lambd])
    ax.set_yticklabels(labels=["0", "lambda"])
    plt.savefig(ax.title.get_text())
    plt.show()  

    fix, ax = plt.subplots()

    ax.hlines(0, 0, lambd)

    xs = np.linspace(0, 1, 10)
    xs2 = np.linspace(-1, 0, 10)
    ax.plot(xs+lambd, xs)
    ax.plot(xs2+0, xs2)
    ax.set_title("(x + lambda F(x))^{-1}")
    ax.set_xticks(ticks=[0, lambd])
    ax.set_xticklabels(labels=["0", "lambda"])
    plt.savefig(ax.title.get_text())
    plt.show()  
b()