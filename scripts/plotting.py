# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 17:59:35 2025

@author: Dunka
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def configure_plotting():
    mpl.rcParams.update({
        "figure.figsize": (5, 3.5),
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 12,
        "xtick.direction": "in",
        "xtick.top": True,
        "ytick.direction": "in",
        "ytick.right": True,
    })
    # optional grid style
    mpl.rc("grid", linestyle="--", linewidth=0.5, alpha=0.7)

def plot_mtf_curves(frequencies, curves, Y_vals, filename, linestyle_list):
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$f/f_{\mathrm{cutoff}}$")
    ax.set_ylabel("modulation")
    # ax.set_xlim(0, 2.2)

    for curve, ls in zip(curves, linestyle_list):
        ax.plot(frequencies, curve, ls)

    labels = [f"{y:.2f}" for y in Y_vals]
    ax.legend(labels, title="Y values", loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    fig.savefig(filename)

def plot_fractions(NZ_BW, curves,y_label, labels, filename,linestyle):
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$f/f_{\mathrm{cutoff}}$")
    ax.set_ylabel(y_label)
    # ax.set_xlim(0, 2.2)

    for curve, ls in zip(curves, linestyle):
        ax.plot(NZ_BW, curve, ls)
    if not labels:
        pass
    else:
        ax.legend(labels, loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    fig.savefig(filename)
