"""
Program to calibrate the rock physics model.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# Setting up the text in plots
font = {'size': 12, 'family': 'DeJavu Serif', 'serif': ['Palatino']}
plt.rc('font', **font)
params = {'legend.fontsize': 14, 'lines.markersize': 5}
plt.rcParams.update(params)
plt.rc('text', usetex=True)


def plot_scatter(x_, y_, coefs, xlab, ylab, sname):
    m, c = coefs[0], coefs[1]
    # y = mx + c => (x,y) = (0, c)
    maxlimit = np.max([x_, y_])

    leg = "Fit: " + ylab + "=" + "{:.2f}".format(m) + xlab + "+" + "{:.2f}".format(c)

    plt.axline(xy1=(0, 0), slope=1, color="k", lw=2, label="$1:1$ Line")
    plt.axline(xy1=(0, c), slope=m, color="r", label=leg, alpha=0.8)
    plt.scatter(x=x_, y=y_, marker="+", color="b", s=80, alpha=0.6)
    # label="Regression:" + ylab + "=" + "{:.2f}".format(m) + xlab + "+" + "{:.2f}".format(c))
    # plt.title("Fit: " + ylab + "=" + "{:.2f}".format(m) + xlab + "+" + "{:.2f}".format(c))
    plt.xlim([-0.1, maxlimit + 0.1])
    plt.ylim([-0.1, maxlimit + 0.1])
    plt.xlabel(xlab + " (km/s)")
    plt.ylabel(ylab + " (km/s)")
    plt.axis('scaled')
    # plt.axis('equal')
    plt.grid(ls="--")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig(sname, dpi=300)
    plt.close()


savehere = "./results/"


# soln_dicto = pickle.load(open('./Well-1_complete', 'rb')); s_ = "Well-1_"
soln_dicto = pickle.load(open('./Well-2_complete', 'rb')); s_ = "Well-2_"

df = pd.DataFrame.from_dict(data=soln_dicto)
y_Vp_model = df["Vp_model"]
x_Vp_observed = df["Vp_true"]
y_Vs_model = df["Vs__model"]
x_Vs_observed = df["Vs__true"]

# Regression based calibration
coefs_vp = np.polyfit(x=x_Vp_observed, y=y_Vp_model, deg=1)
coefs_vs = np.polyfit(x=x_Vs_observed, y=y_Vs_model, deg=1)
plot_scatter(x_=x_Vp_observed, y_=y_Vp_model, coefs=coefs_vp,
             xlab="Vp$_{" + "obs" + "}$",
             ylab="Vp$_{" + "pred" + "}$",
             sname=savehere + s_ + "model_vp_.png")
plot_scatter(x_=x_Vs_observed, y_=y_Vs_model, coefs=coefs_vs,
             xlab="Vs$_{" + "obs" + "}$",
             ylab="Vs$_{" + "pred" + "}$",
             sname=savehere + s_ + "model_vs_.png")

print("Done!")
