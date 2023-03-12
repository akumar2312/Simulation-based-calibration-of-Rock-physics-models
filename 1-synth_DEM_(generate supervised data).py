"""
Latin Hypercube Sampling for generating a variety
of combinations of the aspect ratio, volume fractions
to realize values of effective elastic properties.
"""

from scipy.stats import qmc
from dem_Berry1980 import *
import pickle
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


font = {'size': 18, 'family': 'DeJavu Serif', 'serif': ['Palatino']}
plt.rc('font', **font)
params = {'legend.fontsize': 14, 'lines.markersize': 5}
plt.rcParams.update(params)
plt.rc('text', usetex=True)

"""
α = [0.3, 0.8]
φ = np.linspace(0.01, 0.99, 10)
"""

nsamples = 60000
###############################################################
#          Porosity (between 0.01 and 0.5), Dim = 1           #
###############################################################
φ_candidate = qmc.LatinHypercube(d=1)
φ = φ_candidate.random(n=nsamples)
φ = 0.01 + (0.5 - 0.01) * (φ - np.min(φ)) / (np.max(φ) - np.min(φ))

###############################################################
#         Aspect ratio (between 0.01 and 0.8), Dim = 2        #
###############################################################
"""
1) Currently, the value of Lambda_1 is set as 0.3; this must be
changed to 0.2 and 0.1 to generate datasets corresponding to
"data02.pkl" and "data01.pkl".
2) After changing, uncomment suitable lines in 120-125 range to
save the corresponding dataset.
"""
α_candidate = qmc.LatinHypercube(d=1)
α = α_candidate.random(n=nsamples)
α = 0.01 + (1.0 - 0.01) * (α - np.min(α)) / (np.max(α) - np.min(α))
α = np.hstack([0.3 * np.ones([nsamples, 1]), α])

###############################################################
#      Volume fractions (between 0.01 and 1.0), Dim = 2       #
###############################################################
φi = qmc.LatinHypercube(d=2)
sample_φi = φi.random(n=nsamples)  # volume fractions generator
sample_φi = sample_φi / sample_φi.sum(axis=1)[:, None]

"""
Autocorrelation
"""
curr_fig, curr_ax = plt.subplots(figsize=(7, 5))
my_color = "k"
# change the color of the vlines
plot_acf(x=α[:, 1],
         lags=len(α[:, 1])-1,
         ax=curr_ax,
         color=my_color,
         vlines_kwargs={"colors": my_color},
         title=None,)
for item in curr_ax.collections:
    if type(item) == PolyCollection:
        item.set_facecolor(my_color)
plt.ylim([-1.1, 1.1])
plt.grid(ls="--")
plt.ylabel("ACF")
plt.xlabel("lag")
plt.tight_layout()
plt.savefig("./results/α_autocorr.png", dpi=300)

"""
DEM essentials
"""
# Matrix properties
Km = 55.42  # GPa
Gm = 23.20  # GPa
rhom = 2.71  # g/cm3

# Fluid properties
Kf = 3.0  # GPa
rhof = 1.0  # g/cm3

Kis = np.zeros(2, dtype=float)
Gis = np.zeros(2, dtype=float)

Vp = list()
Vs = list()
vpvs = np.zeros([nsamples, 2])

for i in range(nsamples):
    print("i:{}".format(i))
    K, G, phi = solve_DEM(α[i, :],
                          φ[i] * sample_φi[i],
                          Kis,
                          Gis,
                          Km,
                          Gm)
    rho = (1.0 - phi) * rhom + phi * rhof
    Ks = Gassmann_Ks(K, Km, Kf, phi)
    Vp.append(np.sqrt((Ks + 4.0 * G / 3.0) / rho))
    Vs.append(np.sqrt(G / rho))
    print("Vp, Vs = {}, {}".format(Vp[i][-1], Vs[i][-1]))
    vpvs[i, :] = np.array([Vp[i][-1], Vs[i][-1]])


data_frame = {"Vp": np.array(vpvs[:, 0]),       # vp velocity (dim 1)       60000
              "Vs": np.array(vpvs[:, 1]),       # vs velocity (dim 1)       60000
              "ar": np.array(α),                # aspect ratio (dim 2)      60000
              "phi": np.array(np.squeeze(φ)),   # total porosity (dim 1)    60000
              "fs": sample_φi}                  # volume fractions (dim 2)  60000

# --------------------------------------------------------------------------------------------

# with open('./data_generated/data01.pkl', 'wb') as fp:  # for 0.1's dataset
#     pickle.dump(data_frame, fp)
# with open('./data_generated/data02.pkl', 'wb') as fp:  # for 0.2's dataset
#     pickle.dump(data_frame, fp)
with open('./data_generated/data03.pkl', 'wb') as fp:  # for 0.3's dataset
    pickle.dump(data_frame, fp)

print("Done!")
