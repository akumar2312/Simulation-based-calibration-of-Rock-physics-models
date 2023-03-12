"""
Predict on observed data
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt


# setting up the text in plots
font = {'size': 18, 'family': 'DeJavu Serif', 'serif': ['Palatino']}
plt.rc('font', **font)
params = {'legend.fontsize': 14, 'lines.markersize': 5}
plt.rcParams.update(params)
plt.rc('text', usetex=True)


def error_plot(x2, save_here, num_bins, color):
    plt.figure()
    n2, bins2, patches2 = plt.hist(x2, num_bins,
                                   density=True,  # edgecolor='black',
                                   color="orange",
                                   alpha=0.7,
                                   label=r'$\Lambda_{' + 'predicted' + '}$')

    sigma2 = np.std(x2)
    mu2 = np.mean(x2)
    max_ = bins2[np.argmax(n2)]
    min_ = np.min(x2)
    y2 = ((1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * (1 / sigma2 * (bins2 - mu2)) ** 2))
    # plt.plot(bins2, y2, '--', color='orange')


    plt.xlim([-0.0001, 1.0001])
    plt.ylim([-0.0001, 8.001])
    plt.xlabel(r'$\Lambda$')
    # plt.ylabel(r'$f(AR)$')
    plt.axvline(x=max_, color='r', label=r'$\Lambda_{' + 'Interparticle' + '}$' + '={:.2f}'.format(max_),
                ymax=20)  # '{:.2f}'.format(5.39120)

    plt.legend(loc=1)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_here, dpi=400)
    plt.close()


"""
Uncomment line 51 or 52 for generating figure 8a) and 8b) respectively
"""
s_ = "Well-1_"
# s_ = "Well-2_"
with open('./observed/' + s_ + '.pkl', 'rb') as fp:
    dfs = pickle.load(fp)

dfs.drop(dfs.loc[dfs['dtco'] == -999].index, inplace=True)
dfs.drop(dfs.loc[dfs['dtsm'] == -999].index, inplace=True)
dfs.drop(dfs.loc[dfs['nphi'] == -999].index, inplace=True)
dfs = dfs.reset_index(drop=True)

comp_vel = dfs['dtco']
shear_vel = dfs['dtsm']
porosity = dfs['nphi']

comp_vel = 0.3048 * 1e-3 / (comp_vel / 1e6)  # usec/ft -> km/s
shear_vel = 0.3048 * 1e-3 / (shear_vel / 1e6)  # usec/ft -> km/s


# Vp, Vs, phi are combined
Xtrr_ = np.vstack([comp_vel, shear_vel, porosity]).T


# Predict AR using saved xgb model
with open('./data_generated/xgb_use.pkl', 'rb') as fp:
    xgb = pickle.load(fp)
print("Done loading model!")

pred = xgb.predict(Xtrr_)

save_here = "./results/" + s_ + ".png"
error_plot(pred, save_here, 50, "blue")
