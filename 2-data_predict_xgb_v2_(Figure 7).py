"""
Program for training and testing the proposed scheme
"""
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


# setting up the text in plots
font = {'size': 18, 'family': 'DeJavu Serif', 'serif': ['Palatino']}
plt.rc('font', **font)
params = {'legend.fontsize': 14, 'lines.markersize': 5}
plt.rcParams.update(params)
plt.rc('text', usetex=True)


with open('./data_generated/data01.pkl', 'rb') as fp:
    data01 = pickle.load(fp)
fp.close()
del fp

with open('./data_generated/data02.pkl', 'rb') as fp:
    data02 = pickle.load(fp)
fp.close()
del fp

with open('./data_generated/data03.pkl', 'rb') as fp:
    data03 = pickle.load(fp)
fp.close()
del fp

data = {}
for key, val in data01.items():
    if key == "ar" or key == "fs":
        data[key] = np.vstack([data01[key], data02[key], data03[key]])
    else:
        data[key] = np.hstack([data01[key], data02[key], data03[key]])


# Stacking vp, vs, phi data as predictors and aspect-ration as target
Xtr_ = np.vstack([data['Vp'], data['Vs'], data['phi']]).T
ytr_ = np.array(data['ar'])[:, 1]

"""
1) Uncomment line 53 and 130 for figure 7a)
2) Alternatively, uncomment line 56 and 131 for figure 7b)
"""
# # Test 0.1 aspect-ratio data
# Xtr, Xts, ytr, yts = Xtr_[2*60000:3*60000, :], Xtr_[0*60000:1*60000, :], ytr_[2*60000:3*60000], ytr_[0*60000:1*60000]

# # Test 0.2 aspect-ratio data
Xtr, Xts, ytr, yts = Xtr_[2*60000:3*60000, :], Xtr_[1*60000:2*60000, :], ytr_[2*60000:3*60000], ytr_[1*60000:2*60000]


"""
Training
"""
# Call trainer
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror',
                                 max_depth=500,
                                 n_estimators=10)
xgb_regressor.fit(X=Xtr, y=ytr)
print("Trained XGB")
pred = xgb_regressor.predict(Xts)

r2train = r2_score(y_true=ytr, y_pred=xgb_regressor.predict(Xtr))
r2test = r2_score(y_true=yts, y_pred=pred)
print("Train score = {}".format(xgb_regressor.score(X=Xtr, y=ytr)))
print("Test score = {}".format(xgb_regressor.score(X=Xts, y=yts)))

with open('./data_generated/xgb_use.pkl', 'wb') as fp:
    pickle.dump(xgb_regressor, fp)
print("Done saving model!")


"""
Plot the observed and predicted aspect ratio
"""


def error_plot(x, x2, save_here, num_bins, color):
    plt.figure()
    # x -------------------------------------------------------------------------------------------------------
    n, bins, patches = plt.hist(x, num_bins,
                                density=True,  # edgecolor='black',
                                color="blue",
                                alpha=0.7,
                                label=r'$\Lambda_{' + 'obs' + '}$')

    sigma = np.std(x)
    mu = np.mean(x)
    max_ = 0.1  # bins[np.argmax(n)]
    min_ = np.min(x)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    # plt.plot(bins, y, '--', color='blue')

    # x2 -------------------------------------------------------------------------------------------------------
    n2, bins2, patches2 = plt.hist(x2, num_bins,
                                   density=True,  # edgecolor='black',
                                   color="orange",
                                   alpha=0.7,
                                   label=r'$\Lambda_{' + 'pred' + '}$')

    sigma2 = np.std(x2)
    mu2 = np.mean(x2)
    max2_ = bins2[np.argmax(n2)]
    min_ = np.min(x2)
    y2 = ((1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * (1 / sigma2 * (bins2 - mu2)) ** 2))

    plt.grid('.-')
    plt.xlim([-0.0001, 1.0001])
    plt.ylim([-0.0001, 11.001])
    plt.xlabel(r'$\Lambda$')

    plt.axvline(x=max2_, color='r', label=r'$\Lambda_{' + 'deduced' + '}$' + '={:.2f}'.format(max2_),
                ymax=20)  # '{:.2f}'.format(5.39120)

    plt.legend(loc=1)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_here, dpi=400)
    plt.close()


# Save the plots
# save_here = "./results/show_ar_01_predicted.png"    # Save result for 0.1 aspect-ratio data
save_here = "./results/show_ar_02_predicted.png"  # Save result for 0.2 aspect-ratio data
error_plot(yts, pred, save_here, 30, "blue")
