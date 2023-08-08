# Import all the packages you will need
import os
import sklearn
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVR
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression

# Set this to whatever directory you keep your data and import
os.chdir("/enter/directory/here")

master = pd.read_csv("Combined final.csv")
master_spec = master.iloc[:,3:]

# Use the portion of the spectra you need for malvidin-3-glucoside (M3G)
m3g = master.drop(["TIPs", "PCs"], axis = 1)
m3g = master_spec.iloc[:,200:471]
m3g_final = m3g.assign(M3Gs = master[['M3Gs']])

X = m3g_final.iloc[:,0:271]
y = m3g_final[['M3Gs']]

# Kernel Ridge Regression

krr_linear = KernelRidge(kernel = 'linear', alpha = 0.0004)

krr_m3g_fit_1 = krr_linear.fit(X[0:241], y[0:241]).predict(X[242:323])
krr_m3g_fit_1 = pd.DataFrame(krr_m3g_fit_1)

plt.scatter(y[242:323], krr_m3g_fit_1)
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.show()
r2_score(y[242:323], krr_m3g_fit_1)

# Nu Support Vector Regression

nu_svr_linear = NuSVR(C=1200, nu=1)
nu_svr_fit = nu_svr_linear.fit(X[0:241], y[0:241].values.ravel()).predict(X[242:323])
plt.scatter(y[242:323], nu_svr_fit)
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.show()
r2_score(y[242:323], nu_svr_fit)

# Partial least squares regression

pls2_linear = PLSRegression(n_components=4)
pls2_fit = pls2_linear.fit(X[0:241], y[0:241].values.ravel()).predict(X[242:323])
plt.scatter(y[242:323], pls2_fit)
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.show()
r2_score(y[242:323], pls2_fit)
