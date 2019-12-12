import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 5)
#%%
def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

def sample_beta_0(initb0, s, b1, b2, m, v, gtw, PA, IQ):
    bnew=np.random.normal(initb0, s)
    p=np.min([1, np.product((1+np.exp((1-2*gtw)*(initb0+b1*PA+b2*IQ)))/(1+np.exp((1-2*gtw)*(bnew+b1*PA+b2*IQ))))*np.exp((np.square(initb0-m)-np.square(bnew-m))/(2*v))])
    if np.random.binomial(1, p)==1: return bnew
    else: return initb0
def sample_beta_1(initb1, s, b0, b2, m, v, gtw, PA, IQ):
    bnew=np.random.normal(initb1, s)
    p=np.min([1, np.product((1+np.exp((1-2*gtw)*(b0+initb1*PA+b2*IQ)))/(1+np.exp((1-2*gtw)*(b0+bnew*PA+b2*IQ))))*np.exp((np.square(initb1-m)-np.square(bnew-m))/(2*v))])
    if np.random.binomial(1, p)==1: return bnew
    else: return initb1
def sample_beta_2(initb2, s, b0, b1, m, v, gtw, PA, IQ):
    bnew=np.random.normal(initb2, s)
    p=np.min([1, np.product((1+np.exp((1-2*gtw)*(b0+b1*PA+initb2*IQ)))/(1+np.exp((1-2*gtw)*(b0+b1*PA+bnew*IQ))))*np.exp((np.square(initb2-m)-np.square(bnew-m))/(2*v))])
    if np.random.binomial(1, p)==1: return bnew
    else: return initb2
#%%
#Data Simulation
beta_0_true = 0
beta_1_true = 1
beta_2_true = 2

N = 1000
PA = np.random.normal(0, 1, N)
IQ = np.random.normal(0, 1, N)
p = sigmoid(beta_0_true+beta_1_true*PA+beta_2_true*IQ)
gtw=np.random.binomial(1, p)
#%%
synth_plot = plt.plot(PA+2.0*IQ, p, "o")
plt.xlabel("prediction")
plt.ylabel("gtw")
#%%
## specify initial values
init = {"beta_0": -1,
        "beta_1": -1,
        "beta_2": -1}

## specify hyper parameters
hypers = {"mu_0": 0,
         "tau_0": 0.1,
         "mu_1": 1,
         "tau_1": 0.1,
         "mu_2": 2,
         "tau_2": 0.1}
spread=0.2
#%%
def gibbs(gtw, PA, IQ, iters, init, hypers, spread):
    
    beta_0 = init["beta_0"]
    beta_1 = init["beta_1"]
    beta_2 = init["beta_2"]
    
    trace = np.zeros((iters, 3)) ## trace to store values of beta_0, beta_1, tau
    
    for it in range(iters):
        beta_0 = sample_beta_0(beta_0, spread, beta_1, beta_2, hypers["mu_0"], hypers["tau_0"], gtw, PA, IQ)
        beta_1 = sample_beta_1(beta_1, spread, beta_0, beta_2, hypers["mu_1"], hypers["tau_1"], gtw, PA, IQ)
        beta_2 = sample_beta_2(beta_2, spread, beta_0, beta_1, hypers["mu_2"], hypers["tau_2"], gtw, PA, IQ)
        trace[it,:] = np.array((beta_0, beta_1, beta_2))
        
    trace = pd.DataFrame(trace)
    trace.columns = ['beta_0', 'beta_1', 'beta_2']
        
    return trace
#%%
iters = 10000
trace = gibbs(gtw, PA, IQ, iters, init, hypers, spread)
#%%
traceplot = trace.plot()
traceplot.set_xlabel("Iteration")
traceplot.set_ylabel("Parameter value")
#%%
trace_burnt = trace[9500:9999]
hist_plot = trace_burnt.hist(bins = 30, layout = (1,3))
#%%
from sklearn.linear_model import LogisticRegression
X= np.transpose([np.ones(N), PA, IQ])
y=gtw
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto').fit(X, y)
np.round(clf.coef_,2)
