import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def varma():
    dta = sm.datasets.webuse('lutkepohl2', 'https://www.stata-press.com/data/r12/')
    dta.index = dta.qtr
    dta.index.freq = dta.index.inferred_freq
    endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]

    mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(1, 1))
    res = mod.fit(maxiter=1000, disp=False)
    print(res.summary())

    ax = res.impulse_responses(10, orthogonalized=True, impulse=[1, 0]).plot(figsize=(13, 3))
    ax.set(xlabel='t', title='Responses to a shock to `dln_inv`');
    print(ax)