import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

bx = pd.read_csv('/home/me/Documents/Jupyter/bsdata_xstats.csv')
K = 5
dfresults = pd.DataFrame()
# For each category vector of BX
for vectorName in bx.vector.unique() :
    df = bx[bx.vector==vectorName]
    # Divide by K knots
    kLen = df.vector.size // K
    # For each knot interval
    start = 0
    end = 0
    lbl = 1
    dfrow = pd.DataFrame({"vector":[vectorName]})
    while end < len(df.index):
        end = start + kLen
        if end + kLen > len(df.index):
            end = len(df.index)
        dfk = df[start:end-1]
        kVal = df.pin.iloc[end-1]
        # 3-rd order polynomial
        reg3 = smf.ols(formula='perc_rank ~ 1 + pin + I(pin ** 2.0) + I(pin ** 3.0)', data=dfk).fit()

        dfrow = pd.concat([dfrow, pd.DataFrame({"kend_"+str(lbl): [kVal],
                                                 "B0_"+str(lbl): [reg3.params[0]],
                                                 "X1_"+str(lbl): [reg3.params[1]],
                                                 "X2_"+str(lbl): [reg3.params[2]],
                                                 "X3_"+str(lbl): [reg3.params[3]]})], axis=1)
        start += kLen
        lbl += 1

    dfresults = pd.concat([dfresults,dfrow], ignore_index=1)

dfresults.to_csv('/home/me/Documents/Jupyter/bsdata_regress.csv')
