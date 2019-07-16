'''
__author__: Xuesong Wang, Ning Qian
    2019 - 07 -01
    this file is to calculate correlation coeffcients between inputs, or between inputs and the outputs
    a high coefficient(>0.7) means that the inputs are highly related or the input is related to the output
    This helps filter out redundant inputs and inputs unrelated to the output
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../train_data.csv")
# calculate correlation coefficient
cm = np.corrcoef(data.values.T)

# save result
df = pd.DataFrame(cm)
# df.to_csv("./correlation.csv",index=False)

# plot heatmap
sns.set(font_scale= 1) # set stype of plot
ticks = data.columns.values # get labels
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=ticks, xticklabels=ticks)
plt.show()
plt.savefig('./correlation.jpg')

