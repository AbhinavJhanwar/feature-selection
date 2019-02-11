'''
Created on Jul 30, 2018

@author: abhinav.jhanwar
'''

''' r: pearson correlation coefficient- varies from -1 to 1. 
when r<0 -  negative linear relationship
when r>0 - positive linear relationship
when r close to 0- weak linear relationship
when r close to +-1 then strong linear relationship'''

import pandas as pd
import numpy as np
import seaborn as sn
import scipy
import matplotlib.pyplot as plt

data = pd.read_csv("gapminder.csv", low_memory=False)

data.internetuserate = pd.to_numeric(data.internetuserate, errors='coerce')
data.urbanrate = pd.to_numeric(data.urbanrate, errors='coerce')
data.incomeperperson = pd.to_numeric(data.incomeperperson, errors='coerce')

data.incomeperperson = data.incomeperperson.replace(' ', np.nan)

scat1 = sn.regplot(x="urbanrate", y="internetuserate", fit_reg=True, data=data)
plt.xlabel("Income per person")
plt.ylabel("Internet Use Rate")
plt.title("Scatterplot for the association between income per person and internet use rate")

data_clean = data.dropna()

print("association between urbanrate and internetuserate")
print(scipy.stats.pearsonr(data_clean['urbanrate'], data_clean['internetuserate']))
# value1 = pearson correlation coefficient = r
# r^2 = gives an idea that explanatory variable alone can predict this much of variability of response variable
# value2 = p value

print("association between incomeperperson and internetuserate")
print(scipy.stats.pearsonr(data_clean['incomeperperson'], data_clean['internetuserate']))
