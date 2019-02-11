'''
Created on Jul 23, 2018

@author: abhinav.jhanwar
'''


''' chi square test 
chi-square value threshold - for a 2 by 2 case: 3.84 is considered large
so this square should be higher for null hypothesis to be rejected
p value threshold - 0.05 '''

import pandas as pd
import numpy as np
import scipy.stats
import seaborn
import matplotlib.pyplot as plt

data = pd.read_csv("nesarc_pds.csv", low_memory=False)

# convert string data to numeric
data['S3AQ3B1'] = pd.to_numeric(data['S3AQ3B1'], errors='coerce')
data['S3AQ3C1'] = pd.to_numeric(data['S3AQ3C1'], errors='coerce')
data['CHECK321'] = pd.to_numeric(data['CHECK321'], errors='coerce')

# extract data for young people who smoke
sub1 = data[(data['AGE']>=18) & (data['AGE']<=25) & (data['CHECK321']==1)]

#make a copy of my new subsetted data
sub2 = sub1.copy()

# recode missing values to python missing (NaN)
sub2['S3AQ3B1']=sub2['S3AQ3B1'].replace(9, np.nan)
sub2['S3AQ3C1']=sub2['S3AQ3C1'].replace(99, np.nan)

#recoding values for S3AQ3B1 into a new variable, USFREQMO
recode1 = {1: 30, 2: 22, 3: 14, 4: 6, 5: 2.5, 6: 1}
sub2['USFREQMO']= sub2['S3AQ3B1'].map(recode1)

# contingency table of observed counts
ct1=pd.crosstab(sub2['TAB12MDX'], sub2['USFREQMO'])
print (ct1)

# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

# chi-square
print ('chi-square value, p value, expected counts')
cs1 = scipy.stats.chi2_contingency(ct1)
print(cs1)

# set variable types 
sub2["USFREQMO"] = sub2["USFREQMO"].astype('category')
# new code for setting variables to numeric:
sub2['TAB12MDX'] = pd.to_numeric(sub2['TAB12MDX'], errors='coerce')


# graph percent with nicotine dependence within each smoking frequency group 
seaborn.factorplot(x="USFREQMO", y="TAB12MDX", data=sub2, kind="bar", ci=None)
plt.xlabel('Days smoked per month')
plt.ylabel('Proportion Nicotine Dependent')

# performing post hoc test when there are more than 2 categories in a feature

''' Bonferroni Adjustment
p/c
c = number of comparisons'''




recode2 = {1: 1, 2.5: 2.5}
sub2['COMP1v2']= sub2['USFREQMO'].map(recode2)
# contingency table of observed counts
ct2=pd.crosstab(sub2['TAB12MDX'], sub2['COMP1v2'])
print (ct2)

# column percentages
colsum=ct2.sum(axis=0)
colpct=ct2/colsum
print(colpct)

# chi-square
print ('chi-square value, p value, expected counts')
cs2 = scipy.stats.chi2_contingency(ct2)
print(cs2)




recode3 = {1: 1, 6: 6}
sub2['COMP1v6']= sub2['USFREQMO'].map(recode3)

# contingency table of observed counts
ct3=pd.crosstab(sub2['TAB12MDX'], sub2['COMP1v6'])
print (ct3)

# column percentages
colsum=ct3.sum(axis=0)
colpct=ct3/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs3= scipy.stats.chi2_contingency(ct3)
print (cs3)




recode4 = {1: 1, 14: 14}
sub2['COMP1v14']= sub2['USFREQMO'].map(recode4)

# contingency table of observed counts
ct4=pd.crosstab(sub2['TAB12MDX'], sub2['COMP1v14'])
print (ct4)

# column percentages
colsum=ct4.sum(axis=0)
colpct=ct4/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs4= scipy.stats.chi2_contingency(ct4)
print (cs4)




recode5 = {1: 1, 22: 22}
sub2['COMP1v22']= sub2['USFREQMO'].map(recode5)

# contingency table of observed counts
ct5=pd.crosstab(sub2['TAB12MDX'], sub2['COMP1v22'])
print (ct5)

# column percentages
colsum=ct5.sum(axis=0)
colpct=ct5/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs5= scipy.stats.chi2_contingency(ct5)
print (cs5)




recode6 = {1: 1, 30: 30}
sub2['COMP1v30']= sub2['USFREQMO'].map(recode6)

# contingency table of observed counts
ct6=pd.crosstab(sub2['TAB12MDX'], sub2['COMP1v30'])
print (ct6)

# column percentages
colsum=ct6.sum(axis=0)
colpct=ct6/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs6= scipy.stats.chi2_contingency(ct6)
print (cs6)




recode7 = {2.5: 2.5, 6: 6}
sub2['COMP2v6']= sub2['USFREQMO'].map(recode7)

# contingency table of observed counts
ct7=pd.crosstab(sub2['TAB12MDX'], sub2['COMP2v6'])
print (ct7)

# column percentages
colsum=ct7.sum(axis=0)
colpct=ct7/colsum
print(colpct)

print ('chi-square value, p value, expected counts')
cs7=scipy.stats.chi2_contingency(ct7)
print (cs7)
