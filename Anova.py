'''
Created on Jul 25, 2018

@author: abhinav.jhanwar
'''

'''
null hypothesis- states that there is no correlation among two variables
alternate hypothesis- states that there is correlation among two variables

p value- determines that if it is 0.10 then we will be incorrectly rejecting null hypothesis 10 in 100 times &
will be correct in accepting alternate hypothesis 90 times in 100
p value<0.05- null hypothesis is to be rejected
p value>0.05- null hypothesis is correct that there is no correlation among the two variables

sample = small subsets of a large dataset
groups = various categories of a single feature/target

F = variation among sample means/variation within groups

variation among groups is large - it means that various groups are overlapping & hence 
variation among sample means can be negligible or null hypothesis is true

variation among groups is small - it means variation among sample means dominates and p will be <= 0.05

overall F score is high and p is low -> null hypothesis is rejected

type1 error: wrongly reject the null hypothesis and accept the alternate hypothesis

'''

''' ANOVA - ANalysis Of VAriance 
when feature is quantitative and target is categorical 
we need to also see if there is any moderator i.e. if any particular feature is affecting the correlation between the
two variables'''

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

data = pd.read_csv("nesarc_pds.csv", low_memory=False)

# convert string data to numeric
data['S3AQ3B1'] = pd.to_numeric(data['S3AQ3B1'], errors="coerce")
data['S3AQ3C1'] = pd.to_numeric(data['S3AQ3C1'], errors="coerce")
data['CHECK321'] = pd.to_numeric(data['CHECK321'], errors="coerce")

# extract data for young people who smoke
sub1 = data[(data['AGE']>=18) & (data['AGE']<=25) & (data['CHECK321']==1)]

# setting missing data
sub1['S3AQ3B1'] = sub1['S3AQ3B1'].replace(9, np.nan)
sub1['S3AQ3C1'] = sub1['S3AQ3C1'].replace(99, np.nan)

# recoding the number of days smoked in the past month
recode1 = {1:30, 2:22, 3:14, 4:5, 5:2.5, 6:1}
sub1['USFREQMO'] = sub1['S3AQ3B1'].map(recode1)

# converting new variable to numeric
sub1['USFREQMO'] = pd.to_numeric(sub1['USFREQMO'], errors="coerce")

# creating secondary variable multiplying the days smoked/month and number of cig/per day
# new variable = number of cigrattes per month
sub1['NUMCIGMO_EST'] = sub1['USFREQMO'] * sub1['S3AQ3C1']
sub1['NUMCIGMO_EST'] = pd.to_numeric(sub1['NUMCIGMO_EST'], errors="coerce")

ct1 = sub1.groupby('NUMCIGMO_EST').size()

# using the ols function for calculating F-statistic and associated p value
# explanatory variable -> major depression in life - MAJORDEPLIFE
# here C() is to indicate that variable is categorical
# ols: ordinary least square
model1 = smf.ols(formula='NUMCIGMO_EST ~ C(MAJORDEPLIFE)', data=sub1)
results1 = model1.fit()
print(results1.summary())

# extract only feature that we are using for correlation
sub2 = sub1[['NUMCIGMO_EST', 'MAJORDEPLIFE']].dropna()

print('means for numcigmo_est by major depression status')
m1 = sub2.groupby('MAJORDEPLIFE').mean()
print(m1)

print('standard deviation for numcigmo_est by major depression status')
sd1 = sub2.groupby('MAJORDEPLIFE').std()
print(sd1)

# comparing the p value to be greater than 0.05 and very less variation in the mean
# of two type of categories we conclude that null hypothesis is true


########## lets do the same for ethinicity column which has more that 2 categories
sub3 = sub1[['NUMCIGMO_EST', 'ETHRACE2A']].dropna()
model2 = smf.ols(formula='NUMCIGMO_EST ~ C(ETHRACE2A)', data=sub3).fit()
print(model2.summary())

print('means for numcigmo_est by ethnicity')
m2 = sub3.groupby('ETHRACE2A').mean()
print(m2)

print('standard deviation for numcigmo_est by ethnicity')
sd2 = sub3.groupby('ETHRACE2A').std()
print(sd2)

# when there are more that 2 categories in a categorical variable then we
# use post hoc test for annova
mc1 = multi.MultiComparison(sub3['NUMCIGMO_EST'], sub3['ETHRACE2A'])
res1 = mc1.tukeyhsd()
print(res1.summary())
