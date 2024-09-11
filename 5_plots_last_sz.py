# This script plots the ground truth against predictions for last seizure

# Both code and data are provided for reproducibility

import sys
import os
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

#%% Import data ###############################################################

# path = path here
sys.path.insert(0, path) # insert path

df = pd.read_csv(os.path.join(path,'last_sz_combined.csv')) # PHI removed -- each row corresponds to a patient visit

# Assign number to code from proximate to date (0) to further apart

l=['last_seizure_answer_coded', 'recent_seizure_answer_coded', 
   'last_event_answer_coded','recent_event_answer_coded','last_seizure_coded', 
   'last_sz_coded', 'last_convulsion_coded', 'last_event_coded',
   'last_sz_a', 'last_sz_b', 'last_sz_c', 'last_sz_d']

for i in l:
    df[i][df[i].astype(str) == 'TOD'] = 0
    df[i][df[i].astype(str) == '1DAY'] = 1/7
    df[i][df[i].astype(str) == '1WK'] = 1
    df[i][df[i].astype(str) == '5WK'] = 5
    df[i][df[i].astype(str) == '13WK'] = 13
    df[i][df[i].astype(str) == '6MON'] = 24
    df[i][df[i].astype(str) == '1YR'] = 52
    df[i][df[i].astype(str) == '2YR'] = 104

###############################################################################
# Possible scenarios
###############################################################################

# model - gt 
# one - one 
# several - several 
# one - several 
# several - one 

m=['last_seizure_answer_coded', 'recent_seizure_answer_coded', 
   'last_event_answer_coded','recent_event_answer_coded','last_seizure_coded', 
   'last_sz_coded', 'last_convulsion_coded', 'last_event_coded']

g = ['last_sz_a', 'last_sz_b', 'last_sz_c', 'last_sz_d']
    

# df['unique_m']=df[m].T.agg([pd.unique]).T
# df['unique_gt']=df[g].T.agg([pd.unique]).T

df['unique_ms'] = df[m].apply(lambda x: len(x.dropna().unique()), axis=1)
df['unique_gs'] = df[g].apply(lambda x: len(x.dropna().unique()), axis=1)

# for no model answer we assign the closest error
df['last_seizure_answer_coded'][df['unique_ms'] == 0] = np.nanmin(df[g][df['unique_ms'] == 0], axis = 1)

for i in g:
    df[i][df['unique_ms'] == 0] = np.nan

df['last_sz_a'][df['unique_ms'] == 0] = df['last_seizure_answer_coded'][df['unique_ms'] == 0] 

df['last_seizure_answer_coded'][df['unique_ms'] == 0] = 0

df['unique_ms'][df['unique_ms'] == 0] = 1

for i in g:
    for c in m:    
        df['diff_'+c+i] = np.abs(df[c] - df[i]) # scores as absolute error


diffs=df[df.filter(like='diff_').columns]

# number of unique models outputs varies from 1 to 4

df['min1'] = np.nanmin(diffs.astype(float), axis =1)

diffs['min1'] = np.nanmin(diffs.astype(float), axis =1)

diffs = diffs.astype(float)

diffs = diffs.apply(lambda x: x.replace(np.nanmin(x),np.nan), axis=1)

df['min2'] = np.nanmin(diffs.astype(float), axis =1)

diffs['min2'] = np.nanmin(diffs.astype(float), axis =1)

diffs = diffs.apply(lambda x: x.replace(np.nanmin(x),np.nan), axis=1)

df['min3'] = np.nanmin(diffs.astype(float), axis =1)

diffs['min3'] = np.nanmin(diffs.astype(float), axis =1)

diffs = diffs.apply(lambda x: x.replace(np.nanmin(x),np.nan), axis=1)

df['min4'] = np.nanmin(diffs.astype(float), axis =1)

diffs['min4'] = np.nanmin(diffs.astype(float), axis =1)

diffs = diffs.apply(lambda x: x.replace(np.nanmin(x),np.nan), axis=1)

# model error to closest gt
df['min2'][df['unique_ms'] < 2] = np.nan
df['min3'][df['unique_ms'] < 3] = np.nan
df['min4'][df['unique_ms'] < 4] = np.nan


# a = df.loc[df.index.repeat(df['unique_ms'])]

# create y_true, y_pred columns

for col in list(df[df.filter(like='diff_').columns]):
    for i in list(['min1','min2','min3','min4']):
        df[col][df[col] == df[i]] = i
   
for ii in range(1,5):
    
    df['y_pred'+str(ii)] = np.nan
    df['gt'+str(ii)] = np.nan
    
    for col in list(df[df.filter(like='diff_').columns]):

        c = re.match(r'diff_(.*)last_sz_',col).group(1)
        i = re.match(r'diff_.*(last_sz_.*)',col).group(1) 

        df['y_pred'+str(ii)][df[col] == 'min'+str(ii)] = df[c]
        df['gt'+str(ii)][df[col] == 'min'+str(ii)] = df[i]

# a = df.loc[df.index.repeat(df['unique_ms'])] #3 have repeated differences  

a = pd.DataFrame(np.vstack([df[['y_pred1','gt1']], 
                                    df[['y_pred2','gt2']],
                                    df[['y_pred3','gt3']],
                                    df[['y_pred4','gt4']]
                                    ]), columns=['y_pred','gt']).dropna()#.reset_index(drop=True)


    
a['ae'] = np.abs(a['gt']-a['y_pred'])
   
  
data= a[['gt','y_pred']] 
gt = pd.Series(a['gt'])
y_pred = pd.Series(a['y_pred'])




plt.figure(figsize=(6,6))
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 0.5

y = pd.DataFrame(y_pred)
y=y.rename(columns={0:'y_pred'})
g=pd.DataFrame(gt)
n = pd.concat([pd.DataFrame(gt),y], axis=1)

plt.figure(figsize=(8,8))
ax = sns.boxplot(data=n, x="gt", y="y_pred", linewidth=.75, color = 'steelblue', boxprops=dict(alpha=.6), medianprops = dict(color="steelblue",linewidth=2))
plt.rcParams["font.family"] = "Arial"
plt.xlabel('True label (weeks)', fontsize=22)
plt.ylabel('Assigned label (weeks)', fontsize=22)
plt.rcParams["font.family"] = "Arial"
plt.yticks(ticks = [0,1/7,1,5,13,24,52,104],labels=['TOD','1DAY','1WK','5WK','13WK','6MON','1YR','2YR'])
plt.rcParams.update({'font.size': 20})
ax.set_xticklabels(['TOD','1DAY','1WK','5WK ','13WK ',' 6MON',' 1YR','2YR'])



from sklearn.utils import resample

x = a['ae'] # it can be absolute error or mse

# configure bootstrap
n_iterations = 1000 # here k=number of bootstrapped samples
n_size = int(len(x))
  
# run bootstrap
medians = list()
for i in range(n_iterations):
   scores = resample(x, n_samples=n_size)
   m = np.median(scores)
   medians.append(m)
  
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower =  np.percentile(medians, p)
p = (alpha+((1.0-alpha)/2.0)) * 100
upper =  np.percentile(medians, p)
  
# print(f"\n{alpha*100} confidence interval {lower} and {upper}")
# 95.0 confidence interval 4.0 and 4.857142857142857

# np.median(x)
# 4.857142857142857


plt.figure(figsize=(12,8))
ax=x.hist(align='left',alpha=0.7,rwidth=4)
ax.set_xlabel('Absolute error',fontsize=30)
ax.set_ylabel('Frequency',fontsize=30) 
ax.set_ylim(0, 6000)
ax.tick_params(axis='both', which='major', labelsize=30)

