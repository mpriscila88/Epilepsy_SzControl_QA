# This script calculates performance for prediction of seizure frequency 
# with merged categories

# Both code and data are provided for reproducibility


import sys
import os
import pandas as pd
import numpy as np


#%% Import data ###############################################################
# path = path here
sys.path.insert(0, path) # insert path

from performance import perf

df = pd.read_csv(os.path.join(path,'sz_freq_combined.csv'))  # PHI removed -- each row corresponds to a patient visit

   
# Assign number to code from proximate to date (0) to further apart

m_cols =  ['often_event_answer_coded','often_seizure_answer_coded',
      'frequency_seizure_answer_coded','frequency_event_answer_coded',
      'seizure_freq_coded']

gt_cols = ['sz_freq_a', 'sz_freq_b', 'sz_freq_c', 'sz_freq_d']

labels = ['INN','MULT','DAIL','WKLY','MNTH','NEM','YEAR'] 


for i in labels:
    df[i+'_gt'] = np.where(df[gt_cols].eq(i).any(axis = 1 ), 1, 0)

for i in labels:  
    df[i+'_m'] = np.where(df[m_cols].eq(i).any(axis = 1 ), 1, 0)    
    

def labl(df, flag):
    
    df['INN|DAIL|MULT_'+flag+'_merged'] = df['INN_'+flag] + df['DAIL_'+flag] + df['MULT_'+flag]
    df['INN|DAIL|MULT_'+flag+'_merged'][df['INN|DAIL|MULT_'+flag+'_merged']>1] = 1
        
    df['WKLY_'+flag+'_merged'] = df['WKLY_'+flag]
    
    df['MNTH_'+flag+'_merged'] = df['MNTH_'+flag]
    
    df['YEAR|NEM_'+flag+'_merged'] = df['YEAR_'+flag] + df['NEM_'+flag]
    df['YEAR|NEM_'+flag+'_merged'][df['YEAR|NEM_'+flag+'_merged']>1] = 1
       
    return df

df = labl(df, 'm')
df = labl(df, 'gt')

labels = ['INN|MULT|DAIL','WKLY','MNTH','YEAR|NEM']   


#################################################################
# compute for all models as predictions
#################################################################
   
gt = np.array(df.iloc[:,df.columns.astype(str).str.contains('_gt_merged')])
y_pred = np.array(df.iloc[:,df.columns.astype(str).str.contains('_m_merged')])

boot_all_micro, boot_all_macro, boot_label = perf(gt, y_pred, labels)


# sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(font_scale=4.5)
title_font = {'size':'60'}

f, axes = plt.subplots(2, 2, figsize=(30, 20))
matplotlib.rc('axes',edgecolor='black')
axes = axes.ravel()
matplotlib.rc('axes',edgecolor='black')
for i in range(4):
    disp = ConfusionMatrixDisplay(confusion_matrix(gt[:, i],
                                                   y_pred[:, i]),
                                  display_labels=['N','Y'])
    disp.plot(ax=axes[i], cmap=plt.cm.Blues, values_format='.4g')
    disp.ax_.set_title(labels[i],fontdict=title_font)
    if (i>0) & (i!=2):
        disp.ax_.set_ylabel('')
    # disp.ax_.tick_params(axis='both', which='major', labelsize=40)     
    disp.im_.colorbar.remove()
    disp.ax_.grid(False)
    matplotlib.rc('axes',edgecolor='black')
    disp.ax_.tick_params(axis='both', which='major', labelsize=60)
matplotlib.rc('axes',edgecolor='black')
plt.subplots_adjust(wspace=0.005, hspace=1)
matplotlib.rc('axes',edgecolor='black')
f.colorbar(disp.im_, ax=axes)

plt.show()
