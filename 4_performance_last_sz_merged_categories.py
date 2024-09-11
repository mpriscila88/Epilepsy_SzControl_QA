# This script calculates performance for prediction of last seizure  
# with merged categories

# Both code and data are provided for reproducibility

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


#%% Import data ###############################################################

# path = path here
sys.path.insert(0, path) # insert path

from performance import perf

df = pd.read_csv(os.path.join(path,'last_sz_combined.csv')) # PHI removed -- each row corresponds to a patient visit


# Assign number to code from proximate to date (0) to further apart

l=['last_seizure_answer_coded', 'recent_seizure_answer_coded', 
   'last_event_answer_coded','recent_event_answer_coded','last_seizure_coded', 
   'last_sz_coded', 'last_convulsion_coded', 'last_event_coded',
   'last_sz_a', 'last_sz_b', 'last_sz_c', 'last_sz_d']

m_cols = ['last_seizure_answer_coded', 'recent_seizure_answer_coded', 
    'last_event_answer_coded','recent_event_answer_coded','last_seizure_coded', 
    'last_sz_coded', 'last_convulsion_coded', 'last_event_coded']

gt_cols = ['last_sz_a', 'last_sz_b', 'last_sz_c', 'last_sz_d']

labels = ['TOD','1DAY','1WK','5WK','13WK','6MON','1YR','2YR'] 

for i in labels:
    df[i+'_gt'] = np.where(df[gt_cols].eq(i).any(axis = 1 ), 1, 0)

for i in labels:  
    df[i+'_m'] = np.where(df[m_cols].eq(i).any(axis = 1 ), 1, 0)             
  

def labl(df, flag):
    
    df['TOD-1DAY_'+flag+'_merged'] = df['TOD_'+flag] + df['1DAY_'+flag]
    df['TOD-1DAY_'+flag+'_merged'][df['TOD-1DAY_'+flag+'_merged']>1] = 1
        
    df['1WK-5WK_'+flag+'_merged'] = df['1WK_'+flag] + df['5WK_'+flag]
    df['1WK-5WK_'+flag+'_merged'][df['1WK-5WK_'+flag+'_merged']>1] = 1
    
    df['13WK-6MON_'+flag+'_merged'] = df['13WK_'+flag] + df['6MON_'+flag]
    df['13WK-6MON_'+flag+'_merged'][df['13WK-6MON_'+flag+'_merged']>1] = 1
    
    df['1YR-2YR_'+flag+'_merged'] = df['1YR_'+flag] + df['2YR_'+flag]
    df['1YR-2YR_'+flag+'_merged'][df['1YR-2YR_'+flag+'_merged']>1] = 1
    
    return df

df = labl(df, 'm')
df = labl(df, 'gt')

labels = ['TOD-1DAY','1WK-5WK','13WK-6MON','1YR-2YR']  


#################################################################
# compute for all models as predictions
#################################################################
   
gt = np.array(df.iloc[:,df.columns.astype(str).str.contains('_gt_merged')])
y_pred = np.array(df.iloc[:,df.columns.astype(str).str.contains('_m_merged')])

boot_all_micro, boot_all_macro, boot_label = perf(gt, y_pred, labels)


# sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib

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