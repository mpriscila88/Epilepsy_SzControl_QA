# This script calculates performance for prediction of last seizure

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

labels = ['TOD','1DAY','1WK','5WK','13WK','6MON','1YR','2YR']  

gt_cols = ['last_sz_a', 'last_sz_b', 'last_sz_c', 'last_sz_d']

for i in labels:
    df[i+'_gt'] = np.where(df[gt_cols].eq(i).any(axis = 1 ), 1, 0)

for i in labels:  
    df[i+'_m'] = np.where(df[m_cols].eq(i).any(axis = 1 ), 1, 0)             
  
#################################################################
# compute for all models as predictions
#################################################################
   
gt = np.array(df.iloc[:,df.columns.astype(str).str.contains('_gt')])
y_pred = np.array(df.iloc[:,df.columns.astype(str).str.contains('_m')])

boot_all_micro, boot_all_macro, boot_label = perf(gt, y_pred, labels)



# sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib

import seaborn as sns
sns.set(font_scale=3.5)
title_font = {'size':'50'}

f, axes = plt.subplots(2, 4, figsize=(30, 20))
axes = axes.ravel()
matplotlib.rc('axes',edgecolor='black')
for i in range(8):
    disp = ConfusionMatrixDisplay(confusion_matrix(gt[:, i],
                                                   y_pred[:, i]),
                                  display_labels=['N','Y'])
    disp.plot(ax=axes[i], cmap=plt.cm.Blues, values_format='.4g')
    disp.ax_.set_title(labels[i],fontdict=title_font)
    if (i>0) & (i!=4):
        disp.ax_.set_ylabel('')
    # disp.ax_.tick_params(axis='both', which='major', labelsize=40)     
    disp.im_.colorbar.remove()
    disp.ax_.grid(False)
    matplotlib.rc('axes',edgecolor='black')
    disp.ax_.tick_params(axis='both', which='major', labelsize=50)
matplotlib.rc('axes',edgecolor='black')
plt.subplots_adjust(wspace=0.2, hspace=0.1)
matplotlib.rc('axes',edgecolor='black')
f.colorbar(disp.im_, ax=axes)

plt.show()
