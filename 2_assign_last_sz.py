# This script takes models answers and joins them all together 
# with the ground truth sz control metrics for last seizure

# Only code is provided for reproducibility

import sys
import os
import pandas as pd
import numpy as np
import nltk
import re
import warnings
warnings.filterwarnings("ignore")

#%% Import data ###############################################################
# path = path here
sys.path.insert(0, path) # insert path

df = pd.read_csv(os.path.join(path,'dataset_encounters.csv'), sep=',', index_col=0) # dataset not provided (HPI)

df = df[df['patient_has_epilepsy'] == 1]
   
#-------------------------------------
# Functions
#-------------------------------------
  
#Count tokens 
def count(words):
    text = nltk.Text(words.split())
    counts = len(text)
    return counts

#Remove uncoded rows   
def remove_uncoded(n,col):
    n['tokens'] = n[col].apply(lambda x: count(x))
    n[col][n['tokens']>1] = ''
    return n

#Find expression
def find_(s, d, ntokens):     
    #s = pd.Series(s).str.extract('('+ d + '(:?\??)(\s?)([\w]).{1,'+ntokens+'})').iloc[:,0].str.cat(sep=', ')
    s = pd.Series(s).str.extract('('+ d + '(:?\??[?)(\s?)([\w]).{1,'+ntokens+'})').iloc[:,0].str.cat(sep=', ')
    s = s.replace(':','')
    s = s.replace('?','')
    s = s.replace('~','')
    return s
   
#Assign codes
def coding(n, col, string, code):
    n[col][n[col].str.contains(string)] = code
    return n
   

#-------------------------------------
# LAST Sz
#-------------------------------------

def codes_last_sz(n, col):
    
    #Capture in [x]
    n[col][n[col].str.contains('\[\s?x\s?\] today')] = 'TOD'
    n[col][n[col].str.contains('\[\s?x\s?\] 1-6 days')] = '1DAY'
    n[col][n[col].str.contains('\[\s?x\s?\] 1-4 weeks')] = '1WK'
    n[col][n[col].str.contains('\[\s?x\s?\] 5-12 weeks')] = '5WK'
    n[col][n[col].str.contains('\[\s?x\s?\] 13-26 weeks')] = '13WK'
    n[col][n[col].str.contains('\[\s?x\s?\] 6-12 m')] = '6MON'
    n[col][n[col].str.contains('\[\s?x\s?\] 13-24 m')] = '1YR'
    n[col][n[col].str.contains('\[\s?x\s?\] more than 2 years')] = '2YR'
    n[col][n[col].str.contains('\[\s?x\s?\] decline')] = 'DEC'
    
    
    n = coding(n, col, '(was)?\s?today', 'TOD')
    n = coding(n, col, 'this morning', 'TOD')
    n = coding(n, col, 'this am', 'TOD')
    n = coding(n, col, 'nightly', 'TOD')
    
    n = coding(n, col, '1-6\s?day', '1DAY')
    n = coding(n, col, '\d{3,4} [a|p]m', '1DAY')
    n = coding(n, col, 'tues wed thur fri sat sun', '1DAY')

    n = coding(n, col, 'daily', '1DAY') 
    n = coding(n, col, '(was)?\s?yesterday', '1DAY') 
    n = coding(n, col, 'most recent (\w+\s)+\w+day', '1DAY')
    n = coding(n, col, '1-6\s?d', '1DAY') 
    n = coding(n, col, '.*?\s?[1-5]{1}-[2-6]{1}\s?d', '1DAY') 
    n = coding(n, col, '.*?\s?(([1-7]{1})|(one)|(two)|(three)|(four)|(five)|(six))\s?d(ay)?(s?)', '1DAY') 
    n = coding(n, col, 'this week', '1DAY') 
    n = coding(n, col, 'few days ago', '1DAY') 
    n = coding(n, col, 'weekly', '1DAY') 
    n = coding(n, col, 'day of the visit', '1DAY') 
    n = coding(n, col, '.*?\s?(a|1|(one))\s?week', '1DAY') 
    n = coding(n, col, '.*?\s?(hour(s)?|night(s)?)', '1DAY') 
    n = coding(n, col, 'last \w+day', '1DAY') 
    n = coding(n, col, 'twice a day', '1DAY') 
    
    n = coding(n, col, 'more than (a|1|(one)) week to (2|3|4|(two)|(three)|(four)) weeks', '1WK')
    n = coding(n, col, '.*?\s?last week', '1WK') 
    n = coding(n, col, 'weekend', '1WK')
    n = coding(n, col, 'this past sun', '1WK')
    n = coding(n, col, 'last week', '1WK')
    n = coding(n, col, 'last was week', '1WK')
    n = coding(n, col, 'past week', '1WK')
    n = coding(n, col, 'couple (of)?\s?weeks', '1WK') 
    n = coding(n, col, 'few weeks', '1WK') 
    n = coding(n, col, '.*?\s?(([1-4]{1})|a|(one)|(two)|(three)|(four))\s?w(ee)?k(s?)', '1WK') 
    n = coding(n, col, '.*?\s?[1-3]{1}-[2-4]{1}\s?w(ee)?k(s?)', '1WK') 
    n = coding(n, col, '.*?\s?(a|1|(one))\s?mo(nth)?', '1WK')
    n = coding(n, col, '.*?\s?([7-9]|[1][0-9]|[2][0-9])\s?d(ay)?s?', '1WK')
    n = coding(n, col, 'last month', '1WK')
    n = coding(n, col, 'past month', '1WK')
    
    n = coding(n, col, '>?\s?[1-4]\s?week', '1WK') 
    n = coding(n, col, 'this month', '1WK')
    n = coding(n, col, 'each month', '1WK')
    n = coding(n, col, 'end of the month', '1WK')
    
    n = coding(n, col, 'more than 1 month to 3 months', '5WK')
    n = coding(n, col, '5 weeks', '5WK')
    n = coding(n, col, '.*?\s?5-12\s?w', '5WK')
    n = coding(n, col, '.*?\s?[4-9]-[5-9]\s?w(ee)?k(s?)', '5WK')
    n = coding(n, col, '.*?\s?[4-9]-[1][0-2]\s?w(ee)?k(s?)', '5WK')
    n = coding(n, col, '.*?\s?(([5-9]{1}|[1][0-2])|(five)|(six)|(seven)|(eight)|(nine)|(ten)|(eleven)|(twelve))\s?w(ee)?k(s?) ago', '5WK')
    n = coding(n, col, '.*?\s?[1-2]-[2-3] mo(nths)?\s?ago', '5WK')
    n = coding(n, col, 'few?\s? mo(nths)?\s?ago', '5WK')
    n = coding(n, col, 'last few?\s? mo(nths)?', '5WK')
    n = coding(n, col, 'couple of months ago', '5WK')
    n = coding(n, col, 'couple (of)?\s?months', '5WK')
    n = coding(n, col, '.*?\s?(([1-3]{1})|(one)|(two)|(three))\s?mo(nths)?', '5WK')
     
    n = coding(n, col, '>\s?[1-3]\s?m', '5WK') 
    
    n = coding(n, col, 'more than 3 mo(nths)? to 6 mo(nths)?', '13WK')
    n = coding(n, col, 'several weeks ago', '13WK') #####################
    n = coding(n, col, '.*?\s?13-26\s?w', '13WK') 
    n = coding(n, col, '.*?\s?[3-5]-[4-6]\s?mo(nths)?', '13WK')
    n = coding(n, col, '.*?\s?(([4-6]{1})|(four)|(five)|(six))\s?mo(nths)?', '13WK')
    
    n = coding(n, col, '>\s?[3-5]\s?m', '13WK') 
    
    n = coding(n, col, 'weeks ago', '1WK') # last episode weeks ago
    
    n = coding(n, col, 'more than 6 months to 12 months', '6MON')
    n = coding(n, col, 'several months ago', '6MON')
    n = coding(n, col, '.*?\s?6-12\s?mon', '6MON') 
    n = coding(n, col, '.*?\s?[7-9]-[8-9]\s?mo(nths)?', '6MON')
    n = coding(n, col, '.*?\s?[7-9]-[1][0-2]\s?mo(nths)?', '6MON')
    n = coding(n, col, '.*?\s?([7-9]|[1][0-2])\s?mo(nths)?', '6MON')
    n = coding(n, col, '.*?\s?((one)|1)\s?y(ea)?r?', '6MON')
    
    n = coding(n, col, '>\s?[6-9]\s?m', '6MON') 
    n = coding(n, col, '>\s?[1][0-2]\s?m', '6MON') 
    
    n = coding(n, col, 'more than (a|1)\s?y(ea)?r', '1YR')
    n = coding(n, col, '.*?\s?13-24\s?mon', '1YR') 
    n = coding(n, col, 'less than ((two)|2|a)\s?y(ea)?r?(s)? ago', '1YR')
    n = coding(n, col, '.*?\s?\>?\s?(1|a|(one))\s?y(ea)?r?', '1YR')
    n = coding(n, col, 'last (1[3-9])|(2[0-3]) months', '1YR')
    
    n = coding(n, col, '>\s?1\s?/?y', '1YR') 
    n = coding(n, col, 'last year', '1YR') 
    n = coding(n, col, 'past year', '1YR') 
    
    n = coding(n, col, 'more than 2 y(ea)?r(s)?', '2YR') 
    n = coding(n, col, '.*?\s?([3-9]|[1][0-9]|[2][0-9]|[3][0-9]|[4][0-9])(\+?)\s?y(ea)?r?(s)?', '2YR')
    n = coding(n, col, '.*?\s?(([3-9]|[1][0-9]|[2][0-9]|[3][0-9]|[4][0-9])-([3-9]|[1][0-9]|[2][0-9]|[3][0-9]|[4][0-9]))(\+?)\s?y(ea)?r?(s)?', '2YR')
    
    n = coding(n, col, '>\s?2\s?/?y', '2YR') 
    n = coding(n, col, 'in years', '2YR')
    n = coding(n, col, 'age \d{1,2}', '2YR')
    
    n = coding(n, col, '(was)?\s?((a few)|(several)|(many))?\s?y(ea)?r?(s)?', '2YR') 

    n = coding(n, col, 'decline to answer', 'DEC')
    n = coding(n, col, 'unknown', 'UNK')
    n = coding(n, col, 'unclear', 'UNK')
    n = coding(n, col, 'n\/a', 'NA_replace')
   
        
    ########################################
    #For cases where we have an actual Date
    #---------------------------------------
    
    def month(n, c, col):
        n['m'][c & (n[col].str.contains('jan'))] = 1
        n['m'][c & (n[col].str.contains('feb'))] = 2
        n['m'][c & (n[col].str.contains('mar'))] = 3
        n['m'][c & (n[col].str.contains('apr'))] = 4
        n['m'][c & (n[col].str.contains('may'))] = 5
        n['m'][c & (n[col].str.contains('jun'))] = 6
        n['m'][c & (n[col].str.contains('jul'))] = 7
        n['m'][c & (n[col].str.contains('aug'))] = 8
        n['m'][c & (n[col].str.contains('sep'))] = 9
        n['m'][c & (n[col].str.contains('oct'))] = 10
        n['m'][c & (n[col].str.contains('nov'))] = 11
        n['m'][c & (n[col].str.contains('dec'))] = 12
        return n
    
   
    #-------------------
    #Assign codes
    #-------------------
    
    def assign(n, c, col):
    
        #Subtract from Date year 
        n['y'][c] = (n['year'][c].astype(int) - n['y'][c].astype(int)) * 12 # months
        
        #Subtract from Date month 
        n['m'][c] = n['month'][c].astype(int) - n['m'][c].astype(int)
        
        n['m'][c] = n['y'][c].astype(int) + n['m'][c].astype(int)
        
        # Check same day
        n['d'][c & (n['d'] == '')] = 0
        n['d'][c] = n['day'][c].astype(int) - n['d'][c].astype(int)
          
        n[col][c & (n['m'][c]==0) & (n['d'][c]==0) ] = '1DAY' 
        
        # Less than 1 month
        n[col][c & (n['m'][c]==0) & (n['d'][c]!=0)] = '1WK' 
        
        # More than 1 month to 3 months ago
        n[col][c & (n['m'][c]>=1) & (n['m'][c]<=3)] = '5WK' 
        
        # More than 3 month to 6 months ago
        n[col][c & (n['m'][c]>=4) & (n['m'][c]<=6)] = '13WK' 
        
        # More than 6 month to 12 months ago
        n[col][c & (n['m'][c]>=7) & (n['m'][c]<=12)] = '6MON' 
        
        # More than 12 months ago
        n[col][c & (n['m'][c]>=13) & (n['m'][c]<=24)] = '1YR'
        
        # More than 24 months ago
        n[col][c & (n['m'][c]>24)] = '2YR'
        
        n = n.drop(columns=['y','m','d'])
        
        n[col] = n[col].astype(str)
        
        return n
    
    def extract_date(n,c, flag, col):
        
        n['y'] = ''
        n['m'] = ''
        n['d'] = ''
        
        if flag == 'date_yy':
            n['y'][c] = n[col][c].apply(lambda x: '20' + pd.Series(x).str.extract('\d\d?\/\d\d?\/(\d\d)')[0][0]) 
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d\d?)\/\d\d?\/\d\d')[0][0])
            n['d'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d?\/(\d\d?)\/\d\d')[0][0])
              
        elif flag == 'mddyy':
            n['y'][c] = '20' + n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d\d(\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d\d\d\d')[0][0])
            n['d'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d(\d\d)\d\d')[0][0])
        
        elif flag == 'mdyy':
            n['y'][c] = '20' + n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d(\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d\d\d')[0][0])
            n['d'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d(\d)\d\d')[0][0])
            
        elif flag == 'mmddyy':
            n['y'][c] = '20' + n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d\d\d(\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d\d)\d\d\d\d')[0][0])
            n['d'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d(\d\d)\d\d')[0][0])
        
        elif flag == 'mdyyyy':
            n['y'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d(\d\d\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d\d\d\d\d')[0][0])
            n['d'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d(\d)\d\d\d\d')[0][0])
        
        elif flag == 'myyyy':
            n['y'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d?\d(\d\d\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d?\d)\d\d\d\d')[0][0])
        
        elif flag == 'mddyyyy':
            n['y'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d\d(\d\d\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d\d\d\d\d\d')[0][0])
            n['d'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d(\d\d)\d\d\d\d')[0][0])
        
        elif flag == 'mmddyyyy':
            n['y'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d\d\d(\d\d\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d\d)\d\d\d\d\d\d')[0][0])
            n['d'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d(\d\d)\d\d\d\d')[0][0])
         
        elif (flag == 'month') or (flag == 'mdd') or (flag == 'mmdd') or (flag == 'md'):
            n['y'][c] = n['year'][c]
        
        elif (flag == 'month_prior_year') | (flag == 'mdd_prior_year'):
            n['year'] = n['year'].astype(int)
            n['y'][c] = n['year'][c] - 1
         
        elif (flag == 'month_yy'):
            n['y'][c] = '20' + n[col][c].apply(lambda x: pd.Series(x).str.extract('\w+ (\d{2})')[0][0])      

        else:
            n['y'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d\d\d\d)')[0][0])
      
        #Year > 1900 and year <= np.max(n.year)
        c = c & (n['y'][c].astype(int) > 1900) & (n['y'][c].astype(int) <= np.max(n.year.astype(int))) 
       
        if flag == 'yyyy_month':
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d\d\d\s?(\w+)')[0][0])
     
        if flag == 'month_d_yyyy':
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\w+)\s\w+\s\d\d\d\d')[0][0])
      
        if (flag == 'mm_yyyy'):
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d?\d)\/')[0][0])  

        if flag == 'month_yyyy':
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\w+)\s?\d\d\d\d')[0][0])
            
        if flag == 'yyyy':    
            n['m'][c] = 0    
            
        if (flag == 'mdd') | (flag == 'mdd_prior_year'):
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d\d')[0][0])
 
        if flag == 'md':
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d')[0][0])
            n['d'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d(\d)')[0][0])
        
        if flag == 'mmdd':
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d\d)\d\d')[0][0])
            n['d'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d(\d\d)')[0][0])
 
        if (flag == 'yyyy_month') or (flag == 'month_yyyy') or (flag == 'month_d_yyyy') or (flag == 'month') or  (flag == 'month_prior_year'):
              n = month(n,c,col) #assign month        
       
        c = c & (n['m'][c].astype(str).str.isdigit()) #removes cases were month was not assigned
        c = c & (n['m'][c].astype(int) <= 12) & (n['m'][c].astype(int) >= 0 )  #Month <= 12 
        n = assign(n, c, col)
        
        return n
  
     
    #------------------------------------------
    # Extract month/day/year 
    #------------------------------------------

    c = n[col].str.contains('\d\d?\/\d\d?\/\d\d')
 
    n = extract_date(n,c,'date_yy',col)
    
    #------------------------------------------
    # Extract month/year 
    #------------------------------------------
 
    c = n[col].str.contains('\d\d?\/\d\d\d\d')
    
    n = extract_date(n,c,'mm_yyyy',col)
            
    #------------------------------------------
    # Extract month day year numerical
    #------------------------------------------
      
    c = n[col].str.contains('\d\d\d\d\d\d\d\d')
    
    n = extract_date(n,c, 'mmddyyyy', col)
    
    c = n[col].str.contains('\d\d\d\d\d\d\d')
    
    n = extract_date(n,c, 'mddyyyy', col)
    
    c = n[col].str.contains('\d\d\d\d\d\d')
    
    n = extract_date(n,c,'mmddyy',col)
    
    c = n[col].str.contains('\d\d\d\d\d\d')
    
    n = extract_date(n,c,'mdyyyy',col)
    
    c = n[col].str.contains('\d\d\d\d\d')
    
    n = extract_date(n,c, 'mddyy', col)

    #------------------------------------------
    # Extract year and month written by extense
    #------------------------------------------

    c = n[col].str.contains( '\d\d\d\d')
   
    n = extract_date(n,c, 'yyyy_month',col)
    
    c = n[col].str.contains( '\d\d\d\d')
       
    n = extract_date(n,c, 'yyyy',col)

    c = n[col].str.contains( '\d\d\d\d')
   
    n = extract_date(n,c, 'mdyy',col)

    #-------------------------------------------
    # Extract date written by extense month year
    #-------------------------------------------
    
    c = n[col].str.contains('\d\d\d\d')
    
    n = extract_date(n,c,'month_yyyy',col)
    
    c = n[col].str.contains('\d\d\d\d')
      
    n = extract_date(n,c,'month_d_yyyy',col)
    
    c = n[col].str.contains('\d\d\d\d')

    n = extract_date(n,c, 'yyyy', col)

    #------------------------------------------
    # Extract from n years ago
    #------------------------------------------

    def n_y(n,c,col):# n years ago
        n['y'] = ''
        n['y'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('.*?(\d\.\d)\s?y(ea)?r?(s)?\s?ago')[1][0])
        return n
    
    c = n[col].str.contains('\d\.\d\s?y(ea)?r?(s)?\s?ago')
    n = n_y(n,c,col)
    n[col][c & (n['y'][c].astype(float) <= 2)] = '1YR'
    
    n = n_y(n,c,col)
    n[col][c & (n['y'][c].astype(float) > 2)] = '2YR'

    n = coding(n, col, 'seizure free', '2YR')
    n = coding(n, col, 'sz free', '2YR')
    n = coding(n, col, 'seizurefree', '2YR')
    n = coding(n, col, 'szfree', '2YR')
    n = coding(n, col, 'event free', '2YR')
    n = coding(n, col, 'lifetime', '2YR')
    n = coding(n, col, 'last seizure was in 4th grade', '2YR')    
   
    #------------------------------------------
    # Extract month/year numerical 
    #------------------------------------------

    c = n[col].str.contains('\d\d?\d\d\d\d')
    
    n = extract_date(n,c,'myyyy',col)
    
    c = n[col].str.contains('\d\d\d\d')
    
    n = extract_date(n,c,'mmdd',col)
    
    c = n[col].str.contains('\d\d\d')
    
    n = extract_date(n,c,'mdd',col)
    
    c = n[col].str.contains('\d\d\d')
    
    n = extract_date(n,c,'mdd_prior_year',col)
    
    c = n[col].str.contains('\d\d')
    
    n = extract_date(n,c,'md',col)
    
    #-------------------------------------------
    # Extract date written by extense month
    #-------------------------------------------
    
    c = n[col].str.contains('jan|feb|march|april|may|june|july|aug|sept|oct|nov|dec')
    
    n = extract_date(n,c, 'month', col)

    c = n[col].str.contains('jan|feb|march|april|may|june|july|aug|sept|oct|nov|dec')

    n = extract_date(n,c, 'month_prior_year', col)
    
    #-------------------------------------------
    # Extract date written by season
    #-------------------------------------------
 
    def seasons(n, c, col):
        
        n['month'] = n['month'].astype(int)
        
        cc = ((n['month']== 1) | (n['month']== 2) | (n['month']== 12))  #winter
            
        # More than 6 month to 12 months ago
        n[col][c & cc & (n[col].str.contains('spring'))] = '6MON'
        
        # More than 3 month to 6 months ago
        n[col][c & cc & (n[col].str.contains('summer'))] = '13WK'
        
        # More than 1 month to 3 months ago
        n[col][c & cc & (n[col].str.contains('fall'))] = '5WK'
        
        # Less than 1 month
        n[col][c & cc & (n[col].str.contains('winter|christmas'))] = '1WK'
            
    
        cc = ((n['month']== 3) | (n['month']== 4) | (n['month']== 5))  #spring
        
        # More than 6 month to 12 months ago
        n[col][c & cc & (n[col].str.contains('summer'))] = '6MON'
        
        # More than 3 month to 6 months ago
        n[col][c & cc & (n[col].str.contains('fall'))] = '13WK'
        
        # More than 1 month to 3 months ago
        n[col][c & cc & (n[col].str.contains('winter|christmas'))] = '5WK'
        
        # Less than 1 month
        n[col][c & cc & (n[col].str.contains('spring'))] = '1WK'
            
            
        cc = ((n['month']== 6) | (n['month']== 7) | (n['month']== 8))  #summer
          
        # More than 6 month to 12 months ago
        n[col][c & cc & (n[col].str.contains('fall'))] = '6MON'
        
        # More than 3 month to 6 months ago
        n[col][c & cc & (n[col].str.contains('winter|christmas'))] = '13WK'
        
        # More than 1 month to 3 months ago
        n[col][c & cc & (n[col].str.contains('spring'))] = '5WK'
        
        # Less than 1 month
        n[col][c & cc & (n[col].str.contains('summer'))] = '1WK'
            
        cc = ((n['month']== 9) | (n['month']== 10) | (n['month']== 11))  #fall
        
        # More than 6 month to 12 months ago
        n[col][c & cc & (n[col].str.contains('winter|christmas'))] = '6MON'
        
        # More than 3 month to 6 months ago
        n[col][c & cc & (n[col].str.contains('spring'))] = '13WK'
        
        # More than 1 month to 3 months ago
        n[col][c & cc & (n[col].str.contains('summer'))] = '5WK'
        
        # Less than 1 month
        n[col][c & cc & (n[col].str.contains('fall'))] = '1WK'
     
        n[col] = n[col].astype(str)
            
        return n
    
    c = ((n[col].str.contains('spring|summer|winter|christmas')) | (n[col].str.contains('in the fall')) | (n[col].str.contains('this fall')) | (n[col].str.contains('seizures last fall'))) 
    
    n = seasons(n, c, col)
    
    # Remove uncoded    
    n = remove_uncoded(n,col)
    
    return n



def ndf (n, df):
    
    n = pd.concat([n,df[['PatientID','Date']].reset_index(drop='True')], axis=1)
    
    n = n.rename(columns={'Unstructured':'Notes'})
    
    n.Date = n.Date.astype(str)
    
    n['year'] = n.Date.apply(lambda x:  re.search(r'(\d+)',x).group(1))
    
    n['month'] = n.Date.apply(lambda x:  re.search('\d\d\d\d-(\d\d)',x).group(1))
    
    n['day'] = n.Date.apply(lambda x:  re.search('\d\d\d\d-\d\d-(\d\d)',x).group(1))

    #----------------------------------------
    # Lowercase and remove spaces from notes
    #----------------------------------------

    n.Notes = n.Notes.str.lower() # converts to lower case 
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) # removes duplicated spaces

    n.answer = n.answer.str.lower() # converts to lower case 
    n.answer = n.answer.apply(lambda x: " ".join(x.split())) # removes duplicated spaces

    # Important: All regexes must contain lowercase only

    n['model_answer'] = n['answer'].astype(str).str.replace('.',' ')

    n['model_answer'] = n['model_answer'].astype(str).str.replace(',',' ')

    n = codes_last_sz(n, 'model_answer')
    
    return n


n = pd.read_csv(os.path.join(path,'last_seizure.csv'), index_col=0) # from 1_qa_lastsz_scores.py - not provided (HPI)

n = ndf(n, df)

n = n.rename(columns={'score':'last_seizure_score', 'answer':'last_seizure_answer', 'model_answer':'last_seizure_answer_coded'})


#Extract codes for last sz

n['last_seizure'] =  pd.Series(n['Notes']).apply(lambda x: find_(x,'date of last seizure',ntokens='50'))

n['last_seizure'][n['last_seizure'].astype(str) == ''] =  pd.Series(n['Notes'][n['last_seizure'].astype(str) == '']).apply(lambda x: find_(x,'last seizure',ntokens='50'))

n['last_seizure_coded'] = n['last_seizure']

n = codes_last_sz(n, 'last_seizure_coded')

n['last_sz'] =  pd.Series(n['Notes']).apply(lambda x: find_(x,'last sz',ntokens='50'))

n['last_sz_coded'] = n['last_sz']

n = codes_last_sz(n, 'last_sz_coded')

n['last_convulsion'] =  pd.Series(n['Notes']).apply(lambda x: find_(x,'last convulsion',ntokens='50'))

n['last_convulsion_coded'] = n['last_convulsion']

n = codes_last_sz(n, 'last_convulsion_coded')

n['last_event'] =  pd.Series(n['Notes']).apply(lambda x: find_(x,'last event',ntokens='50'))

n['last_event_coded'] = n['last_event']

n = codes_last_sz(n, 'last_event_coded')

# Other model qa

r = pd.read_csv(os.path.join(path,'recent_seizure.csv'), index_col=0)  # from 1_qa_lastsz_scores.py - not provided (HPI)

r = ndf(r, df)

r = r.rename(columns={'score':'recent_seizure_score', 'answer':'recent_seizure_answer', 'model_answer':'recent_seizure_answer_coded'})

l = pd.read_csv(os.path.join(path,'last_event.csv'), index_col=0)  # from 1_qa_lastsz_scores.py - not provided (HPI)

l = ndf(l, df)

l = l.rename(columns={'score':'last_event_score', 'answer':'last_event_answer', 'model_answer':'last_event_answer_coded'})

e = pd.read_csv(os.path.join(path,'recent_event.csv'), index_col=0)  # from 1_qa_lastsz_scores.py - not provided (HPI)

e = ndf(e, df)

e = e.rename(columns={'score':'recent_event_score', 'answer':'recent_event_answer', 'model_answer':'recent_event_answer_coded'})

nr = pd.concat([n,r[['recent_seizure_score', 'recent_seizure_answer', 'recent_seizure_answer_coded']]], axis = 1)

nrl = pd.concat([nr,l[['last_event_score', 'last_event_answer', 'last_event_answer_coded']]], axis = 1)

nrle = pd.concat([nrl,e[['recent_event_score', 'recent_event_answer', 'recent_event_answer_coded']]], axis = 1)

n = nrle

#TOD == Today
#1DAY == More than 1 day to 6 days ago
#1WK == More than 1 week to 4 weeks ago
#5WK == More than 1 month to 3 months ago
#13WK == More than 3 months to 6 months ago
#6MON == More than 6 months to 12 months ago
#1YR == More than 1 year to 2 years ago
#2YR == More than 2 years ago
#DEC == Decline to answer
#UNK == Unknown

n = n[['PatientID','Date','Notes','last_seizure_answer', 'recent_seizure_answer',
       'last_event_answer', 'recent_event_answer', 'last_seizure', 'last_sz', 
       'last_convulsion', 'last_event', 'last_seizure_answer_coded', 
       'recent_seizure_answer_coded', 'last_event_answer_coded', 
       'recent_event_answer_coded', 'last_seizure_coded', 'last_sz_coded', 
       'last_convulsion_coded',  'last_event_coded']]
 
# merge with dataset with sz control metrics

d = pd.read_csv(os.path.join(path,'data_sz_last.csv')) # not provided (HPI)

n = pd.merge(n,d[['PatientID','Date','last_sz_a', 'last_sz_b', 'last_sz_c', 'last_sz_d']], on =['PatientID','Date'], how='inner')

# Stats

cols = ['last_seizure_answer', 'recent_seizure_answer', 
    'last_event_answer','recent_event_answer','last_seizure', 
    'last_sz', 'last_convulsion', 'last_event']

for i in cols:
    
    cond = (n[i].astype(str) != 'nan') 

    print(i)
    print(len(n[cond]))


cols = ['last_seizure_answer_coded', 'recent_seizure_answer_coded', 
    'last_event_answer_coded','recent_event_answer_coded','last_seizure_coded', 
    'last_sz_coded', 'last_convulsion_coded', 'last_event_coded']


for col in cols:
    n[col][~(n[col].astype(str).str.contains('TOD|DAY|WK|MON|YR'))] = ''

for i in cols:
    cond = (n[i].astype(str) != '') 
    print(i)
    print(len(n[cond]))
    print(len(n[cond])/len(n)*100)
    

n.to_csv(os.path.join(path,'last_sz_combined.csv'), index=False)
