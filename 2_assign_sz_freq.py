# This script takes models answers and joins them all together 
# with the ground truth sz control metrics for seizure frequency

# Only code is provided for reproducibility


import sys
import os
import pandas as pd
import numpy as np
import nltk
import re

#%% Import data ###############################################################
# path = path here
sys.path.insert(0, path) # insert path

df = pd.read_csv(os.path.join(path,'dataset_encounters.csv'), sep=',', index_col=0) # dataset not provided (HPI)

df = df[df['patient_has_epilepsy'] == 'YES']
   
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
# Sz FREQ
#-------------------------------------

# INN == Innumerable (i.e. &gt;= 10 per day on most days)
# MULT == Multiple per day (i.e. 4 days per week with &gt;=2 seizures)
# DAIL == Daily (i.e. 4 or more days in the past week)
# WKLY == Weekly but not daily (i.e. 1 - 3 in the past week)
# MNTH == Monthly but not weekly (i.e. 1 - 3 in the past month)
# NEM == At least once per year, but not every month (i.e. 10 or fewer in past 12 months)
# YEAR == Less than once per year

def codes_sz_freq(n, col):

    n = coding(n, col, 'innumerable', 'INN') 
    n = coding(n, col, '50100 (seizures)?\s?per day', 'INN') 
    n = coding(n, col, '2030 (seizures)?(events)?\s?per day', 'INN') 
    n = coding(n, col, 'high (seizure)?\s?frequency', 'INN') 
    n = coding(n, col, '23 per day', 'INN') 
    
    n = coding(n, col, 'multiple', 'MULT') # per day
    n = coding(n, col, 'times a?(per)? day', 'MULT')
    n = coding(n, col, 'twice a day', 'MULT')
    n = coding(n, col, '6 per day', 'MULT')
    n = coding(n, col, '[2-9]a?\s?day', 'MULT')
    n = coding(n, col, '[2-9] nightly', 'MULT')
    n = coding(n, col, '[2-9]\s?(events)?(seizures)? a day', 'MULT')
    n = coding(n, col, 'five seizures day', 'MULT')
    n = coding(n, col, '56 spells per day', 'MULT')
    n = coding(n, col, 'up to 2 every 8 hours', 'MULT')
    n = coding(n, col, 'several in a day', 'MULT')

    n = coding(n, col, 'daily', 'DAIL')
    n = coding(n, col, 'once a day', 'DAIL') 
    n = coding(n, col, 'every day', 'DAIL')
    n = coding(n, col, 'every morning', 'DAIL') 
    n = coding(n, col, 'every evening', 'DAIL')
    n = coding(n, col, 'every night', 'DAIL')
    n = coding(n, col, 'in (one)?1? day', 'DAIL')
    n = coding(n, col, 'spells a day', 'DAIL')
    n = coding(n, col, '1 (seizure)?(per)?a?\s?day', 'DAIL')

    
    n = coding(n, col, 'weekly', 'WKLY')
    n = coding(n, col, 'weekly but not daily', 'WKLY')
    n = coding(n, col, '([\w\d]).{1,15}\s?week', 'WKLY')
    n = coding(n, col, 'every [3-9] days', 'WKLY')
    n = coding(n, col, 'every 810 days', 'WKLY')
    n = coding(n, col, '[1-3] nocturnal seizures?', 'WKLY')
    n = coding(n, col, 'events on tues wed thur fri sat sun', 'WKLY')
    n = coding(n, col, 'every few days', 'WKLY')
    n = coding(n, col, 'one to a few seizures every couple of nights', 'WKLY')
    n = coding(n, col, '79 events in 30 days', 'WKLY')
    
    

    n = coding(n, col, 'at least once per month', 'MNTH') 
    n = coding(n, col, 'monthly but not weekly', 'MNTH')
    n = coding(n, col, '([\w\d]).{1,15}\s?month', 'MNTH')
    n = coding(n, col, 'month', 'MNTH')
    n = coding(n, col, '13 times each year', 'MNTH')
    n = coding(n, col, '12xyear', 'MNTH')
    n = coding(n, col, '34 per year', 'MNTH')
    n = coding(n, col, 'currently 1 q 5 wks', 'MNTH')
    n = coding(n, col, 'freq 1 q 2wk', 'MNTH')
    
    
    
    n = coding(n, col, 'at least once per y', 'NEM') #NEM == At least once per year, but not every month
    n = coding(n, col, 'per year', 'NEM')
    n = coding(n, col, 'a year', 'NEM')
    n = coding(n, col, 'every year', 'NEM')
    n = coding(n, col, '([1-9])?(seizures)?(events)?\s?in the past year', 'NEM')
    n = coding(n, col, '\d\/\s?year', 'NEM')
    n = coding(n, col, '<\s?1/\s?/?y', 'NEM')
    n = coding(n, col, 'yearly', 'NEM')
    n = coding(n, col, 'in (the)?\s?last year', 'NEM')
    n = coding(n, col, 'one seizure q 60 days', 'NEM')
    n = coding(n, col, 'over the past year', 'NEM')
    n = coding(n, col, 'couple of (times)?\s?a?(per)?\s?year', 'NEM')
    n = coding(n, col, 'less than 1 q 3mos', 'NEM')
    n = coding(n, col, 'a few times year', 'NEM')
    n = coding(n, col, 'one to two events each year', 'NEM')
    n = coding(n, col, '2 seizures over the course of the last year', 'NEM')
    n = coding(n, col, '6 to 8 total seizures in three years', 'NEM')
    n = coding(n, col, 'last year she has had 5 small seizures', 'NEM')
    n = coding(n, col, '2 gtc year', 'NEM')
    


    n = coding(n, col, '[1-9] y(ea)?r', 'YEAR')
    n = coding(n, col, '>\s?2\s?/?y', 'YEAR')
    n = coding(n, col, 'sz\s?free', 'YEAR')
    n = coding(n, col, 'seizure\s?free', 'YEAR')
    n = coding(n, col, 'event\s?free', 'YEAR')
    n = coding(n, col, 'for years', 'YEAR')
    n = coding(n, col, 'no (sz?)?(seizure?)?(s?)? since \d\d\d\d ', 'YEAR')
    n = coding(n, col, 'less than 2 years ago', 'YEAR')
    n = coding(n, col, 'none for 2?\s?years', 'YEAR')
    n = coding(n, col, 'none recently', 'YEAR')
    n = coding(n, col, '2year seizure recurrence', 'YEAR')
    n = coding(n, col, 'none in (many)?(several)?\s?years', 'YEAR')
    n = coding(n, col, 'less than onc?e per y', 'YEAR')
    n = coding(n, col, 'onc?e (or)?(twice)?\s?a?\s?y', 'YEAR')
    n = coding(n, col, 'every (other)?(few)? year', 'YEAR')
    n = coding(n, col, 'in (his)?(her)?\s?life', 'YEAR')
    n = coding(n, col, 'life\s?time', 'YEAR')
    n = coding(n, col, 'three in adult life', 'YEAR')
    n = coding(n, col, 'frequency none', 'YEAR')
    n = coding(n, col, 'since (19)?(20)?', 'YEAR')
    n = coding(n, col, 'frequency is approximately 1.53', 'YEAR')
    n = coding(n, col, 'once in a blue moon', 'YEAR')
    n = coding(n, col, 'at least one seizure each year', 'YEAR')
    n = coding(n, col, 'over his life he has had a total of 3', 'YEAR')
    n = coding(n, col, 'no seizures for many years', 'YEAR')
    n = coding(n, col, 'rarely', 'YEAR')
    n = coding(n, col, 'no (gtc)?(seizures)?\s?in years', 'YEAR')
    
    n = coding(n, col, 'frequency not well defined', 'NDEF')
    n = coding(n, col, 'frequency not wel', 'NDEF') 
    n = coding(n, col, 'frequency variable', 'NDEF')
    n = coding(n, col, 'not available', 'NDEF')
    n = coding(n, col, 'frequency (is)?\s?(quite)?\s?low', 'NDEF')    
    n = coding(n, col, 'unknown', 'UNK')
    n = coding(n, col, 'unsure', 'UNK')
    n = coding(n, col, 'uncertain', 'UNK')
    n = coding(n, col, 'unclear', 'UNK')
    n = coding(n, col, 'n\/a', 'NA_replace')
    n = coding(n, col, 'frequency na', 'NA_replace')
    

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
        
        # Less than 1 month
        n[col][c & (n['m'][c]==0)] = 'WKLY' 
        
        # More than 1 month to 3 months ago
        n[col][c & (n['m'][c]>=1) & (n['m'][c]<=3)] = 'MNTH' 
        
        # More than 3 month to 6 months ago
        n[col][c & (n['m'][c]>=4) & (n['m'][c]<=6)] = 'MNTH' 
        
        # More than 6 month to 12 months ago
        n[col][c & (n['m'][c]>=7) & (n['m'][c]<=12)] = 'NEM' 
        
        # More than 12 months ago
        n[col][c & (n['m'][c]>=13) & (n['m'][c]<=24)] = 'YEAR'
        
        # More than 24 months ago
        n[col][c & (n['m'][c]>24)] = 'YEAR'
        
        n = n.drop(columns=['y','m'])
        
        n[col] = n[col].astype(str)
        
        return n
    
    def extract_date(n,c, flag, col):
        
        n['y'] = ''
        n['m'] = ''
        
        if flag == 'date_yy':
            n['y'][c] = n[col][c].apply(lambda x: '20' + pd.Series(x).str.extract('\d\d?\/\d\d?\/(\d\d)')[0][0]) 
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d\d?)\/\d\d?\/\d\d')[0][0])
              
        elif flag == 'mddyy':
            n['y'][c] = '20' + n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d\d(\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d\d\d\d')[0][0])
        
        elif flag == 'mdyy':
            n['y'][c] = '20' + n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d(\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d\d\d')[0][0])
            
        elif flag == 'mmddyy':
            n['y'][c] = '20' + n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d\d\d(\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d\d)\d\d\d\d')[0][0])
        
        elif flag == 'mdyyyy':
            n['y'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d(\d\d\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d\d\d\d\d')[0][0])
        
        elif flag == 'myyyy':
            n['y'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d?\d(\d\d\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d?\d)\d\d\d\d')[0][0])
        
        elif flag == 'mddyyyy':
            n['y'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d\d(\d\d\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d\d\d\d\d\d')[0][0])
        
        elif flag == 'mmddyyyy':
            n['y'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('\d\d\d\d(\d\d\d\d)')[0][0])
            n['m'][c] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d\d)\d\d\d\d\d\d')[0][0])
         
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
            n['m'] = 0    
            
        if (flag == 'mdd') | (flag == 'mdd_prior_year'):
            n['m'] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d\d')[0][0])
 
        if flag == 'md':
            n['m'] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d)\d')[0][0])
        
        if flag == 'mmdd':
            n['m'] = n[col][c].apply(lambda x: pd.Series(x).str.extract('(\d\d)\d\d')[0][0])
 
        if (flag == 'yyyy_month') or (flag == 'month_yyyy') or (flag == 'month_d_yyyy') or (flag == 'month') or  (flag == 'month_prior_year'):
              n = month(n,c,col) #assign month        
        #       c = c & (n['m'][c].astype(str).str.isdigit()) #removes cases were month was not assigned
              # n = assign(n, c, col)
             
        # if (flag == 'mm_yyyy') or (flag == 'myyyy') or (flag == 'date_yy') or (flag == 'yyyy') or  (flag == 'mmddyyyy') or (flag == 'mdd'): 
        #     #Month <= 12
        
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
    n[col][c & (n['y'][c].astype(float) <= 2)] = 'YEAR'
    
    n = n_y(n,c,col)
    n[col][c & (n['y'][c].astype(float) > 2)] = 'YEAR'

    n = coding(n, col, 'seizure free', 'YEAR')
    n = coding(n, col, 'sz free', 'YEAR')
    n = coding(n, col, 'seizurefree', 'YEAR')
    n = coding(n, col, 'szfree', '2YR')
    n = coding(n, col, 'event free', 'YEAR')

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
    
    # Remove uncoded    
    n = remove_uncoded(n,col)
    
    return n
             


def ndf (n, df):
    
    n = pd.concat([n,df[['PatientID','Date']].reset_index(drop='True')], axis=1)
    
    n = n.rename(columns={'Unstructured':'Notes'})
    
    n.Date = n.Date.astype(str)
    
    n['year'] = n.Date.apply(lambda x:  re.search(r'(\d+)',x).group(1))
    
    n['month'] = n.Date.apply(lambda x:  re.search('\d\d\d\d-(\d\d)',x).group(1))
    
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

    n = codes_sz_freq(n, 'model_answer')
    
    return n


n = pd.read_csv(os.path.join(path,'often_event.csv'), index_col=0)  # from 1_qa_szfreq_scores.py - not provided (HPI)

n = ndf(n, df)

n = n.rename(columns={'score':'often_event_score', 'answer':'often_event_answer', 'model_answer':'often_event_answer_coded'})


#Extract codes for sz freq


n['seizure_freq'] = pd.Series(n['Notes']).apply(lambda x: find_(x, 'seizure frequency',ntokens='50'))

n['seizure_freq'][n['seizure_freq'].astype(str) == ''] = pd.Series(n['Notes'][n['seizure_freq'].astype(str) == '']).apply(lambda x: find_(x, 'sz frequency',ntokens='50'))

n['seizure_freq_coded'] = n['seizure_freq']

n = codes_sz_freq(n, 'seizure_freq_coded')

# n['current_seizure_freq'] = pd.Series(n['Notes']).apply(lambda x: find_(x, 'current seizure frequency',ntokens='50'))

# frequency (only) to try

# Other model qa

r = pd.read_csv(os.path.join(path,'often_seizure.csv'), index_col=0) # from 1_qa_szfreq_scores.py - not provided (HPI)

r = ndf(r, df)

r = r.rename(columns={'score':'often_seizure_score', 'answer':'often_seizure_answer', 'model_answer':'often_seizure_answer_coded'})

l = pd.read_csv(os.path.join(path,'frequency_seizure.csv'), index_col=0) # from 1_qa_szfreq_scores.py - not provided (HPI)

l = ndf(l, df)

l = l.rename(columns={'score':'frequency_seizure_score', 'answer':'frequency_seizure_answer', 'model_answer':'frequency_seizure_answer_coded'})

e = pd.read_csv(os.path.join(path,'frequency_event.csv'), index_col=0) # from 1_qa_szfreq_scores.py - not provided (HPI)

e = ndf(e, df)

e = e.rename(columns={'score':'frequency_event_score', 'answer':'frequency_event_answer', 'model_answer':'frequency_event_answer_coded'})

nr = pd.concat([n,r[['often_seizure_score', 'often_seizure_answer', 'often_seizure_answer_coded']]], axis = 1)

nrl = pd.concat([nr,l[['frequency_seizure_score', 'frequency_seizure_answer', 'frequency_seizure_answer_coded']]], axis = 1)

nrle = pd.concat([nrl,e[['frequency_event_score', 'frequency_event_answer', 'frequency_event_answer_coded']]], axis = 1)

n = nrle

# INN == Innumerable (i.e. &gt;= 10 per day on most days)
# MULT == Multiple per day (i.e. 4 days per week with &gt;=2 seizures)
# DAIL == Daily (i.e. 4 or more days in the past week)
# WKLY == Weekly but not daily (i.e. 1 - 3 in the past week)
# MNTH == Monthly but not weekly (i.e. 1 - 3 in the past month)
# NEM == At least once per year, but not every month (i.e. 10 or fewer in past 12 months)
# YEAR == Less than once per year
# UNK == Unknown
# NDEF == Frequency not well defined 

n = n[['PatientID','Date','Notes','often_event_answer', 'often_seizure_answer',
       'frequency_seizure_answer', 'frequency_event_answer', 'seizure_freq', 
       'often_event_answer_coded','often_seizure_answer_coded',
       'frequency_seizure_answer_coded','frequency_event_answer_coded',
       'seizure_freq_coded']]
  
 
# merge with dataset with sz control metrics

d = pd.read_csv(os.path.join(path,'data_sz_freq.csv'))  # not provided (HPI)

    
n = pd.merge(n,d[['PatientID','Date','sz_freq_a', 'sz_freq_b', 'sz_freq_c', 'sz_freq_d']], on =['PatientID','Date'], how='inner')


# Assign 'nan' for models answers and regexes

cs = ['often_event_answer_coded','often_seizure_answer_coded',
      'frequency_seizure_answer_coded','frequency_event_answer_coded',
      'seizure_freq_coded']

for col in cs:
    n[col][(n[col].astype(str).str.contains('NDEF|UNK|NA'))] = ''

# stats

cols = ['often_event_answer','often_seizure_answer','frequency_seizure_answer',
        'frequency_event_answer','seizure_freq']

for i in cols:
    
    cond = (n[i].astype(str) != '') 

    print(i)
    print(len(n[cond]))

cols = ['often_event_answer_coded','often_seizure_answer_coded',
        'frequency_seizure_answer_coded', 'frequency_event_answer_coded','seizure_freq_coded']


for col in cols:
    n[col][~(n[col].astype(str).str.contains('INN|MULT|DAIL|WKLY|MNTH|NEM|YEAR'))] = ''


# n2 = n

# for i in cols:
#     n2 = n2[(n[i].astype(str) == '')]

for i in cols:
    
    cond = (n[i].astype(str) != '') 

    print(i)
    print(len(n[cond])/len(n)*100)

n.to_csv(os.path.join(path,'sz_freq_combined.csv'), index=False) 