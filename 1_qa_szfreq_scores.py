
# This script takes the notes and computes RoBERTa_for_seizureFrequency_QA 
# scores for seizure frequency

# Only code is provided for reproducibility

import sys
import os
import pandas as pd
from transformers import pipeline
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("CNT-UPenn/RoBERTa_for_seizureFrequency_QA")

model = AutoModelForQuestionAnswering.from_pretrained("CNT-UPenn/RoBERTa_for_seizureFrequency_QA")

qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)

#%% Import data ###############################################################

# path = path here
sys.path.insert(0, path) # insert path

df = pd.read_csv(os.path.join(path,'Unstructured_Notes.csv'))  # dataset not provided (HPI)

df = df[df['patient_has_epilepsy'] == 'YES']

notes = df['Unstructured'] 

questions = {'often_event':"How often does the patient have events?",
              'often_seizure':'How often does the patient have seizures?',
              'frequency_seizure':'What is the seizure frequency?',
              'frequency_event': 'What is the event frequency?'}

dfs = dict()

for k, v in questions.items():
    
    a = notes.apply(lambda x: qa_model(question = v, context = x))
    
    answers = pd.DataFrame()
    n = 0

    a = a.reset_index().drop(columns='index')
    a = a['Unstructured']
    
    for i in range(len(a)):
        b = pd.DataFrame([a[i]], columns=['score','start','end','answer'])
        answers = pd.concat([answers, b], ignore_index=True)
        n+=1
        print(n)
      
    # notes = notes.reset_index()
    answers = pd.concat([answers, notes.reset_index()], axis=1).drop(columns='index')
    answers.to_csv(f'path/{k}.csv')
    dfs[k] = answers
    print(k)
    