# Import all libraries
import pandas as pd; pd.set_option('display.max_columns',None)
import numpy as np
import re
import gc
import os

## Data Aug. packages
import nlpaug.augmenter.word as naw
from nlpaug.util import Action
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


path_root = "."

##################################################################################
def strip_df(df):
  for col in df.columns:
    if df[col].dtype == 'object':
      df[col] = df[col].apply(lambda x: x.strip())
  return df

def pre_process(text):
  
  # fetch alphabetic characters
  text = re.sub("[^a-zA-Z]", " ", text)

  # convert text to lower case
  text = text.lower()

  # split text into tokens to remove whitespaces
  tokens = text.split()

  return " ".join(tokens)

##################################################################################
def pbt_flow(df, aug_models:'list'=None, text_col='Definition', label_col='TopClass', add_input_data=True ):
  import gc
  import pandas as pd
  df_out = df.copy() # To store generated data + input data
  df_new = pd.DataFrame({text_col:[], label_col:[] }) # To store generated data only
  
  for aug_model in aug_models:
    x= [] # To store the texts
    y= [] # to store the corresponding label(s)
    for _ , row in df_out.iterrows():
      x.append(aug_model.augment(row[text_col]))
      y.append(row[label_col])
    df_out=df_out.append(pd.DataFrame({text_col:x, label_col:y })).reset_index(drop=True)
    df_new=df_new.append(pd.DataFrame({text_col:x, label_col:y })).reset_index(drop=True)
    gc.collect()

  from sklearn.utils import shuffle
  if add_input_data:
    return shuffle(df_out)
  else:
    return shuffle(df_new)

#
def augment_data(df, aug_models:'list'=None, text_col='Definition', label_col='TopClass', add_input_data=True):
  df_new = pd.DataFrame()
  num_labels = df[label_col].nunique()
  for k,label in enumerate(df[label_col].unique(),1):
    df_temp = pbt_flow(df[df[label_col] == label], aug_models, text_col, label_col, add_input_data=False)
    df_new=df_new.append(df_temp)
    print('Label {}/{} processed'.format(k,num_labels ))
    
  from sklearn.utils import shuffle
  if add_input_data:
    df_out = df_new.append(df).reset_index(drop=True)
    return shuffle(df_out)
  else:
    return shuffle(df_new.reset_index(drop=True))


#
def compute_number_of_aug_required(n, num_entries_of_majority_class=10, majority_class_factor=2, margin=0.1):
  import math
  if n<num_entries_of_majority_class*(1-margin):
    total = num_entries_of_majority_class*2**(majority_class_factor-1)
    m = math.log2(total/n) / math.log2(2)
    return round(m)
  elif n <= num_entries_of_majority_class and n >= num_entries_of_majority_class*(1-margin) and majority_class_factor>1:
    return majority_class_factor-1
  else:
    return 0
    
def oversampling_pbt_flow(df, aug_models:'list'=None, text_col='Definition', 
  label_col='TopClass', majority_class_factor=2, add_input_data=True ):
    
  df_out = pd.DataFrame()
 
  num_entries_of_majority_class = df[label_col].value_counts()[0]
  
  df_new = pd.DataFrame()
  num_labels = df[label_col].nunique()
  for k,label in enumerate(df[label_col].unique(),1):
    df_temp = df[df[label_col] == label]
    m = compute_number_of_aug_required(
          df_temp.shape[0], 
          num_entries_of_majority_class=num_entries_of_majority_class, 
          majority_class_factor=majority_class_factor
        )
    try:
      df_temp = pbt_flow(df_temp, aug_models[:m], text_col, label_col, add_input_data=False)
      df_new=df_new.append(df_temp)
    except:
        pass
    
    print('Label {}/{} processed'.format(k,num_labels ))
    
  from sklearn.utils import shuffle
  if add_input_data:
    df_out = df_new.append(df).reset_index(drop=True)
    return shuffle(df_out)
  else:
    return shuffle(df_new.reset_index(drop=True))

def initiate_aug_models(path_root=path_root):

  ## Augment text by contextual word embeddings (BERT, DistilBERT, RoBERTA )
  augWordEmb_bert = naw.ContextualWordEmbsAug(
      model_path='bert-base-uncased', 
      action="substitute",
    )

  ## Synonym Augmentation
  augSyn_ppdb = naw.SynonymAug(
    aug_src='ppdb', 
    model_path=os.path.join(path_root, "models" ,'ppdb-2-002.0-s-all'),
  ) 
  augSyn_wordnet = naw.SynonymAug(
    aug_src='wordnet', 
  ) 

  ## Random word Swap augmentation technique
  augRandWord = naw.RandomWordAug(action="swap")

  ## back Translation agu. tecnique: English to Dutch
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  aug_backTranslation= naw.BackTranslationAug(
      from_model_name='facebook/wmt19-en-de', 
      to_model_name='facebook/wmt19-de-en'
  )
  return [ aug_backTranslation,  augRandWord, augSyn_ppdb , augSyn_wordnet, augWordEmb_bert ]

def augment_data_and_save_to_file(x_train, y_train, text_col='Definition', label_col='TopClass',
filename='augmented train data.csv', path_root=path_root, use_oversampling=False, majority_class_factor=2):
  
  aug_models = initiate_aug_models()

  ## Augment train data only
  df = pd.DataFrame({text_col:x_train,label_col:y_train}).reset_index(drop=True)
  if  use_oversampling:
      df = oversampling_pbt_flow(df, aug_models, text_col=text_col, label_col=label_col, majority_class_factor=majority_class_factor)
  else:
      df = augment_data(df, aug_models, text_col=text_col, label_col=label_col)
    
  
  print("\nCheck for duplicated entries", "\nBefore", df.shape)
  df = df.drop_duplicates().reset_index(drop=True)
  print("After", df.shape)
  
  df.to_csv(os.path.join(path_root,filename), index=False)
  
  del df; gc.collect()
