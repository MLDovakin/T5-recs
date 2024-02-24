
import pandas as pd
import numpy as np
import random
import gzip
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def split_list(row):
    items = [i for i in row['title'] if type(i) != float]
    n = len(items) - 1
    unsold = list(set(random.sample(set_targets, 7)) ^ set(items) ) + [items[n]]
    return pd.Series({
        'Col1': items[:n],
        'outputs': items[n],
        'recs_d': unsold
    })

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('/content/Musical_Instruments.json.gz')
meta_df = getDF('/content/meta_Musical_Instruments.json.gz').loc[:, ['title','asin']]

df = df.merge(meta_df, on='asin', how='left')
set_targets = set(meta_df['title'].values)

def check_length(element):
    return len(element) > 4

set_items = set(meta_df['title'])
del meta_df
new_df = df.apply(split_list, axis=1)

def modify_unpurchased_items_list_english(Unpurchased_items_list):
  concatenated_list = ''.join(Unpurchased_items_list) #Join into a string with commas in between
  concatenated_list = concatenated_list.rstrip(',') #Drop last comma
  concatenated_list = "Candidates: {" + concatenated_list + "}"
  return concatenated_list


def modify_purchased_items_english(purchase_history):
  purchase_history = "Purchases: {" + ''.join(purchase_history) + "}"
  return purchase_history

new_df['Prompt'] = new_df['Col1'].apply(modify_purchased_items_english) + " - " + new_df['recs_d'].apply(modify_unpurchased_items_list_english) +  " - RECOMMENDATION: "
new_df.to_csv('test_df.csv',encoding='utf-8')
