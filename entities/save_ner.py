import numpy as np
import torch
import spacy
import json
from collections import Counter
from collections import defaultdict

dataset='blogauth' # 'blogauth'

def show_ents(docs):
  for doc in docs:
    if doc.ents:
      for ent in doc.ents:
        print(ent.text + " - " + ent.label_)

def count_ner_types(docs):
  labels = []
  for doc in docs:
    if doc.ents:
      for ent in doc.ents:
        labels.append(ent.label_)
  return Counter(labels)

def save_extracted_ner(docs, dataset):
  ner_dict = defaultdict(list)
  for doc in docs:
    if doc.ents:
      for ent in doc.ents:
        # In enron dataset remove org identifers after name (i.e: Kerri Thompson/Corp/Enron@ENRON)
        if dataset=='enron' and ent.label_=='PERSON':
          person = ent.text.split('/',1)[0]
          ner_dict[ent.label_].append(person)
        else:
          ner_dict[ent.label_].append(ent.text)

  with open('entities/ner_data2_'+dataset+'.json', 'w') as fp:
    json.dump(ner_dict, fp)
    
# Run Named Enitity Extraction    
def main():
  nlp = spacy.load("en_core_web_sm")
  train_list = json.load(open( "datasets/"+dataset+"/train_data.json",))
  train_data = list(map(lambda x: (list(x.values())[:1]), train_list))
  
  print("Identifying Named Entities...")
  docs = []
  for sample in train_data:
    s = "".join(sample)
    docs.append(nlp(s))
  
  print("Entity counts by type:\n"+str(count_ner_types(docs)))
  
  print("Saving Entities...")
  save_extracted_ner(docs, dataset)
  
if __name__ == '__main__':
    main()