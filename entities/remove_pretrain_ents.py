import datasets
import json
import numpy as np
from timeit import default_timer as timer
from ahocorapy.keywordtree import KeywordTree


entity_file = "ner_data_blogauth.json"


def remove_ents(dataset, ents, kwtree):
   lines = dataset['train']['text']
   results = kwtree.search_all(' '.join(lines))
   
   result_vals = set()
   for result in results:
      if result[0] not in result_vals:
         result_vals.add(result[0])
   #print(result_vals)

   for v in ents.values():
      for x in v:
         if x in result_vals:
            v.remove(x)
   return ents 

def select_entities(entities, type):
   ent_list = []
   for k,v in entities.items():
      if k == type:
         for s in v:
            entity = s.strip()
            ent_list.append(entity)
   return ent_list

   
def main():
   bookcorpus = datasets.load_dataset('bookcorpus')
   wikipedia = datasets.load_dataset("wikipedia", "20220301.en")

   all_ent = json.load(open("entities/"+entity_file))
   search_types = ["PERSON","ORG","LOC","GPE","FAC","MONEY","CARDINAL"]
   kwtree = KeywordTree(case_insensitive=True)

   for type in search_types:
      select_ents = select_entities(all_ent, type)
      unique_ents = set(select_ents)
      for ent in unique_ents:
         kwtree.add(ent)

   kwtree.finalize()
   start = timer()
   all_ent = remove_ents(bookcorpus,all_ent,kwtree)

   for k,v in all_ent.items():
      print(len(v))

   end = timer()
   print("minutes spent after bookcorpus: ",(end-start)/60)

   all_ent = remove_ents(wikipedia,all_ent,kwtree)

   for k,v in all_ent.items():
      print(len(v))
   
   end = timer()
   print("minutes spent after wikipedia: ",(end-start)/60)

   with open('entities/ner_data_blogauth_wo_pretrained.json', 'w') as fp:
      json.dump(all_ent, fp)

if __name__ == '__main__':
    main()


