import json
from collections import defaultdict
from collections import Counter
import numpy as np
from ahocorapy.keywordtree import KeywordTree
from timeit import default_timer as timer

################################################
# Experiment Setup
################################################

entity_file = "ner_data_blogauth_wo_pretrained.json"
text = "none_blogauth_naive.txt"
k_value = 1 # parameter for k-eidetic search, None if looking for all entities

################################################
# Create list of given entity type
################################################
def process_entity_type(entities, type):
   ent_list = []
   for k,v in entities.items():
      if k == type:
         for s in v:
            entity = s.strip()
            if len(entity)>3:
               ent_list.append(entity)
   return ent_list

################################################
# Find entities in text samples
################################################
def find_entities(entities, text, type, k_value=None):
   kwtree = KeywordTree(case_insensitive=False)
   val, cnt = np.unique(entities, return_counts=True)

   if k_value is not None:
      eidetic_ents = val[cnt==1]
      print("Number of ",k_value," eidetic",type," ents: ",eidetic_ents.size)
      for ent in eidetic_ents:
         kwtree.add(ent)
   else:
      unique_ents = set(val)
      print("Number of unique" ,type," ents: ",len(list(unique_ents)))
      for ent in unique_ents:
         kwtree.add(ent)
 
   kwtree.finalize()
   with open ("samples/"+text) as f:
      lines = f.readlines()
      results = kwtree.search_all(' '.join(lines))
      #print(len(list(results)))
      result_set = set()
      for result in results:
         if result[0] not in result_set:
            result_set.add(result[0])
   return result_set
      
#######################################################
# Run Experiment
#######################################################
def main():
   start = timer()
   
   all_count = 0

   all_ent = json.load(open("entities/"+entity_file))
   
   search_types = ["PERSON","ORG","LOC","GPE","FAC","MONEY","CARDINAL"]

   for type in search_types:
      select_ents = process_entity_type(all_ent,type)
      found_ents = find_entities(select_ents,text,type,k_value)
      print(type, " count: ",len(found_ents))
      all_count += len(found_ents)
      end = timer()
      #print("minute spent after ",type, ": ",(end-start)/60)
   print("Total number of entities found: ",all_count)

if __name__ == '__main__':
    main()