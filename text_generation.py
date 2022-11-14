import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertForSequenceClassification
from torch import cuda
from collections import OrderedDict
import os

import util

os.environ["CUDA_VISIBLE_DEVICES"]="2"

########################################################################
# Experiment Setup
########################################################################

fine_tuning = "partial" # full, partial, dp, none
dataset = "blogauth" # enron
prompt_type = "naive" # naive, informed
n_samples = 20000

########################################################################
# Prepare Fine-tuned model for text generation
########################################################################
def prepare_fine_tuned_model(pretrained_model, n_labels, model_version):
    
    config = BertConfig.from_pretrained(
           model_version,
           num_labels=n_labels)
           
    model = BertForSequenceClassification.from_pretrained(
       model_version,
       config=config)
    
    model_mlm = BertForMaskedLM.from_pretrained(model_version)
    
    # Load fine-tuned model (have to overwrite saved keys due to saved models having _module. prefix)
    state_dict = torch.load(pretrained_model)
    finetuned_dict = OrderedDict([(key.split("_module.")[-1], state_dict[key]) for key in state_dict])
    model.load_state_dict(finetuned_dict)
    
    
    # Create BERT MLM state dicts
    model_mlm_dict = model_mlm.state_dict()

    # Load fine-tuned encoder weights into BERT with MLM head

    # 1. filter out unnecessary keys (classifier head)
    finetuned_dict = {k: v for k, v in finetuned_dict.items() if k in model_mlm_dict}
    # 2. overwrite entries in the existing state dict
    model_mlm_dict.update(finetuned_dict) 
    # 3. load the new state dict
    model_mlm.load_state_dict(model_mlm_dict)

    return model_mlm

####################################################################
# Parse CommonCrawl data for naive prompting
####################################################################
def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file) as f:
        lines = f.readlines() 
    
    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    
    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs)-1):
        start = start_idxs[i]
        end = start_idxs[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start+7]:
            count_eng += 1
            for j in range(start+10, end):
                all_eng += lines[j]
    
    all_eng = all_eng.replace('\n',' ')
    all_eng = all_eng.replace('\t',' ')
    return all_eng

####################################################################
# Parse test data for informed prompting
####################################################################
def parse_test_json(dataset):

    train_data,test_data,train_labels,test_labels = util.load_dataset(dataset)
    
    all_text = ""
    
    for i in range(len(test_data)):
        all_text +=test_data[i]
        
    return str(all_text)


#####################################################################
# Generate Text
#####################################################################
def generate_text(model, seed_ids):
    
    output = model.generate(
       seed_ids, 
       max_length=256, 
       do_sample=True,
       beam_size=30,
       #top_k = 100,
       top_p=0.8,
       num_return_sequences=1,
       temperature=2.0,
       no_repeat_ngram_size=3, 
       early_stopping=False
    )
    
    return output    

#####################################################################
# Main
#####################################################################
def main():
    
    model_version = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_version)

    #################################################################
    # Choose Model Type
    #################################################################
    if fine_tuning != "none":
       pretrained_model= 'finetuned_models/bert-base-cased_'+dataset+'_'+fine_tuning+'.pth'
       if 'enron' in pretrained_model:
          n_labels = 7
       if 'blogauth' in pretrained_model:
          n_labels = 39
       model = prepare_fine_tuned_model(pretrained_model,n_labels,model_version)

    else:
      model = BertForMaskedLM.from_pretrained(model_version)
    
    model.to('cuda')
    
    if prompt_type == "naive":
       prompt_data = parse_commoncrawl("common_crawl/commoncrawl.warc.wet")
    elif prompt_type == "informed":
       prompt_data = parse_test_json("blogauth")
    else:
       print("Invalid prompt type")    

    file_object = open(fine_tuning + '_' + dataset + '_' + prompt_type + '.txt', 'w', encoding="utf-8")


    for i in range(n_samples):
       r = np.random.randint(0, len(prompt_data))
       seed_text = " ".join(prompt_data[r:r+100].split(" ")[1:-1])
       seed_ids = tokenizer.encode(seed_text, return_tensors='pt').to('cuda')
       seed_length = len(seed_ids[0])

       output = generate_text(model,seed_ids)
       generated_tokens = output[0][seed_length:]

       #print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
       file_object.write(tokenizer.decode(generated_tokens, skip_special_tokens=True)+"\n")

       if i%1000 == 0:
          print(i, " samples have been generated")

    file_object.close()

if __name__ == '__main__':
    main()