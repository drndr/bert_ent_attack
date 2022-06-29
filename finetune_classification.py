import numpy as np

import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader

from transformers import BertConfig, BertForSequenceClassification

from opacus import PrivacyEngine

from sklearn import metrics

import os
import logging
import random
from timeit import default_timer as timer

#Custom Lib
import util

os.environ["CUDA_VISIBLE_DEVICES"]="0"

##############################################################
# Set Random Seeds for Reproducibility
##############################################################

seed_value = 420
torch.manual_seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

##############################################################
# Experiment Setup
##############################################################

# Base Configs
dataset = "enron"
bert_model_type = "bert-base-cased"  # bert-base-cased
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
LEARNING_RATE = 1e-3
n_epochs = 10

if dataset=="enron":
    n_labels=7 #Enron
else:
    n_labels=39 #Blogauth

# Fine-tuning method
freeze_pretrained_layers = True # Must be True if differential_privacy is True
differential_privacy = False

# DP specific parameters
grad_clipping_threshold = 10.0
noise_multiplier = 0.5

##############################################################
# Train Function
##############################################################
def train_model(start_epochs, n_epochs, training_loader, model, optimizer, device, privacy_engine=None, DELTA=None):
    model.train()
    for epoch in range(start_epochs, n_epochs + 1):
        iteration_start = timer()
        train_loss = []

        ######################
        # Train the model #
        ######################

        print('############# Epoch {}: Training Start   #############'.format(epoch))
        for batch_idx, data in enumerate(training_loader):
        
            optimizer.zero_grad()

            # Forward            
            batch_ids = data['ids'].to(device, dtype=torch.long)
            batch_masks = data['mask'].to(device, dtype=torch.long)
            batch_token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            batch_targets = data['targets'].to(device, dtype=torch.long)

            inputs = {'input_ids':            batch_ids,
                      'attention_mask':           batch_masks,
                      'token_type_ids': batch_token_type_ids,
                      'labels': batch_targets}

            outputs = model(**inputs)

            loss = outputs[0]   # output = loss, logits, hidden_states, attentions
            train_loss.append(loss.item())
            
            # Backward
            loss.backward()
            optimizer.step()
            
            if privacy_engine is not None:
               eps = privacy_engine.get_epsilon(DELTA)
        
        if privacy_engine is not None:            
           print(
                 f"Train Loss: {np.mean(train_loss):.4f} \t"
                 f"(ε = {eps:.4f})"
                )
        else:
           print(
                 f"Train Loss: {np.mean(train_loss):.4f} \t"
                )

        iteration_end = timer()
        print("Epoch time in minutes: ",(iteration_end-iteration_start)/60)
        
    return model

##############################################################
# Test Function
##############################################################
def test_model(testing_loader, model, device):
    model.eval()	
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
        
            batch_ids = data['ids'].to(device, dtype=torch.long)
            batch_masks = data['mask'].to(device, dtype=torch.long)
            batch_token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            batch_targets = data['targets'].to(device, dtype=torch.long)

            inputs = {'input_ids':            batch_ids,
                      'attention_mask':           batch_masks,
                      'token_type_ids': batch_token_type_ids,
                      'labels': batch_targets}

            outputs = model(**inputs)
            preds = np.argmax(outputs[1].detach().cpu().numpy(), axis=1)
            #print(preds)
            #print(targets)
            
            fin_targets.extend(batch_targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(preds.tolist())
    return fin_outputs, fin_targets

##############################################################
# Run Fine-tuning
##############################################################
def main():
    
    # Prepare data
    train_data,test_data,train_labels,test_labels = util.load_dataset(dataset)
      
    training_set,test_set = util.preprocess_tokenize(train_data, test_data, train_labels, test_labels, bert_model_type)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': TEST_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }      

    training_loader = DataLoader(training_set,drop_last=True, **train_params)
    test_loader = DataLoader(test_set, drop_last=True, **test_params)
    
    # Set Cuda    
    device = 'cuda' if cuda.is_available() else 'cpu'
    logging.basicConfig(level=logging.ERROR)
    
    # Load model config
    model_name = bert_model_type
    
    
    if differential_privacy:
       config = BertConfig.from_pretrained(
           model_name,
           num_labels=n_labels,
           hidden_dropout_prob=0.0,
           classifier_dropout=0.0
       )
       
    else:
       config = BertConfig.from_pretrained(
           model_name,
           num_labels=n_labels,
           classifier_dropout=0.3
       )

    # Load model
    model = BertForSequenceClassification.from_pretrained(
       "bert-base-cased",
       config=config,
    )
    
    model.to(device)
    model.train()
    
    # Freeze weights
    if freeze_pretrained_layers:
       model = util.freeze_bert_weights(model)

    # Define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    start = timer() # Start measuring time for Train and Inference
    
    # Train model
    if differential_privacy:
        privacy_engine = PrivacyEngine()                               
        model, optimizer, training_dataloader = privacy_engine.make_private(module=model, optimizer=optimizer, data_loader=training_loader, noise_multiplier=noise_multiplier,max_grad_norm=grad_clipping_threshold)
        DELTA = 1/len(train_data)
        print("DELTA value: ",DELTA)
        trained_model = train_model(1, n_epochs, training_loader, model, optimizer, device, privacy_engine, DELTA)
        
    else:
        trained_model = train_model(1, n_epochs, training_loader, model, optimizer, device)
    
    # Save model
    torch.save(trained_model.state_dict(), bert_model_type+"_"+dataset+".pth")
    
    # Test Model
    outputs, targets = test_model(test_loader, trained_model, device)

    end = timer()

    targets=np.array(targets).astype(int)
    outputs=np.array(outputs).astype(int)
    
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print("Evaluation")
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    print("Train+Inference time in minutes: ",(end-start)/60)
    
if __name__ == '__main__':
    main()