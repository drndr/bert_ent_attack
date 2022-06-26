import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

        def __init__(self, dataframe, tokenizer, max_len):
            self.tokenizer = tokenizer
            self.data = dataframe
            self.text = dataframe.text
            self.targets = self.data.labels
            self.max_len = max_len

        def __len__(self):
            return len(self.text)

        def __getitem__(self, index):
            text = str(self.text[index])
            text = " ".join(text.split())

            inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                padding='max_length',
                max_length=self.max_len,
                truncation=True,
                pad_to_max_length=True,
                return_token_type_ids=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            token_type_ids = inputs["token_type_ids"]

            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.long)
            }

def load_dataset(dataset):

    train_list = json.load(open("datasets/"+dataset+"/train_data.json",))
    train_data = np.array(list(map(lambda x: (list(x.values())[:1]), train_list)),dtype=object)
    train_labels= np.array(list(map(lambda x: list(x.values())[1], train_list)),dtype=object)
    #print("Training data size: ",len(train_data))

    test_list = json.load(open("datasets/"+dataset+"/test_data.json",))
    test_data = np.array(list(map(lambda x: list(x.values())[:1], test_list)),dtype=object)
    test_labels = np.array(list(map(lambda x: list(x.values())[1], test_list)),dtype=object)
    #print("Test data size: ",len(test_data))

    train_list = json.load(open("datasets/"+dataset+"/train_data.json",))
    train_data = np.array(list(map(lambda x: (list(x.values())[:1]), train_list)),dtype=object)
    train_labels= np.array(list(map(lambda x: list(x.values())[1], train_list)),dtype=object)
    #print("Training data size: ",len(train_data))

    test_list = json.load(open("datasets/"+dataset+"/test_data.json",))
    test_data = np.array(list(map(lambda x: list(x.values())[:1], test_list)),dtype=object)
    test_labels = np.array(list(map(lambda x: list(x.values())[1], test_list)),dtype=object)
    #print("Test data size: ",len(test_data))
    
    return train_data,test_data,train_labels,test_labels
    
def preprocess_tokenize(train_data, test_data,train_labels, test_labels, bert_model_type):
        
    ################################################################
    # Preprocess Labels
    ################################################################
    label_encoder = LabelEncoder()
    label_encoder.fit([*train_labels,*test_labels])
    train_labels_enc = label_encoder.transform(train_labels)
    test_labels_enc = label_encoder.transform(test_labels)

    ################################################################
    # Create DataFrames
    ################################################################
    train_df = pd.DataFrame()
    train_df['text'] = train_data[:,0]
    train_df['labels'] = train_labels_enc.tolist()

    test_df = pd.DataFrame()
    test_df['text'] = test_data[:,0]
    test_df['labels'] = test_labels_enc.tolist()

    print("Number of train texts ",len(train_df['text']))
    print("Number of train labels ",len(train_df['labels']))
    print("Number of test texts ",len(test_df['text']))
    print("Number of test labels ",len(test_df['labels']))

    ###############################################################
    # Define Tokenizer
    ###############################################################
    
    if bert_model_type == "bert-base-cased":
       tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif bert_model_type == "large":
       tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    elif bert_model_type == "distil":
       tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    elif bert_model_type == "roberta":
       tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    ###############################################################
    # Train-Val-Test Split
    ###############################################################
    MAX_LEN = 512
    #train_size = 0.8
    #train_dataset = train_df.sample(frac=train_size,random_state=200)
    #valid_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
    #train_dataset = train_dataset.reset_index(drop=True)
    train_dataset = train_df.reset_index(drop=True)
    test_dataset  = test_df.reset_index(drop=True)

    print("TRAIN Dataset: {}".format(train_dataset.shape))
    #print("VAL Dataset: {}".format(valid_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    #validation_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)
    test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    #return training_set,validation_set,test_set
    return training_set, test_set
    
    
def freeze_bert_weights(model):

    trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        p.requires_grad = False
        total_params += p.numel()

    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()

    print(f"Total parameters count: {total_params}") # ~108M
    print(f"Trainable parameters count: {trainable_params}") # ~7M
    
    return model