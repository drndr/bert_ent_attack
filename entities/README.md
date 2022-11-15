This folder contains the collected entity lists as .json files and two executable python scripts used for preparing the evaluation of the experiments:

### Collect Entities
The save_ner.py script contains the implementation to collect the entities from the
fine-tuning datasets

#### Get up and running
1. Make sure the ./datasets folder contains the train_data.json files
2. Check for dependencies: numpy, pythorch, spacy

#### Configuration

A entity list can be created by setting the following global variables in
the script:
* dataset

### Remove Entities that are also present in the BERT pre-training data
The remove_pretrain_ents.py script contains the implementation to remove entities from the list of
fine-tuned entities that are also present in the pretraining datasets (Wikipedia, Book
Corpus)

#### Get up and running
1. Make sure the ./entities folder contains the original entity files,
collected with save_ner.py
2. Check for dependencies: datasets, numpy, ahocorapy

#### Configuration
A specific entity list can be created by setting the following global
variables in the script:
* entity_file
