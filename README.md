# A Study on Extracting Named Entities from Fined-tuned BERT Models

This repository contains the code and instructions on how to reproduce the results of the paper ”A Study on Extracting Named Entities from Fined-tuned BERT Models 
The goal of the study was to analyze the extent of named entity memorization in fine-tuned BERT
models. We ran experiments on two datasets, using three different fine-tuning methods and two
prompting strategies. We split the repository into different subfolders and scripts, based on the
different components.

## Folder structure:
    ├── Main                                 # Main scripts for running the experiments 
    |   ├── common_crawl                     # Script for sourcing the CC dataset, location of the downloaded .wet file
    │   ├── datasets                         # Datasets used in the study 
        │   ├── blogauth                     # Train-test split of the BlogAuthorship dataset with the preprocessing script
        |   ├── enron                        # Train-test split of the Enron dataset with the preprocessing script
    |   ├── entities                         # Collected entity lists stored as .json, scripts for collecting the entities
    |   ├── finetuned_models                 # Saved fine-tuned models
    |   ├── samples                          # Generated text samples
    └── README                               # Project structure overview
    
The subfolders contain additional README files to prepare the experiments.

## Run experiments

### Fine-tuning

The finetune_classification.py contains our implementation of running the fine-tuning setups.

#### Get up and running

1. Make sure the ./datasets contains the training and test set
2. Check for dependencies: numpy, pytorch, transformers, opacus, sklearn

#### Configuration

A specific fine-tuning setup can be configured by setting the following
global variables in the script:
● dataset
● bert_model_type
● TRAIN_BATCH_SIZE
● TEST_BATCH_SIZE
● LEARNING_RATE
● n_epochs
● freeze_pretrained_layers (optional)
● differential_privacy (optional)
● noise_multiplier (optional)
● grad_clipping_threshold (optional
