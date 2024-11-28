# Memorization of Named Entities in Fine-tuned BERT Models

This repository contains the code and instructions on how to reproduce the results of the paper ”Memorization of Named Entities in Fine-tuned BERT Models" published at CD-MAKE 2023. 
The goal of the study was to analyze the extent of named entity memorization in fine-tuned BERT
models. We ran experiments on two datasets, using three different fine-tuning methods and two
prompting strategies. We split the repository into different subfolders and scripts, based on the
different components.

arxiv link: https://arxiv.org/abs/2212.03749

## Folder structure:
    ├── Main                                 # Main scripts for running the experiments 
    |   ├── common_crawl                     # Script for sourcing the CC dataset, location of the downloaded .wet file
    │   ├── datasets                         # Datasets used in the study 
        │   ├── blogauth                     # Train-test split of the BlogAuthorship dataset with the preprocessing script
        |   ├── enron                        # Train-test split of the Enron dataset with the preprocessing script
    |   ├── entities                         # Collected entity lists stored as .json, scripts for collecting the entities
    |   ├── finetuned_models                 # Saved fine-tuned models
    |   ├── samples                          # Generated text samples as .txt files
    └── README                               # Project structure overview
    
The subfolders datasets and entities contain additional README files with information on how to prepare the experiments.

## Run experiments

### Fine-tuning

The finetune_classification.py script contains our implementation of running the fine-tuning setups.

#### Get up and running

1. Make sure the ./datasets contains the training and test set
2. Check for dependencies: `numpy`, `pytorch`, `transformers`, `opacus`, `sklearn`

#### Configuration

A specific fine-tuning setup can be configured by setting the following
global variables in the script:
* dataset
* bert_model_type
* TRAIN_BATCH_SIZE
* TEST_BATCH_SIZE
* LEARNING_RATE
* n_epochs
* freeze_pretrained_layers (optional)
* differential_privacy (optional)
* noise_multiplier (optional)
* grad_clipping_threshold (optional)

### Text Generation

The text_generation.py script contains our implementation of running the text generation setups.

#### Get up and running

1. Make sure the ./finetuned_models contains the models to be used
2. Check for dependencies: `numpy`, `pytorch`, `transformers`

#### Configuration

A specific text generation setup can be configured by setting the following
global variables in the script:
* fine_tuning
* dataset
* prompt_type
* n_samples

### Evaluation

In this executable python script contains the implementation for evaluating our experimental
configurations.

#### Get up and running

1. Make sure the ./entities and ./samples folders contains the entity lists and text
samples
2. Check for dependencies: `numpy` , `ahocorapy`

#### Confifugration

A specific evaluation setup can be configured by setting the following
global variables in the script:
* entity_file
* text
* k_value
