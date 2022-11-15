# A Study on Extracting Named Entities from Fined-tuned BERT Models

This repository contains the code and instructions on how to reproduce the results of the paper ”A Study on Extracting Named Entities from Fined-tuned BERT Models? 
The goal of the study was to analyze the extent of named entity memorization in fine-tuned BERT
models. We ran experiments on two datasets, using three different fine-tuning methods and two
prompting strategies. We split the repository into different subfolders and scripts, based on the
different components.

### Folder structure:
    ├── Main                                 # Code
    |   ├── common_crawl                     # Code 
    │   ├── datasets                         # Code 
        │   ├── blogauth                     # Code 
        |   ├── enron
    |   ├── entities                        
    |   ├── finetuned_models
    |   ├── samples
    └── README                               # Project structure overview
