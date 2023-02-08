# Parler Hate Speech Classification

- Every functionality in this project can be accessed from run.py
- We used Weights & Biases platform to log our experiments.
- We used BGU's GPU Cluster to run the experiments.
- The raw & preprocessed datasets, the bertattack results (adversarial data), and the roc plot, are provided in "data.zip"
- "data.zip" can be downloaded from: https://drive.google.com/file/d/1ku2XMjs7cf5O0oqJ4okmoXLIjO3Bykj3/view?usp=sharing

## BertAttack - https://github.com/LinyangLee/BERT-Attack
- The code for BERTATTACK was copied from the mentioned repository, and modified for our needs.
- The code was designed specifically for "bert-base-uncased", which we will use with as well.

## Download Data

- mkdir data
- wget -O data/parler_annotated_data.csv https://github.com/NasLabBgu/parler-hate-speech/raw/main/parler_annotated_data.csv

## Preprocess

- python run.py --data-dir data preprocess --test-size 0.2

## Train

- python run.py --data-dir data train \[--with-bart-aug\] \[--with-t5-aug\]

## BertAttack (Attack & Fine-Tune)

- python run.py --data-dir data bertattack \[--with-bart-aug\] \[--with-t5-aug\]

## Utilities (used for analysis, not required at training)

- bertattack_statistics - used to extract statistics from bertattack results
- plot_roc - used to plot ROC plot of the results
- upload (& upload_all) - uploads local models to huggingface's hub
