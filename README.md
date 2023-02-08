# nlp_project

## Download Data

- mkdir data
- wget -O data/parler_annotated_data.csv https://github.com/NasLabBgu/parler-hate-speech/raw/main/parler_annotated_data.csv

## Preprocess

- python run.py --data-dir data preprocess --test-size 0.2

## Train

- python run.py --data-dir data train
- python run.py --data-dir data train --with-bart-aug
- python run.py --data-dir data train --with-t5-aug
- python run.py --data-dir data train --with-bart-aug --with-t5-aug

## BertAttack (Attack & Fine-Tune)

- python run.py --data-dir data bertattack
- python run.py --data-dir data bertattack --with-bart-aug
- python run.py --data-dir data bertattack --with-t5-aug
- python run.py --data-dir data bertattack --with-bart-aug --with-t5-aug

## BertAttack Statistics

- python run.py --data-dir data bertattack_statistics
