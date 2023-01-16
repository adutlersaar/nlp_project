from pathlib import Path

import pandas as pd


def load_datasets(data_dir='data', with_bart_aug=False, with_t5_aug=False):
    train_csvs = ['train.csv']
    if with_bart_aug:
        train_csvs.append('bart_aug_train.csv')
    if with_t5_aug:
        train_csvs.append('t5_aug_train.csv')
    train_df = pd.concat([pd.read_csv(Path(data_dir, f_name))[['text', 'label']] for f_name in train_csvs]).reset_index()
    cleaned_shuffled_train_df = train_df[~train_df.text.isna()].sample(frac=1).reset_index()
    test_df = pd.read_csv(Path(data_dir, 'test.csv'))
    return cleaned_shuffled_train_df, test_df
