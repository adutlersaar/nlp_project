import csv
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
import pandas as pd
from paraphrase import bart_paraphrase, t5_paraphrase
from tqdm.auto import tqdm
tqdm.pandas()


def save_processed_dataset(test_size=0.2, num_rows=None, files_prefix=''):
    train_df, val_df, test_df = process_dataset('data/parler_annotated_data.csv', test_size=test_size, num_rows=num_rows)
    train_df.to_csv(Path('data', f'{files_prefix}train.csv'))
    val_df.to_csv(Path('data', f'{files_prefix}val.csv'))
    test_df.to_csv(Path('data', f'{files_prefix}test.csv'))
    aug_train_df = apply_augmentations(train_df)
    aug_train_df.to_csv(Path('data', f'{files_prefix}aug_train.csv'))


def process_dataset(csv_file, val_size=0.2, test_size=0.2, num_rows=None):
    df = pd.read_csv(csv_file)
    print(df.describe())
    df['label'] = df['label_mean'].map(regression_to_labels)
    print(df['label'].value_counts())
    df = df[df['label'] != -1][['text', 'label']]
    if num_rows:
        df = df.iloc[:num_rows]
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['label'])
    return train_df, val_df, test_df


def regression_to_labels(x):
    if x >= 3.0:
        return 1
    elif x <= 2.0:
        return 0
    return -1


def apply_augmentations(df):
    df = df.copy()[['text', 'label']]
    df['is_augmentation'] = 'Original'
    return pd.concat([
        df,
        process_augmentation(df, lambda x: filter_paraphrase(bart_paraphrase(x)), 'bart'),
        process_augmentation(df, lambda x: filter_paraphrase(t5_paraphrase(x)), 't5')
    ]).sample(frac=1).reset_index()


def process_augmentation(df, aug_func, name):
    aug_df = df['text'].progress_map(aug_func).explode().reset_index().join(df[['label']], on='index')[
        ['text', 'label']]
    aug_df['is_augmentation'] = name
    return aug_df.reset_index()


def filter_paraphrase(x):
    return x[:2]


if __name__ == '__main__':
    save_processed_dataset()
