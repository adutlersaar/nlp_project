from pathlib import Path

from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


def preprocess_dataset(test_size=0.2, num_rows=None, data_dir='data'):
    train_df, val_df, test_df = process_dataset(Path(data_dir, 'parler_annotated_data.csv'), test_size=test_size,
                                                num_rows=num_rows)
    train_df.to_csv(Path(data_dir, 'train.csv'))
    val_df.to_csv(Path(data_dir, 'val.csv'))
    test_df.to_csv(Path(data_dir, 'test.csv'))


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


if __name__ == '__main__':
    preprocess_dataset()
