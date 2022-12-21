from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from paraphrase import bart_paraphrase, t5_paraphrase


def process_dataset(csv_file, test_size=0.2, num_rows=None):
    df = pd.read_csv(csv_file)
    print(df.describe())
    df['label'] = df['label_mean'].map(regression_to_labels)
    print(df['label'].value_counts())
    df = df[df['label'] != -1][['text', 'label']]
    if num_rows:
        df = df.iloc[:num_rows]
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'])
    return apply_augmentations(train_df), test_df


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
    ])


def process_augmentation(df, aug_func, name):
    aug_df = df['text'].progress_map(aug_func).explode().reset_index().join(df[['label']], on='index')[['text', 'label']]
    aug_df['is_augmentation'] = name
    return aug_df.reset_index()


def filter_paraphrase(x):
    return x[:2]
