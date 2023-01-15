from pathlib import Path

import pandas as pd
from paraphrase import PARAPHRASERS
from tqdm.auto import tqdm
tqdm.pandas()


def augment_dataset(paraphraser_name, data_dir='', num_samples=2, **kwargs):
    train_df = pd.read_csv(Path(data_dir, 'train.csv'))
    aug_train_df = train_df['text'].progress_map(lambda x: apply_paraphrasing(x, paraphraser_name, num_samples=num_samples)).explode().reset_index()
    aug_train_df = aug_train_df.join(train_df[['label']], on='index')[['text', 'label', 'index']].reset_index()
    aug_train_df.to_csv(Path(data_dir, f'{paraphraser_name}_aug_train.csv'))


def apply_paraphrasing(x, paraphraser_name, num_samples=2):
    try:
        return PARAPHRASERS[paraphraser_name](x)[:num_samples]
    except Exception as e:
        print(str(e))
        return []


if __name__ == '__main__':
    augment_dataset()
