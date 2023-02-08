from pathlib import Path
import pandas as pd


def save_bertattack_statistics(pretrained_weights='bert-base-uncased', data_dir='data', **kwargs):
    stats = []

    for bertattack_result_file in Path('data').glob(f'{pretrained_weights}-fine-tuned-{data_dir}-*-bertattack.json'):
        file_name = bertattack_result_file.name
        stat = {'BPE': 'with_bpe' in file_name, 'T5': 'with_t5' in file_name, 'BART': 'with_bart' in file_name}

        adv_train_df = pd.read_json(bertattack_result_file)
        adv_train_df = adv_train_df[adv_train_df['success'].isin([1, 2, 4])]
        stat['attack_succeed'] = len(adv_train_df[adv_train_df['success'] == 4]) / len(adv_train_df)
        stat['attack_failed'] = len(adv_train_df[adv_train_df['success'].isin([1, 2])]) / len(adv_train_df)

        change_rate = adv_train_df[adv_train_df['success'] == 4].apply(lambda x: (x['change'] / x['num_word']), axis=1)
        stat['mean_change_rate'] = change_rate.values.mean()
        stat['std_change_rate'] = change_rate.values.std()
        stat['avg_queries'] = adv_train_df[adv_train_df['success'] == 4]['query'].values.mean()
        stats.append(stat)
    stats_df = pd.DataFrame(stats).sort_values(['BPE', 'T5', 'BART'], ascending=True)
    stats_df.to_csv(Path(data_dir, f'{pretrained_weights}-bertattack-statistics.csv'))
