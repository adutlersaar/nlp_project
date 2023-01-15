import argparse

from bertattack import load_and_attack
from paraphrase import PARAPHRASERS
from train_model import load_and_train
from augment_dataset import augment_dataset
from preprocess_dataset import preprocess_dataset


def run():
    parser = argparse.ArgumentParser(description='Run Bert Experiment')
    parser.add_argument('--data-dir', type=str, default='data')

    subparsers = parser.add_subparsers(help='module', required=True)

    parser_preprocess = subparsers.add_parser('preprocess')
    parser_preprocess.set_defaults(func=preprocess_dataset)
    parser_preprocess.add_argument('--test-size', type=float, default=0.2)

    parser_augment = subparsers.add_parser('augment')
    parser_augment.set_defaults(func=augment_dataset)
    parser_augment.add_argument('--paraphraser-name', choices=PARAPHRASERS, required=True)

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(func=load_and_train)
    parser_train.add_argument('--with-bart-aug', action='store_true')
    parser_train.add_argument('--with-t5-aug', action='store_true')
    parser_train.add_argument('--output-dir', type=str)

    parser_bertattack = subparsers.add_parser('bertattack')
    parser_bertattack.set_defaults(func=load_and_attack)
    parser_bertattack.add_argument('--with-bart-aug', action='store_true')
    parser_bertattack.add_argument('--with-t5-aug', action='store_true')
    parser_bertattack.add_argument('--output-path', type=str)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    # wget -O parler_annotated_data.csv https://github.com/NasLabBgu/parler-hate-speech/raw/main/parler_annotated_data.csv
    run()
