import argparse

from bertattack import load_and_attack
from bertattack_statistics import save_bertattack_statistics
from train_model import load_and_train
from preprocess_dataset import preprocess_dataset
from plot import plot_roc
from upload_to_hub import upload_locally_saved_models, upload_model


def run():
    parser = argparse.ArgumentParser(description='Run Bert Experiment')
    parser.add_argument('--data-dir', type=str, default='data')

    subparsers = parser.add_subparsers(help='module', required=True)

    parser_preprocess = subparsers.add_parser('preprocess')
    parser_preprocess.set_defaults(func=preprocess_dataset)
    parser_preprocess.add_argument('--test-size', type=float, default=0.2)

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(func=load_and_train)
    parser_train.add_argument('--with-bart-aug', action='store_true')
    parser_train.add_argument('--with-t5-aug', action='store_true')
    parser_train.add_argument('--pretrained-weights', type=str, default='bert-base-uncased')
    parser_train.add_argument('--epochs', type=int, default=10)
    parser_train.add_argument('--learning-rate', type=float, default=2e-5)
    parser_train.add_argument('--upload', action='store_true')

    parser_bertattack = subparsers.add_parser('bertattack')
    parser_bertattack.set_defaults(func=load_and_attack)
    parser_bertattack.add_argument('--with-bart-aug', action='store_true')
    parser_bertattack.add_argument('--with-t5-aug', action='store_true')
    parser_bertattack.add_argument('--pretrained-weights', type=str, default='bert-base-uncased')
    parser_bertattack.add_argument('--use-bpe', action='store_true')
    parser_bertattack.add_argument('--epochs', type=int, default=5)
    parser_bertattack.add_argument('--learning-rate', type=float, default=1e-5)
    parser_bertattack.add_argument('--upload', action='store_true')

    parser_plot = subparsers.add_parser('plot_roc')
    parser_plot.set_defaults(func=plot_roc)
    parser_plot.add_argument('--pretrained-weights', type=str, default='bert-base-uncased')

    parser_plot = subparsers.add_parser('bertattack_statistics')
    parser_plot.set_defaults(func=save_bertattack_statistics)
    parser_plot.add_argument('--pretrained-weights', type=str, default='bert-base-uncased')

    parser_upload = subparsers.add_parser('upload')
    parser_upload.set_defaults(func=upload_model)
    parser_upload.add_argument('--pretrained-weights', type=str, required=True)

    parser_upload_all = subparsers.add_parser('upload_all')
    parser_upload_all.set_defaults(func=upload_locally_saved_models)
    parser_upload_all.add_argument('--pretrained-weights', type=str, default='bert-base-uncased')

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == '__main__':
    # wget -O parler_annotated_data.csv https://github.com/NasLabBgu/parler-hate-speech/raw/main/parler_annotated_data.csv
    run()
