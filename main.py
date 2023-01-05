from process_dataset import process_dataset, apply_augmentations
from train_model import train


def main():
    # wget -O parler_annotated_data.csv https://github.com/NasLabBgu/parler-hate-speech/raw/main/parler_annotated_data.csv
    train_df, val_df, test_df = process_dataset('parler_annotated_data.csv', num_rows=10)
    train_df = apply_augmentations(train_df)
    train(train_df, test_df, output_dir='test-bert-fine-tune')


if __name__ == '__main__':
    main()
