from process_dataset import process_dataset
from train_model import train


def main():
    # wget -O parler_annotated_data.csv https://github.com/NasLabBgu/parler-hate-speech/raw/main/parler_annotated_data.csv
    train_df, test_df = process_dataset('parler_annotated_data.csv', num_rows=10)
    print(train_df)
    train(train_df, test_df)


if __name__ == '__main__':
    main()
