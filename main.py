from process_dataset import process_dataset


def main():
    # wget -O parler_annotated_data.csv https://github.com/NasLabBgu/parler-hate-speech/raw/main/parler_annotated_data.csv
    train_df, test_df = process_dataset('parler_annotated_data.csv', num_rows=10)
    print(train_df)


if __name__ == '__main__':
    main()
