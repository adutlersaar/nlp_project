from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm

from train_model import tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = ['bert-base-uncased-fine-tuned-data-no_bart-no_t5',
          'bert-base-uncased-fine-tuned-data-no_bart-with_t5',
          'bert-base-uncased-fine-tuned-data-with_bart-no_t5',
          'bert-base-uncased-fine-tuned-data-with_bart-with_t5']

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)


def plot_roc(data_dir='data'):
    test_df = pd.read_csv(Path(data_dir, 'test.csv'))
    X_test, y_test = test_df['text'].values, test_df['label'].values

    fig, ax = plt.subplots(figsize=(12, 12))
    with torch.no_grad():
        for model_name in models:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
            y_pred = []
            for sent in tqdm(X_test):
                y_pred.append(model(**tokenize(tokenizer, {'text': sent}).to(device))[0].softmax(dim=1).squeeze()[1].item())
            del model
            RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax, name=model_name)
    ax.legend(prop={'size': 13}, loc='lower right')
    plt.savefig(str(Path(data_dir, 'roc_curve.png')), bbox_inches='tight')


if __name__ == '__main__':
    plot_roc()