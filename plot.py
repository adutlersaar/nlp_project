from pathlib import Path

import pandas as pd
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm

from paraphrase.paraphrased_dataset import ParaphrasedDataset
from upload_to_hub import get_locally_saved_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_roc(data_dir='data', pretrained_weights='bert-base-uncased', **kwargs):
    test_df = pd.read_csv(Path(data_dir, 'test.csv'))
    tokenize = ParaphrasedDataset(test_df, pretrained_weights).tokenize_text
    X_test, y_test = test_df['text'].values, test_df['label'].values
    saved_models = get_locally_saved_models(pretrained_weights)

    fig, ax = plt.subplots(figsize=(12, 12))
    with torch.no_grad():
        for model_name in saved_models:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
            y_pred = []
            for sent in tqdm(X_test):
                y_pred.append(model(**tokenize(sent).to(device))[0].softmax(dim=1).squeeze()[1].item())
            del model
            RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax, name=model_name)
    ax.legend(prop={'size': 13}, loc='lower right')
    plt.savefig(str(Path(data_dir, 'roc_curve.png')), bbox_inches='tight')


if __name__ == '__main__':
    plot_roc()
