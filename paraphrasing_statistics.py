import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm

from metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_models = [
    'bert-base-uncased-fine-tuned-data-no_bart-no_t5',
    'bert-base-uncased-fine-tuned-data-no_bart-with_t5',
    'bert-base-uncased-fine-tuned-data-with_bart-no_t5',
    'bert-base-uncased-fine-tuned-data-with_bart-with_t5'
]


def tokenize(tokenizer, text):
    return tokenizer(text,
                     add_special_tokens=True,
                     truncation=True,
                     padding="max_length",
                     max_length=100,
                     return_tensors="pt")


def evaluate_on_aug(pretrained_weights='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, use_fast=False)
    t5_df = pd.read_csv('data/t5_aug_train.csv')
    bart_df = pd.read_csv('data/bart_aug_train.csv')
    train_df = pd.read_csv('data/train.csv')
    bart_texts = bart_df[~bart_df['text'].isna()].groupby('index')['text'].apply(list).reset_index().rename(
        columns={'text': 'bart_text'})
    t5_texts = t5_df[~t5_df['text'].isna()].groupby('index')['text'].apply(list).reset_index().rename(
        columns={'text': 't5_text'})
    aug_df = train_df.join(bart_texts.set_index('index')).join(t5_texts.set_index('index'))
    aug_df = aug_df[~aug_df['text'].isna() & ~aug_df['t5_text'].isna() & ~aug_df['bart_text'].isna()]
    X_test, y_test = aug_df[['text', 't5_text', 'bart_text']].values, aug_df['label'].values
    res = []

    with torch.no_grad():
        for model_name in saved_models:
            cloud_name = f'adutlersaar/{model_name.replace("data", "parler_data")}'
            model = AutoModelForSequenceClassification.from_pretrained(cloud_name, num_labels=2).to(device)
            y_pred = []
            y_t5_pred = []
            y_bart_pred = []
            for text, t5_text, bart_text in tqdm(X_test):
                y_pred.append(model(**tokenize(tokenizer, text).to(device))[0].softmax(dim=1).squeeze().cpu().numpy())
                y_t5_pred.append(
                    model(**tokenize(tokenizer, t5_text).to(device))[0].softmax(dim=1).squeeze().cpu().numpy())
                y_bart_pred.append(
                    model(**tokenize(tokenizer, bart_text).to(device))[0].softmax(dim=1).squeeze().cpu().numpy())
            del model
            orig_wrong = (np.array(y_test) != np.array(y_pred).argmax(axis=1))
            t5_changed = (np.array(y_pred).argmax(axis=1) != np.array(y_t5_pred).argmax(axis=1))
            bart_changed = (np.array(y_pred).argmax(axis=1) != np.array(y_bart_pred).argmax(axis=1))
            model_info = {'T5': 'with_t5' in model_name, 'BART': 'with_bart' in model_name}
            orig_metrics = compute_metrics((np.array(y_pred), y_test))
            bart_metrics = compute_metrics((np.array(y_bart_pred), y_test))
            t5_metrics = compute_metrics((np.array(y_t5_pred), y_test))
            res.append({**model_info,
                        'orig_accuracy': orig_metrics['accuracy'],
                        't5_accuracy': t5_metrics['accuracy'],
                        'bart_accuracy': bart_metrics['accuracy'],
                        't5_changed_pred': t5_changed.mean(),
                        'bart_changed_pred': bart_changed.mean()})
    return res


res = evaluate_on_aug()
pd.DataFrame(res)
