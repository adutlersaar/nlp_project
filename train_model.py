from pathlib import Path

from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import pandas as pd
import evaluate

accuracy_metric = evaluate.load("accuracy")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def tokenize(batch, max_length=100):
    return bert_tokenizer(
        batch['text'],
        #add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def model_init():
    return BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)


def train(train_df, test_df, output_dir='bert-fine-tuned', model_init=model_init):
    train_args = TrainingArguments(
        output_dir=f"./{output_dir}-results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=50,
        weight_decay=0.01
    )
    trainer = Trainer(
        model_init=model_init,
        tokenizer=bert_tokenizer,
        args=train_args,
        train_dataset=Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True),
        eval_dataset=Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize, batched=True),
        compute_metrics=compute_accuracy,
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == '__main__':
    train(pd.read_csv(Path('data', 'train.csv')), pd.read_csv(Path('data', 'test.csv')), output_dir='bert-fine-tuned')
    train(pd.read_csv(Path('data', 'aug_train.csv')), pd.read_csv(Path('data', 'test.csv')), output_dir='bert-fine-tuned-with-paraphrasing')
