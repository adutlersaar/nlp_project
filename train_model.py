from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import pandas as pd
import evaluate

accuracy_metric = evaluate.load("accuracy")


def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def model_init():
    return AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)


def train(train_df, test_df, output_dir='bert-fine-tuned'):
    trainer = Trainer(
        model_init=model_init,
        args=TrainingArguments(output_dir="bert_trainer", evaluation_strategy="epoch"),
        train_dataset=Dataset.from_pandas(train_df[['text', 'label']]),
        eval_dataset=Dataset.from_pandas(test_df[['text', 'label']]),
        compute_metrics=compute_accuracy,
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == '__main__':
    train(pd.read_csv('train.csv'), pd.read_csv('test.csv'))
