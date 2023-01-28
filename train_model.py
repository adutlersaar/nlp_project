from pathlib import Path

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer

from metrics import compute_metrics
from paraphrase.paraphrased_dataset import ParaphrasedDataset
from upload_to_hub import upload_model


def train(train_ds, test_ds, pretrained_weights, output_dir, epochs=10, learning_rate=2e-5, upload=False):
    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_weights, num_labels=2)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    if upload:
        upload_model(output_dir)


def load_and_train(pretrained_weights, data_dir='data', with_bart_aug=False, with_t5_aug=False, epochs=10,
                   learning_rate=2e-5, upload=False, **kwargs):
    output_dir = f'{pretrained_weights}-fine-tuned-{data_dir}-{"with_bart" if with_bart_aug else "no_bart"}-{"with_t5" if with_t5_aug else "no_t5"}'
    train_df, test_df = pd.read_csv(Path(data_dir, 'train.csv')), pd.read_csv(Path(data_dir, 'test.csv'))
    train_ds = ParaphrasedDataset(train_df, pretrained_weights, with_bart_aug=with_bart_aug, with_t5_aug=with_t5_aug)
    test_ds = ParaphrasedDataset(test_df, pretrained_weights)
    train(train_ds, test_ds, pretrained_weights, output_dir, epochs, learning_rate, upload=upload)
