from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from load_data import load_datasets
from metrics import compute_metrics

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)


def tokenize(batch, max_length=100):
    return bert_tokenizer(
        batch['text'],
        # add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def train(train_df, test_df, pretrained_weights, output_dir, epochs=10):
    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )
    trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained(pretrained_weights, num_labels=2),
        tokenizer=bert_tokenizer,
        args=train_args,
        train_dataset=Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True),
        eval_dataset=Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize, batched=True),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)


def load_and_train(pretrained_weights, data_dir='data', with_bart_aug=False, with_t5_aug=False, output_dir=None, epochs=10, **kwargs):
    if not output_dir:
        output_dir = f'{pretrained_weights}-fine-tuned-{data_dir}-{"with_bart" if with_bart_aug else "no_bart"}-{"with_t5" if with_t5_aug else "no_t5"}'
    train_df, test_df = load_datasets(data_dir=data_dir, with_bart_aug=with_bart_aug, with_t5_aug=with_t5_aug)
    train(train_df, test_df, pretrained_weights, output_dir, epochs)
