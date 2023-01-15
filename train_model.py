from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from load_data import load_datasets
from metrics import compute_metrics

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def tokenize(batch, max_length=100):
    return bert_tokenizer(
        batch['text'],
        # add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def train(train_df, test_df, output_dir):
    train_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=50,
        weight_decay=0.01
    )
    trainer = Trainer(
        model=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2),
        tokenizer=bert_tokenizer,
        args=train_args,
        train_dataset=Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True),
        eval_dataset=Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize, batched=True),
        compute_metrics=compute_metrics,
    )
    trainer.train()


def load_and_train(data_dir='data', with_bart_aug=False, with_t5_aug=False, output_dir=None, **kwargs):
    if not output_dir:
        output_dir = f'bert-fine-tuned-{data_dir}-{"with_bart" if with_bart_aug else "no_bart"}-{"with_t5" if with_t5_aug else "no_t5"}'
    train_df, test_df = load_datasets(data_dir=data_dir, with_bart_aug=with_bart_aug, with_t5_aug=with_t5_aug)
    train(train_df, test_df, output_dir=output_dir)
