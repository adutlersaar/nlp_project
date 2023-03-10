from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def upload_model(pretrained_weights, **kwargs):
    print(f'Loading {pretrained_weights}')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_weights, num_labels=2)
    print(f'Uploading {pretrained_weights}')
    tokenizer.push_to_hub(pretrained_weights)
    model.push_to_hub(pretrained_weights)
    print(f'Uploaded {pretrained_weights} Successfully')


def get_locally_saved_models(pretrained_weights):
    return [p.name for p in Path('.').glob(f'{pretrained_weights}-fine-tuned-*/')]


def upload_locally_saved_models(pretrained_weights, **kwargs):
    saved_models = get_locally_saved_models(pretrained_weights)
    for model_name in saved_models:
        upload_model(model_name)
