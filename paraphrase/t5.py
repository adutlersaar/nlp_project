import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t5 = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser').to(device)
t5_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')


@torch.no_grad()
def t5_paraphrase(sent):
    encoding = t5_tokenizer.encode_plus(f"paraphrase: {sent} </s>", return_tensors="pt")
    outputs = t5.generate(input_ids=encoding["input_ids"].to(device),
                          attention_mask=encoding["attention_mask"].to(device),
                          do_sample=True,
                          min_length=int(0.8 * len(encoding["input_ids"][0])),
                          max_length=int(1.2 * len(encoding["input_ids"][0])),
                          top_k=50,
                          top_p=0.9,
                          early_stopping=True,
                          num_return_sequences=10
                          )

    decoded_outputs = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [dx for dx in set(decoded_outputs) if dx.lower() != sent.lower()]
