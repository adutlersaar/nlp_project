import torch
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bart = BartForConditionalGeneration.from_pretrained("stanford-oval/paraphraser-bart-large").to(device)
bart_tokenizer = BartTokenizer.from_pretrained("stanford-oval/paraphraser-bart-large")


@torch.no_grad()
def bart_paraphrase(sent):
    input_ids = bart_tokenizer(sent, return_tensors="pt")['input_ids']
    outputs = bart.generate(input_ids.to(device),
                            num_beams=10,
                            top_k=50,
                            top_p=0.9,
                            temperature=0.5,
                            min_length=int(0.8 * len(input_ids[0])),
                            max_length=int(1.2 * len(input_ids[0])),
                            early_stopping=True,
                            num_return_sequences=5)
    decoded_outputs = bart_tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [dx for dx in set(decoded_outputs) if dx.lower().strip() != sent.lower().strip()]
