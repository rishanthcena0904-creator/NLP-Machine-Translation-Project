# Generated from: BartEval.ipynb
# Converted at: 2026-01-04T15:21:16.454Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import torch
import nltk
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import matplotlib.pyplot as plt

nltk.download("punkt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


FINE_TUNED_PATH = r"C:\Users\AKASH\Desktop\New folder\final_model_full\final_model_full"

tokenizer_ft = AutoTokenizer.from_pretrained(FINE_TUNED_PATH)
model_ft = AutoModelForSeq2SeqLM.from_pretrained(FINE_TUNED_PATH)

model_ft.to(device)
model_ft.eval()


BASELINE_MODEL = "facebook/bart-large-cnn"

tokenizer_base = AutoTokenizer.from_pretrained(BASELINE_MODEL)
model_base = AutoModelForSeq2SeqLM.from_pretrained(BASELINE_MODEL)

model_base.to(device)
model_base.eval()


TEST_DATA_PATH = r"C:\Users\AKASH\Desktop\New folder\cnn_dailymail_processed\cnn_dailymail_processed\test\data-00000-of-00001.arrow"

dataset = Dataset.from_file(TEST_DATA_PATH)
print(dataset)


dataset.column_names


def decode_text(token_ids, tokenizer):
    token_ids = [t for t in token_ids if t != -100]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def generate_summary(text, model, tokenizer, max_length=150, input_max_length=1024):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=input_max_length,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


sample = dataset[0]

article = decode_text(sample["input_ids"], tokenizer_base)
reference = decode_text(sample["labels"], tokenizer_base)

baseline_summary = generate_summary(
    article, model_base, tokenizer_base
)

finetuned_summary = generate_summary(
    article, model_ft, tokenizer_ft
)

print("INPUT ARTICLE:\n", article)
print("\nGROUND TRUTH SUMMARY:\n", reference)
print("\nBEFORE (Baseline Model):\n", baseline_summary)
print("\nAFTER (Fine-Tuned Model):\n", finetuned_summary)


dataset_eval = dataset.select(range(100))

pred_base = []
pred_ft = []
refs = []

for item in dataset_eval:
    article = decode_text(item["input_ids"], tokenizer_base)
    reference = decode_text(item["labels"], tokenizer_base)

    pred_base.append(
        generate_summary(article, model_base, tokenizer_base)
    )

    pred_ft.append(
        generate_summary(article, model_ft, tokenizer_ft)
    )

    refs.append(reference)


rouge = evaluate.load("rouge")

base_scores = rouge.compute(predictions=pred_base, references=refs)
ft_scores = rouge.compute(predictions=pred_ft, references=refs)


print("BASELINE ROUGE SCORES")
for k, v in base_scores.items():
    print(f"{k}: {v:.4f}")

print("\nFINE-TUNED ROUGE SCORES")
for k, v in ft_scores.items():
    print(f"{k}: {v:.4f}")
