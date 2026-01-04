from pathlib import Path
import torch
import nltk
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import numpy as np

# --- CONFIGURATION (Adapted for local environment) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cnn_dailymail_processed"
FINE_TUNED_PATH = PROJECT_ROOT / "models" / "final_model_full"
BASELINE_MODEL = "facebook/bart-large-cnn"

def decode_text(token_ids, tokenizer):
    token_ids = [t for t in token_ids if t != -100]
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def generate_summary(text, model, tokenizer, device, max_length=128):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=1024,
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

def main():
    nltk.download("punkt", quiet=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Models
    print(f"Loading Fine-Tuned Model from: {FINE_TUNED_PATH}")
    try:
        tokenizer_ft = AutoTokenizer.from_pretrained(str(FINE_TUNED_PATH))
        model_ft = AutoModelForSeq2SeqLM.from_pretrained(str(FINE_TUNED_PATH)).to(device)
        model_ft.eval()
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return

    print(f"Loading Baseline Model: {BASELINE_MODEL}")
    tokenizer_base = AutoTokenizer.from_pretrained(BASELINE_MODEL)
    model_base = AutoModelForSeq2SeqLM.from_pretrained(BASELINE_MODEL).to(device)
    model_base.eval()

    # 2. Load Data
    print(f"Loading Test Data from: {PROCESSED_DATA_PATH}")
    dataset_full = load_from_disk(str(PROCESSED_DATA_PATH))
    dataset = dataset_full["test"]
    print(f"Test Set Size: {len(dataset)}")

    # 3. Single Sample Demonstration
    print("\n" + "="*50)
    print("DEMONSTRATION (Sample 0)")
    print("="*50)
    
    sample = dataset[0]
    article = decode_text(sample["input_ids"], tokenizer_base)
    reference = decode_text(sample["labels"], tokenizer_base)

    baseline_summary = generate_summary(article, model_base, tokenizer_base, device)
    finetuned_summary = generate_summary(article, model_ft, tokenizer_ft, device)

    print("\nINPUT ARTICLE:\n", article[:500] + "...") # Truncate for display
    print("\nGROUND TRUTH SUMMARY:\n", reference)
    print("\nBEFORE (Baseline Model):\n", baseline_summary)
    print("\nAFTER (Fine-Tuned Model):\n", finetuned_summary)

    # 4. Batch Evaluation (First 100 samples)
    EVAL_SIZE = 100
    print("\n" + "="*50)
    print(f"RUNNING EVALUATION ON FIRST {EVAL_SIZE} SAMPLES")
    print("="*50)

    dataset_eval = dataset.select(range(EVAL_SIZE))
    pred_base = []
    pred_ft = []
    refs = []

    for i, item in enumerate(dataset_eval):
        if (i+1) % 10 == 0:
            print(f"Processing {i+1}/{EVAL_SIZE}...")
            
        article = decode_text(item["input_ids"], tokenizer_base)
        reference = decode_text(item["labels"], tokenizer_base)

        pred_base.append(generate_summary(article, model_base, tokenizer_base, device))
        pred_ft.append(generate_summary(article, model_ft, tokenizer_ft, device))
        refs.append(reference)

    # 5. Compute ROUGE
    rouge = evaluate.load("rouge")
    
    # ROUGE requires newlines for sentences
    pred_base_nl = ["\n".join(nltk.sent_tokenize(p.strip())) for p in pred_base]
    pred_ft_nl = ["\n".join(nltk.sent_tokenize(p.strip())) for p in pred_ft]
    refs_nl = ["\n".join(nltk.sent_tokenize(r.strip())) for r in refs]

    base_scores = rouge.compute(predictions=pred_base_nl, references=refs_nl)
    ft_scores = rouge.compute(predictions=pred_ft_nl, references=refs_nl)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    print("\nBASELINE ROUGE SCORES:")
    for k, v in base_scores.items():
        print(f"{k}: {v * 100:.4f}")

    print("\nFINE-TUNED ROUGE SCORES:")
    for k, v in ft_scores.items():
        print(f"{k}: {v * 100:.4f}")

if __name__ == "__main__":
    main()
