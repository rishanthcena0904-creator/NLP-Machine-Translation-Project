
from datasets import load_dataset
import os

print("Step 1: Downloading CNN/DailyMail dataset from Hugging Face...")
# We use version '3.0.0' which is the standard benchmark
raw_datasets = load_dataset("cnn_dailymail", "3.0.0")

# Quick check
print(f"Downloaded {len(raw_datasets['train'])} training examples.")

# ---------------------------------------------------------
# SAVE RAW DATA TO DISK
# ---------------------------------------------------------

save_path_raw = "./cnn_dailymail_datasets"
print(f"Step 2: Saving raw dataset to '{save_path_raw}'...")

raw_datasets.save_to_disk(save_path_raw)

print("✅ DONE!")