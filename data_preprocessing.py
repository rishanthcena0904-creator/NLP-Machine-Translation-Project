
from datasets import load_from_disk
from transformers import AutoTokenizer

# ---------------------------------------------------------
# 1. LOAD THE SAVED RAW DATA
# ---------------------------------------------------------
# 
print("Step 1: Loading raw data from local folder...")
try:
    raw_datasets = load_from_disk("./cnn_dailymail_datasets")
except FileNotFoundError:
    print("❌ Error: Could not find the folder './cnn_dailymail_datasets'.")
    exit()

# ---------------------------------------------------------
# 2. SETUP BART TOKENIZER
# ---------------------------------------------------------
print("Step 2: Loading BART tokenizer...")
model_checkpoint = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ---------------------------------------------------------
# 3. DEFINE PROCESSING FUNCTION
# ---------------------------------------------------------
def preprocess_function(examples):
    inputs = examples["article"]
    targets = examples["highlights"]

    # Tokenize inputs (Articles) - Max 1024
    model_inputs = tokenizer(
        inputs, 
        max_length=1024, 
        truncation=True
    )

    # Tokenize targets (Summaries) - Max 128
    labels = tokenizer(
        text_target=targets, 
        max_length=128, 
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ---------------------------------------------------------
# 4. RUN PROCESSING
# ---------------------------------------------------------
print("Step 3: Tokenizing data")

tokenized_datasets = raw_datasets.map(
    preprocess_function, 
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# ---------------------------------------------------------
# 5. SAVE FINAL PROCESSED DATA
# ---------------------------------------------------------
save_path_final = "./cnn_dailymail_processed"
print(f"Step 4: Saving processed data to '{save_path_final}'...")

tokenized_datasets.save_to_disk(save_path_final)

print("\nWayyy to Go Gurlll! Data is fully processed and saved.")