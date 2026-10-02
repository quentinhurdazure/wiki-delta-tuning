import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# ---------------------------------------------------------------------------
# 1. Configuration & Type Hinting
# ---------------------------------------------------------------------------
MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR: str = "./models/base_wiki_model"

def main() -> None:
    print(f"--- Initializing Lab Stage 1: Loading {MODEL_NAME} ---")

    # -----------------------------------------------------------------------
    # 2. Load Tokenizer & Model
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    # -----------------------------------------------------------------------
    # 3. Baseline Wikipedia Dataset (Topic -> Text -> Keywords)
    # -----------------------------------------------------------------------
    mock_wiki_data = {
        "text": [
            "Topic: Artificial Intelligence. Text: Artificial intelligence is simulation of human intelligence by machines. Keywords: AI, machine learning, intelligence",
            "Topic: Microsoft Copilot. Text: Microsoft Copilot is an enterprise AI assistant built on LLMs. Keywords: Copilot, Microsoft, AI assistant",
            "Topic: Quantum Computing. Text: Quantum computing utilizes superposition and entanglement to process data. Keywords: quantum, superposition, physics"
        ]
    }
    dataset = Dataset.from_dict(mock_wiki_data)

    # -----------------------------------------------------------------------
    # 4. Tokenization Pipeline
    # -----------------------------------------------------------------------
    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # -----------------------------------------------------------------------
    # 5. MLOps Training Configuration
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,              # Low epochs for rapid local execution
        per_device_train_batch_size=1,   # Keeps memory usage low
        logging_steps=1,
        save_strategy="no",              # Avoid saving intermediate large checkpoints
        learning_rate=5e-5,
        weight_decay=0.01,
        use_cpu=not torch.cuda.is_available()  # Uses CUDA GPU if available, else falls back to CPU
    )

    # -----------------------------------------------------------------------
    # 6. Initialize Trainer & Execute
    # -----------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting Base Model Training...")
    trainer.train()

    # -----------------------------------------------------------------------
    # 7. Save Model Weights & Tokenizer Output
    # -----------------------------------------------------------------------
    print(f"Saving baseline weights to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("--- Stage 1 Base Training Complete! ---")

if __name__ == "__main__":
    main()
