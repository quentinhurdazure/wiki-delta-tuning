import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ---------------------------------------------------------------------------
# 1. Configuration & Paths
# ---------------------------------------------------------------------------
BASE_MODEL_DIR: str = "./models/base_wiki_model"
LORA_OUTPUT_DIR: str = "./models/lora_wiki_delta"

def main() -> None:
    print(f"--- Initializing Lab Stage 2: Delta Tuning via LoRA ---")
    print(f"Loading frozen base weights from: {BASE_MODEL_DIR}")

    # -----------------------------------------------------------------------
    # 2. Load Local Baseline Model & Tokenizer
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    # -----------------------------------------------------------------------
    # 3. Configure Parameter-Efficient Fine-Tuning (LoRA)
    # -----------------------------------------------------------------------
    # We target the attention projection layers (q_proj, v_proj).
    # Rank (r=8) keeps adapter footprint under 10 MB!
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )

    # Inject LoRA layers and freeze base model parameters
    model = get_peft_model(base_model, peft_config)
    
    print("\n--- Trainable Parameter Breakdown ---")
    model.print_trainable_parameters()
    print("-------------------------------------\n")

    # -----------------------------------------------------------------------
    # 4. Simulated Updated Wikipedia Data (Data Shift Delta)
    # -----------------------------------------------------------------------
    updated_wiki_data = {
        "text": [
            "Topic: Microsoft Copilot Studio. Text: Copilot Studio allows organizations to build custom AI agents with enterprise data sources. Keywords: Copilot Studio, agents, customization",
            "Topic: Quantum Supremacy. Text: Quantum supremacy marks the point where quantum computers outperform classical supercomputers. Keywords: quantum supremacy, computing, benchmarks"
        ]
    }
    updated_dataset = Dataset.from_dict(updated_wiki_data)

    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    tokenized_dataset = updated_dataset.map(tokenize_function, batched=True)

    # -----------------------------------------------------------------------
    # 5. Training Arguments for LoRA Delta Update
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        num_train_epochs=5,              # Slightly higher epochs for tiny adapter weights
        per_device_train_batch_size=1,
        logging_steps=1,
        save_strategy="no",
        learning_rate=3e-4,              # Higher learning rate standard for LoRA adapters
        weight_decay=0.01,
        use_cpu=not torch.cuda.is_available()
    )

    # -----------------------------------------------------------------------
    # 6. Execute Trainer on Adapter Weights Only
    # -----------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting LoRA Delta Training...")
    trainer.train()

    # -----------------------------------------------------------------------
    # 7. Save Delta Adapter Weights
    # -----------------------------------------------------------------------
    print(f"Saving lightweight LoRA adapters to {LORA_OUTPUT_DIR}...")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print("--- Stage 2 LoRA Delta Training Complete! ---")

if __name__ == "__main__":
    main()
