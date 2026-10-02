import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = "./models/base_wiki_model"
LORA_DIR = "./models/lora_wiki_delta"

def generate_text(model, tokenizer, prompt_text: str) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip out the prompt text to isolate generated response
    return full_output[len(prompt_text):].strip()

def main():
    print("--- Loading Stage 1 Base Model ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    base_model.eval()

    print("--- Loading Stage 2 LoRA Delta Model ---")
    lora_base = AutoModelForCausalLM.from_pretrained(
        BASE_DIR,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    lora_model = PeftModel.from_pretrained(lora_base, LORA_DIR)
    lora_model.eval()

    print("\n=======================================================")
    print("      LOCAL DELTA-TUNING LAB: INTERACTIVE EVAL         ")
    print("=======================================================\n")
    print("Type a prompt (e.g., 'Topic: Microsoft Copilot Studio.')")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            prompt = input("Enter Prompt > ")
            if prompt.strip().lower() in ["exit", "quit"]:
                break
            
            if not prompt.strip():
                continue

            print("\n" + "-"*50)
            base_res = generate_text(base_model, tokenizer, prompt)
            print(f"[BASE MODEL OUTPUT]:\n{base_res}\n")
            
            lora_res = generate_text(lora_model, tokenizer, prompt)
            print(f"[LORA DELTA OUTPUT]:\n{lora_res}")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
