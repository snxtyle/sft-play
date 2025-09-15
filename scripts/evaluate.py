import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import PeftModel
from torch.utils.data import Dataset
import json
from sklearn.model_selection import train_test_split
import math
import os

class QnADataset(Dataset):
    def __init__(self, tokenizer, data, block_size):
        self.examples = []
        for item in data:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            tokenized_output = tokenizer(text, truncation=True, max_length=block_size, padding="max_length")
            self.examples.append(tokenized_output)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = {key: torch.tensor(val) for key, val in self.examples[i].items()}
        item["labels"] = item["input_ids"].clone()
        return item

# --- Model Loading ---
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load base model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add and resize for padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load the fine-tuned LoRA adapter from the best checkpoint
# The main script saves the best model to "./fine-tuned-gpt-neo"
model = PeftModel.from_pretrained(model, "./fine-tuned-gpt-neo")
model = model.merge_and_unload() # Merge for evaluation

# --- Data Loading ---
with open("data/extracted_qna.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Replicate the same split to get the test set
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

test_dataset = QnADataset(
    tokenizer=tokenizer,
    data=test_data,
    block_size=128
)

# --- Evaluation ---
training_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=4,
    do_train=False,
    do_eval=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
)

print("--- Evaluating on Test Set ---")
eval_results = trainer.evaluate(eval_dataset=test_dataset)

# Save evaluation results
if not os.path.exists("./eval_results"):
    os.makedirs("./eval_results")
with open("./eval_results/eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=4)

print("\n--- Test Set Evaluation Summary ---")
test_loss = eval_results.get("eval_loss")
if test_loss:
    perplexity = math.exp(test_loss)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
else:
    print("Could not retrieve evaluation loss.")
print("------------------------------------")
print("\nEvaluation results saved to ./eval_results/eval_results.json")

# --- Qualitative Examples ---
print("\n--- Generating examples from the test set ---")
model.to("mps" if torch.backends.mps.is_available() else "cpu") # Move model to device for generation
for i in range(min(3, len(test_data))):
    prompt_text = f"Question: {test_data[i]['question']}\nAnswer:"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=120, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n✅ Example {i+1}:")
    print(generated_text)
    print(f"\n➡️ Actual Answer: {test_data[i]['answer']}")
    print("-" * 30)
