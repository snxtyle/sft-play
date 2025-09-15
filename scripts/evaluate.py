import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import PeftModel
from torch.utils.data import Dataset
import json
import math
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

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
with open("data/test.jsonl", "r") as f:
    test_data = [json.loads(line) for line in f]

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
eval_metrics = trainer.evaluate(eval_dataset=test_dataset)

# --- Qualitative Examples & BLEU Score ---
print("\n--- Generating examples and calculating BLEU scores ---")
model.to("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer.pad_token = tokenizer.eos_token # Suppress the warning
qualitative_results = []
total_bleu_score = 0
chencherry = SmoothingFunction()

for item in tqdm(test_data, desc="Generating and Scoring"):
    prompt_text = f"Question: {item['question']}\nAnswer:"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=120, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt_text, "").strip()
    
    reference = item['answer'].split()
    candidate = generated_text.split()
    bleu_score = sentence_bleu([reference], candidate, smoothing_function=chencherry.method1)
    total_bleu_score += bleu_score

    qualitative_results.append({
        "question": item['question'],
        "generated_answer": generated_text,
        "actual_answer": item['answer'],
        "bleu_score": bleu_score
    })

# --- Final Report ---
average_bleu = total_bleu_score / len(test_data)
test_loss = eval_metrics.get("eval_loss")
perplexity = math.exp(test_loss) if test_loss else None

report = {
    "quantitative_metrics": {
        "test_loss": test_loss,
        "perplexity": perplexity,
        "average_bleu_score": average_bleu
    },
    "qualitative_examples": qualitative_results
}

if not os.path.exists("./eval_results"):
    os.makedirs("./eval_results")
with open("./eval_results/eval_results.json", "w") as f:
    json.dump(report, f, indent=4)

print("\n--- Detailed Test Set Evaluation Summary ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")
print(f"Average BLEU Score: {average_bleu:.4f}")
print("---------------------------------------------")
print("\n✅ Detailed evaluation report saved to ./eval_results/eval_results.json")
