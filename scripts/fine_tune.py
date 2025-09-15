import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
import json

from sklearn.model_selection import train_test_split

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

model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name
)

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Get PEFT model
model = get_peft_model(model, config)

# Load data
with open("data/extracted_qna.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Split data into training (80%), validation (10%), and test (10%)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_dataset = QnADataset(
    tokenizer=tokenizer,
    data=train_data,
    block_size=128
)

eval_dataset = QnADataset(
    tokenizer=tokenizer,
    data=eval_data,
    block_size=128
)

test_dataset = QnADataset(
    tokenizer=tokenizer,
    data=test_data,
    block_size=128
)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    logging_steps=100,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True, # To prevent overfitting
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

trainer.save_model("./fine-tuned-gpt-neo")
