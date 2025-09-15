import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
import json

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
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "c_fc", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Get PEFT model
model = get_peft_model(model, config)

# Load data
def load_data(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

train_data = load_data("data/train.jsonl")
eval_data = load_data("data/validation.jsonl")

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

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=50,
    save_steps=50,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    warmup_steps=100,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

trainer.save_model("./fine-tuned-gpt-neo")
