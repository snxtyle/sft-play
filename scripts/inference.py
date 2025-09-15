import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Add and resize for padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load the fine-tuned LoRA adapter
model = PeftModel.from_pretrained(model, "./fine-tuned-gpt-neo")

# Merge the adapter with the base model
model = model.merge_and_unload()

# Set up the prompt
question = "What is the workflow of the ...?"
prompt = f"Question: {question}\nAnswer:"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate the output
outputs = model.generate(**inputs, max_new_tokens=1000, eos_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
