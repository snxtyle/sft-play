# Makefile for the GPT-Neo Fine-Tuning Project

# Define the Python interpreter
PYTHON = python3

# --- Main Targets ---

all: prepare-dataset train evaluate plot
	@echo "✅ Full pipeline completed successfully!"

prepare-dataset:
	@echo "--- splitting dataset ---"
	$(PYTHON) scripts/prepare_dataset.py

train:
	@echo "--- Starting model fine-tuning ---"
	$(PYTHON) scripts/fine_tune.py

evaluate:
	@echo "--- Evaluating the fine-tuned model ---"
	$(PYTHON) scripts/evaluate.py

plot:
	@echo "--- Generating loss curve plot ---"
	$(PYTHON) scripts/plot_loss.py

inference:
	@echo "--- Running a sample inference ---"
	$(PYTHON) scripts/inference.py

# --- Utility Targets ---

clean:
	@echo "--- Cleaning up generated files ---"
	rm -rf results/
	rm -rf eval_results/
	rm -f loss_curve.png
	rm -f data/train.jsonl data/validation.jsonl data/test.jsonl
	@echo "✅ Cleanup complete."

.PHONY: all prepare-dataset train evaluate plot inference clean
