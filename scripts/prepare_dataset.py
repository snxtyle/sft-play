import json
from sklearn.model_selection import train_test_split

def prepare_dataset():
    # Load the original dataset
    with open("data/qa_dataset.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    # Split data into training (80%), validation (10%), and test (10%)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Save each split to a new file
    def save_to_jsonl(data, filename):
        with open(f"data/{filename}", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    save_to_jsonl(train_data, "train.jsonl")
    save_to_jsonl(eval_data, "validation.jsonl")
    save_to_jsonl(test_data, "test.jsonl")

    print("✅ Dataset successfully split and saved into:")
    print("- data/train.jsonl")
    print("- data/validation.jsonl")
    print("- data/test.jsonl")

if __name__ == "__main__":
    prepare_dataset()
