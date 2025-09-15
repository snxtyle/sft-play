import json
import matplotlib.pyplot as plt

def plot_loss_curve(log_file):
    steps = []
    losses = []
    eval_steps = []
    eval_losses = []

    with open(log_file, 'r') as f:
        log_data = json.load(f)
        for entry in log_data['log_history']:
            if 'loss' in entry:
                steps.append(entry['step'])
                losses.append(entry['loss'])
            if 'eval_loss' in entry:
                eval_steps.append(entry['step'])
                eval_losses.append(entry['eval_loss'])

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss')
    plt.plot(eval_steps, eval_losses, label='Validation Loss', marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Loss curve saved to loss_curve.png")

if __name__ == '__main__':
    # The trainer state is saved in the last checkpoint directory
    # First, find the latest checkpoint
    import os
    checkpoints = [d for d in os.listdir('results') if d.startswith('checkpoint-')]
    checkpoints_with_state = [c for c in checkpoints if os.path.exists(os.path.join('results', c, 'trainer_state.json'))]
    if not checkpoints_with_state:
        print("No completed checkpoints with trainer_state.json found.")
    else:
        latest_checkpoint = sorted(checkpoints_with_state, key=lambda x: int(x.split('-')[1]))[-1]
        log_file = os.path.join('results', latest_checkpoint, 'trainer_state.json')
        plot_loss_curve(log_file)
