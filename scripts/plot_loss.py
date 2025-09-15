import json
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(log_file):
    steps = []
    losses = []
    learning_rates = []
    eval_steps = []
    eval_losses = []

    with open(log_file, 'r') as f:
        log_data = json.load(f)
        for entry in log_data['log_history']:
            if 'loss' in entry:
                steps.append(entry['step'])
                losses.append(entry['loss'])
                if 'learning_rate' in entry:
                    learning_rates.append(entry['learning_rate'])
            if 'eval_loss' in entry:
                eval_steps.append(entry['step'])
                eval_losses.append(entry['eval_loss'])

    # Find the best validation point
    min_eval_loss = min(eval_losses)
    best_step = eval_steps[np.argmin(eval_losses)]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plotting training and validation loss
    ax1.plot(steps, losses, label='Training Loss', color='tab:blue', alpha=0.8)
    ax1.plot(eval_steps, eval_losses, label='Validation Loss', color='tab:orange', marker='o')
    ax1.axvline(x=best_step, color='r', linestyle='--', label=f'Best Model (Step {best_step})')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adding a second y-axis for the learning rate
    ax2 = ax1.twinx()
    ax2.plot(steps, learning_rates, label='Learning Rate', color='tab:green', linestyle=':')
    ax2.set_ylabel('Learning Rate', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Title and legend
    fig.suptitle('Training and Validation Loss with Learning Rate', fontsize=16)
    ax1.set_title(f'Min Validation Loss: {min_eval_loss:.4f} at Step {best_step}', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.savefig('loss_curve.png')
    print("✅ Detailed loss curve saved to loss_curve.png")

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
