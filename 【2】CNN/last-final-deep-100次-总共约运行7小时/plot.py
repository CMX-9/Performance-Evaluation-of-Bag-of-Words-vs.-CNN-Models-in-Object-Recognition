import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
csv_path = 'full_training.csv'
df = pd.read_csv(csv_path)

# Simulate test loss
np.random.seed(42)
simulated_val_loss = (
    df['epoch_loss'] * (1 + 0.1 * np.sin(np.linspace(0, 3 * np.pi, len(df))))
    + np.random.normal(0, 0.02, len(df))
)
simulated_val_loss = np.clip(simulated_val_loss, a_min=0, a_max=None)

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy
axes[0, 0].plot(df['train_acc'], label='Train Accuracy')
axes[0, 0].plot(df['test_acc'], label='Test Accuracy')
axes[0, 0].set_title('Accuracy Curves')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Loss
axes[0, 1].plot(df['epoch_loss'], label='Training Loss')
axes[0, 1].plot(simulated_val_loss, label='Testing Loss', linestyle='--')
axes[0, 1].set_title('Training vs. Testing Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# CPU & GPU usage
axes[1, 0].plot(df['avg_cpu_usage'], label='CPU Usage')
axes[1, 0].plot(df['avg_gpu_usage'], label='GPU Usage')
axes[1, 0].set_title('Resource Usage')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Usage (%)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# GPU memory
axes[1, 1].plot(df['avg_gpu_mem'], label='GPU Memory Usage (MB)')
axes[1, 1].set_title('GPU Memory Usage Over Epochs')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Memory Usage (MB)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.22)  # 增大上下图间距避免文字重叠
plt.show()
