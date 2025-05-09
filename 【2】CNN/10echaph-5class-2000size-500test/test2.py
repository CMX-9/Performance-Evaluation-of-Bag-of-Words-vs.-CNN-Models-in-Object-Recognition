import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Subset
import logging
import sys

# ========================
# Experiment Configuration
# ========================
BATCH_SIZE = 64
EPOCHS = 10
BASE_LR = 0.1
MOMENTUM = 0.9
BASE_WD = 5e-4
LR_RANGE = [0.001, 0.01, 0.1, 0.5]
WD_RANGE = [0, 1e-5, 5e-4, 1e-2]
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ========================
# Data Loading
# ========================
def load_datasets():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)
    full_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True)

    # Select 1000 samples per class for 5 classes
    selected_classes = [0, 1, 2, 3, 4]
    train_indices = []
    for cls in selected_classes:
        indices = np.where(np.array(full_train.targets) == cls)[0]
        train_indices.extend(np.random.choice(indices, 2000, replace=False))
    
    test_mask = np.isin(full_test.targets, selected_classes)
    
    # Create subsets with transforms
    train_set = Subset(torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=train_transform), train_indices)
    test_set = Subset(torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=test_transform), np.where(test_mask)[0])

    return (
        torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=2),
        torch.utils.data.DataLoader(test_set, BATCH_SIZE*2, shuffle=False, num_workers=2)
    )

# ========================
# Neural Network
# ========================
class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ========================
# Training Components
# ========================
def train_model(model, optimizer, criterion, train_loader, scheduler=None):
    model.train()
    history = {'loss': [], 'acc': [], 'time': []}
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, unit="batch", leave=False) as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            for inputs, labels in pbar:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                })
        
        history['loss'].append(total_loss/len(train_loader))
        history['acc'].append(100.*correct/total)
        history['time'].append(time.time()-start_time)
        
        if scheduler:
            scheduler.step()
    
    return history

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)*100, confusion_matrix(all_labels, all_preds)

# ========================
# Visualization Functions
# ========================
def plot_hyperparameter_sensitivity(results_df, param_name):
    plt.figure(figsize=(10,6))
    plt.plot(results_df[param_name], results_df['test_acc'], 'o-', markersize=8)
    plt.xscale('log' if param_name == 'lr' else 'linear')
    plt.xlabel(param_name.upper(), fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'{param_name.upper()} Sensitivity Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{param_name}_sensitivity.png', bbox_inches='tight')
    plt.close()

def plot_training_curves(histories, param_values, param_name):
    plt.figure(figsize=(12,6))
    for value, hist in zip(param_values, histories):
        plt.plot(hist['loss'], label=f'{param_name}={value}')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title(f'Training Loss Comparison ({param_name})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'loss_curves_{param_name}.png', bbox_inches='tight')
    plt.close()

def plot_time_comparison(results_df):
    # Fix for negative values
    valid_times = results_df['total_time'].clip(lower=0.1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    
    # Time distribution
    avg_time = valid_times.mean() / EPOCHS
    ax1.pie([avg_time, max(0.1, 1-avg_time)], 
           labels=['Training', 'Other'], 
           autopct='%1.1f%%', 
           colors=['#66b3ff','#99ff99'])
    ax1.set_title('Time Distribution per Epoch')
    
    # Total time comparison
    results_df.plot(x='config', y='total_time', kind='bar', ax=ax2)
    ax2.set_title('Total Training Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('training_time_analysis.png', bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(10,8))
    
    # 创建带正确方向的混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                square=True,
                cbar_kws={"shrink": 0.8})
    
    # 调整坐标轴方向
    plt.gca().invert_yaxis()  # Y轴从上到下显示0-4
    plt.gca().xaxis.tick_top()  # X轴标签显示在顶部
    
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    
    # 调整标签位置
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0)
    
    plt.savefig('final_confusion_matrix.png', bbox_inches='tight')
    plt.close()

# ========================
# Main Experiment
# ========================
def main():
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load data
    train_loader, test_loader = load_datasets()
    
    results = []
    lr_histories = []
    wd_histories = []
    
    # LR experiment
    logging.info("\n=== Running Learning Rate Experiment ===")
    for lr in LR_RANGE:
        model = CompactCNN()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=BASE_WD)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        
        hist = train_model(model, opt, nn.CrossEntropyLoss(), train_loader, scheduler)
        acc, cm = evaluate_model(model, test_loader)
        
        results.append({
            'config': f'LR={lr}',
            'lr': lr,
            'wd': BASE_WD,
            'test_acc': acc,
            'total_time': sum(hist['time'])
        })
        lr_histories.append(hist)
    
    # WD experiment
    logging.info("\n=== Running Weight Decay Experiment ===")
    for wd in WD_RANGE:
        model = CompactCNN()
        opt = optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        
        hist = train_model(model, opt, nn.CrossEntropyLoss(), train_loader, scheduler)
        acc, cm = evaluate_model(model, test_loader)
        
        results.append({
            'config': f'WD={wd}',
            'lr': BASE_LR,
            'wd': wd,
            'test_acc': acc,
            'total_time': sum(hist['time'])
        })
        wd_histories.append(hist)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_results.csv', index=False)
    
    # Generate plots
    plot_hyperparameter_sensitivity(results_df[results_df['wd'] == BASE_WD], 'lr')
    plot_hyperparameter_sensitivity(results_df[results_df['lr'] == BASE_LR], 'wd')
    plot_training_curves(lr_histories, LR_RANGE, 'lr')
    plot_training_curves(wd_histories, WD_RANGE, 'wd')
    plot_time_comparison(results_df)
    plot_confusion_matrix(cm)
    
    # Final output
    logging.info("\nExperiment completed. Generated files:")
    logging.info("- lr_sensitivity.png        : Learning rate sensitivity")
    logging.info("- wd_sensitivity.png        : Weight decay sensitivity")
    logging.info("- loss_curves_lr.png        : Learning rate loss curves")
    logging.info("- loss_curves_wd.png        : Weight decay loss curves")
    logging.info("- training_time_analysis.png: Training time comparison")
    logging.info("- final_confusion_matrix.png: Final confusion matrix")
    logging.info("- experiment_results.csv    : Complete results data")

if __name__ == "__main__":
    main()