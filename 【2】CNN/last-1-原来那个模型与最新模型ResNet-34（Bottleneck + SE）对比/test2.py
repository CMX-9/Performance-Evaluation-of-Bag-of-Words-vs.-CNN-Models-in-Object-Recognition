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
import logging
import sys
import psutil
import GPUtil
import warnings
import gc
from torch.utils.data import Subset
from torchvision.models.resnet import ResNet, Bottleneck

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
# 在模型定义之后添加设备定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# Common Components
# ========================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (Added for improved model)"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResourceMonitor:
    """Shared resource monitoring class"""
    def __init__(self):
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_mem = []
        
    def update(self):
        self.cpu_usage.append(psutil.cpu_percent())
        try:
            gpu = GPUtil.getGPUs()[0]
            self.gpu_usage.append(gpu.load*100)
            self.gpu_mem.append(gpu.memoryUsed)
        except:
            self.gpu_usage.append(0)
            self.gpu_mem.append(0)

class EarlyStopper:
    """Early stopping mechanism"""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_accuracy = 0

    def __call__(self, val_accuracy):
        if val_accuracy > self.max_accuracy + self.min_delta:
            self.max_accuracy = val_accuracy
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# ========================
# Model Definitions
# ========================
class CIFARNet(nn.Module):
    """Original model architecture"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class SEResNet(ResNet):
    """Improved model with SE blocks"""
    def __init__(self, block, layers, num_classes=10, dropout=0.5):
        super().__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
        # Add SE blocks
        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x); x = self.se1(x)
        x = self.layer2(x); x = self.se2(x)
        x = self.layer3(x); x = self.se3(x)
        x = self.layer4(x); x = self.se4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# ========================
# Enhanced Data Loading
# ========================
def load_datasets(improved=False):
    """Load datasets with different configs for original/improved models"""
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_transform = base_transform if not improved else transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=base_transform)

    # Dataset sizing
    train_size = 5000 if improved else 2000
    test_size = 1000 if improved else 500
    
    train_subset = Subset(train_set, range(train_size))
    test_subset = Subset(test_set, range(test_size))

    # DataLoader config
    num_workers = 4 if improved else 0
    batch_size = 256 if improved else 128

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=improved
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=improved
    )
    
    return train_loader, test_loader

# ========================
# Training Components
# ========================
def train_model(model, optimizer, criterion, train_loader, test_loader, config):
    """Generic training function for both models"""
    model.train()
    history = {
        'epoch_loss': [], 'train_acc': [], 'test_acc': [],
        'epoch_time': [], 'avg_cpu_usage': [], 
        'avg_gpu_usage': [], 'avg_gpu_mem': []
    }
    
    early_stopper = EarlyStopper(patience=config.get('patience',5))
    best_weights = model.state_dict()

    for epoch in range(config['epochs']):
        epoch_monitor = ResourceMonitor()
        epoch_start = time.time()
        total_loss, correct, total = 0.0, 0, 0
        
        # Training phase
        with tqdm(train_loader, unit="batch", leave=False) as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{config['epochs']}")
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Metrics calculation
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:
                    epoch_monitor.update()
                
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Validation phase
        test_acc = evaluate_model(model, test_loader, return_acc_only=True)
        
        # Record metrics
        history['epoch_loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(100. * correct / total)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(time.time() - epoch_start)
        history['avg_cpu_usage'].append(np.mean(epoch_monitor.cpu_usage))
        history['avg_gpu_usage'].append(np.mean(epoch_monitor.gpu_usage))
        history['avg_gpu_mem'].append(np.mean(epoch_monitor.gpu_mem))
        
        # Learning rate scheduling
        if 'scheduler' in config:
            config['scheduler'].step()
        
        # Early stopping
        if early_stopper(test_acc):
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_weights)
            break
        if test_acc > early_stopper.max_accuracy:
            best_weights = model.state_dict().copy()
            
    return history, model

def evaluate_model(model, test_loader, return_acc_only=False, top_k=(1,3,5)):
    """Enhanced evaluation with Top-K accuracy"""
    model.eval()
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.append(outputs.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    _, preds = torch.max(all_outputs, 1)
    acc = accuracy_score(all_labels, preds.numpy()) * 100
    
    if return_acc_only:
        return acc
    
    # Calculate Top-K accuracy
    max_k = max(top_k)
    _, topk_preds = torch.topk(all_outputs, max_k, dim=1)
    topk_acc = {
    f'top_{k}_acc': 100.0 * sum(label in pred[:k] for label, pred in zip(all_labels, topk_preds.numpy())) / len(all_labels)
    for k in top_k}
    
    return acc, confusion_matrix(all_labels, preds.numpy()), topk_acc

# ========================
# Visualization Functions
# ========================
def plot_comparison(original, improved, ylabel, title, legends=('Original', 'Improved')):
    plt.figure(figsize=(10,6))
    plt.plot(original, '--', label=legends[0])
    plt.plot(improved, '-', linewidth=2, label=legends[1])
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_full_comparison(orig_history, imp_history):
    """Generate all comparison plots"""
    # Accuracy Comparison
    plt = plot_comparison(orig_history['test_acc'], imp_history['test_acc'], 
                        'Accuracy (%)', 'Test Accuracy Comparison')
    plt.savefig('accuracy_comparison.png', bbox_inches='tight')
    plt.close()
    
    # Loss Comparison
    plt = plot_comparison(orig_history['epoch_loss'], imp_history['epoch_loss'],
                        'Loss', 'Training Loss Comparison')
    plt.savefig('loss_comparison.png', bbox_inches='tight')
    plt.close()
    
    # Resource Comparison
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    metrics = ['avg_cpu_usage', 'avg_gpu_usage', 'avg_gpu_mem']
    titles = ['CPU Utilization (%)', 'GPU Utilization (%)', 'GPU Memory (MB)']
    
    for ax, metric, title in zip(axs, metrics, titles):
        ax.plot(orig_history[metric], '--', label='Original')
        ax.plot(imp_history[metric], '-', label='Improved')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('resource_comparison.png', bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_comparison(orig_cm, imp_cm, class_names):
    """Side-by-side confusion matrix comparison"""
    plt.figure(figsize=(24, 10))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(orig_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Original Model Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(imp_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Improved Model Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('cm_comparison.png', bbox_inches='tight')
    plt.close()

# ========================
# Main Workflow
# ========================
def main():
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    global device
    print(f"Initializing on {device}")
    # Environment setup
    torch.manual_seed(42)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
    )
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Train Original Model
    orig_train, orig_test = load_datasets(improved=False)
    orig_model = CIFARNet().to(device)
    orig_opt = optim.SGD(orig_model.parameters(), lr=0.1, momentum=0.9)
    
    logging.info("\n=== Training Original Model ===")
    orig_history, orig_model = train_model(
        orig_model, orig_opt, nn.CrossEntropyLoss(),
        orig_train, orig_test,
        config={'epochs': 30, 'patience': 5}
    )
    orig_acc, orig_cm, orig_topk = evaluate_model(orig_model, orig_test)

    # Train Improved Model
    imp_train, imp_test = load_datasets(improved=True)
    imp_model = SEResNet(Bottleneck, [2,2,2,2]).to(device)
    imp_opt = optim.SGD(imp_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    imp_scheduler = optim.lr_scheduler.CosineAnnealingLR(imp_opt, T_max=50)
    
    logging.info("\n=== Training Improved Model ===")
    imp_history, imp_model = train_model(
        imp_model, imp_opt, nn.CrossEntropyLoss(),
        imp_train, imp_test,
        config={'epochs': 50, 'patience': 5, 'scheduler': imp_scheduler}
    )
    imp_acc, imp_cm, imp_topk = evaluate_model(imp_model, imp_test)

    # Generate all visualizations
    plot_full_comparison(orig_history, imp_history)
    plot_confusion_matrix_comparison(orig_cm, imp_cm, CLASS_NAMES)
    
    # Save results
    pd.DataFrame(orig_history).to_csv('original_results.csv')
    pd.DataFrame(imp_history).to_csv('improved_results.csv')
    
    # Print final report
    logging.info("\n=== Final Report ===")
    logging.info(f"Original Model | Test Acc: {orig_acc:.2f}% | Top-3 Acc: {orig_topk['top_3_acc']:.2f}%")
    logging.info(f"Improved Model | Test Acc: {imp_acc:.2f}% | Top-3 Acc: {imp_topk['top_3_acc']:.2f}%")
    logging.info(f"Accuracy Improvement: {imp_acc-orig_acc:.2f}%")

if __name__ == "__main__":
    main()