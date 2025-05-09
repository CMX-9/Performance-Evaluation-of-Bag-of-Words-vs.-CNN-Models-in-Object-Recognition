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

# Suppress known warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# ========================
# Improved Model Definition with SE Blocks
# ========================
class SEBlock(nn.Module):
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

class SEResNet(ResNet):
    def __init__(self, block, layers, num_classes=10, dropout=0.5):
        super().__init__(block, layers, num_classes=num_classes)
        # Adjust for CIFAR-10 32x32 input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.Identity()  # Remove initial maxpool
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
        
        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.layer4(x)
        x = self.se4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ========================
# Experiment Configuration
# ========================
BATCH_SIZE = 256
EPOCHS = 50  #50
PATIENCE = 5
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_subset = Subset(train_set, range(5000))
    test_subset = Subset(test_set, range(1000))

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader

# ========================
# Training Components
# ========================
class EarlyStopper:
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

class ResourceMonitor:
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

def train_model(model, optimizer, criterion, train_loader, test_loader, scheduler=None):
    model.train()
    history = {
        'epoch_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch_time': [],
        'avg_cpu_usage': [],
        'avg_gpu_usage': [],
        'avg_gpu_mem': []
    }
    
    early_stopper = EarlyStopper(patience=PATIENCE)
    best_weights = model.state_dict()  # Initialize with initial weights
    
    try:
        for epoch in range(EPOCHS):
            epoch_monitor = ResourceMonitor()
            epoch_start = time.time()
            total_loss = 0.0
            correct = 0
            total = 0
            
            with tqdm(train_loader, unit="batch", leave=False) as pbar:
                pbar.set_description(f"Epoch {epoch+1}/{EPOCHS}")
                for batch_idx, (inputs, labels) in enumerate(pbar):
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if batch_idx % 10 == 0:
                        epoch_monitor.update()
                    
                    if batch_idx % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            history['epoch_loss'].append(total_loss / len(train_loader))
            history['train_acc'].append(100. * correct / total)
            history['test_acc'].append(100. * test_correct / test_total)
            history['epoch_time'].append(time.time() - epoch_start)
            history['avg_cpu_usage'].append(np.mean(epoch_monitor.cpu_usage))
            history['avg_gpu_usage'].append(np.mean(epoch_monitor.gpu_usage))
            history['avg_gpu_mem'].append(np.mean(epoch_monitor.gpu_mem))
            
            if scheduler:
                scheduler.step()
                
            if early_stopper(history['test_acc'][-1]):
                print(f"Early stopping triggered at epoch {epoch+1}")
                model.load_state_dict(best_weights)  # Now guaranteed to have weights
                break
            if history['test_acc'][-1] > early_stopper.max_accuracy:
                best_weights = model.state_dict().copy()  # Use explicit copy
                
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user!")
        return None
    
    return history

# ========================
# Evaluation & Visualization
# ========================
def evaluate_model(model, test_loader, top_k=(1,3,5)):
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
    
    # Calculate Top-K accuracy
    max_k = max(top_k)
    _, topk_preds = torch.topk(all_outputs, max_k, dim=1)
    topk_acc = {}
    
    for k in top_k:
        correct = sum(any(label in pred for pred in topk_preds[:,:k].numpy()) 
            for label, pred in zip(all_labels, topk_preds[:,:k].numpy()))
        topk_acc[f'top_{k}_acc'] = 100. * correct / len(all_labels)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, preds.numpy())
    return accuracy_score(all_labels, preds.numpy()) * 100, cm, topk_acc

def plot_confusion_matrix_comparison(original_cm, improved_cm, class_names=CLASS_NAMES):
    plt.figure(figsize=(24, 10))
    
    # Original Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(original_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Original Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Improved Confusion Matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(improved_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Improved Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png', bbox_inches='tight')
    plt.close()

def plot_performance_comparison(original_history, improved_history):
    plt.figure(figsize=(18, 12))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    plt.plot(original_history['test_acc'], label='Original Test', linestyle='--')
    plt.plot(improved_history['test_acc'], label='Improved Test', linewidth=2)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Loss comparison
    plt.subplot(2, 2, 2)
    plt.plot(original_history['epoch_loss'], label='Original Loss', linestyle='--')
    plt.plot(improved_history['epoch_loss'], label='Improved Loss', linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Resource comparison
    plt.subplot(2, 2, 3)
    plt.plot(original_history['avg_gpu_usage'], label='Original GPU Util', linestyle='--')
    plt.plot(improved_history['avg_gpu_usage'], label='Improved GPU Util', linewidth=2)
    plt.title('GPU Utilization Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Utilization (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', bbox_inches='tight')
    plt.close()

# ========================
# Main Workflow
# ========================
def main():
    try:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.benchmark = True
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        train_loader, test_loader = load_datasets()
        
        # Original ResNet Model
        original_model = torchvision.models.resnet18(num_classes=10).to(device)
        original_opt = optim.SGD(original_model.parameters(), lr=0.1, momentum=0.9)
        original_scheduler = optim.lr_scheduler.MultiStepLR(original_opt, milestones=[30, 45], gamma=0.1)
        
        # Train original model
        logging.info("\n=== Training Original Model ===")
        original_history = train_model(original_model, original_opt, 
                                     nn.CrossEntropyLoss(), train_loader, test_loader,
                                     original_scheduler)
        original_acc, original_cm, original_topk = evaluate_model(original_model, test_loader)
        
        # Improved SEResNet Model
        improved_model = SEResNet(Bottleneck, [2, 2, 2, 2], num_classes=10).to(device)
        improved_opt = optim.SGD(improved_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
        improved_scheduler = optim.lr_scheduler.CosineAnnealingLR(improved_opt, EPOCHS)
        
        # Train improved model
        logging.info("\n=== Training Improved Model ===")
        improved_history = train_model(improved_model, improved_opt,
                                     nn.CrossEntropyLoss(), train_loader, test_loader,
                                     improved_scheduler)
        improved_acc, improved_cm, improved_topk = evaluate_model(improved_model, test_loader)
        
        # Generate comparisons
        plot_performance_comparison(original_history, improved_history)
        plot_confusion_matrix_comparison(original_cm, improved_cm)
        
        # Save results
        pd.DataFrame(original_history).to_csv('original_training.csv', index=False)
        pd.DataFrame(improved_history).to_csv('improved_training.csv', index=False)
        
        # Print Top-K results
        logging.info("\n=== Top-K Accuracy Comparison ===")
        for k in original_topk:
            logging.info(f"Top-{k.split('_')[1]} Accuracy:")
            logging.info(f"  Original: {original_topk[k]:.2f}%")
            logging.info(f"  Improved: {improved_topk[k]:.2f}%")

        logging.info("\n=== Final Results ===")
        logging.info(f"Original Model Test Accuracy: {original_acc:.2f}%")
        logging.info(f"Improved Model Test Accuracy: {improved_acc:.2f}%")

    except Exception as e:
        logging.error(f"Execution failed: {str(e)}")
        raise
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()