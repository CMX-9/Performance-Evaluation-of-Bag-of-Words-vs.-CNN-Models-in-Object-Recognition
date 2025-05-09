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
# Enhanced Model with SE Blocks and Residual Connections
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
    def __init__(self, block, layers, num_classes=2, dropout=0.5):  # 修改为2类
        super().__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
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
FULL_EPOCHS = 50
PATIENCE = 15
TARGET_CLASSES = [1, 8]  # automobile(1) 和 ship(8)
CLASS_NAMES = ['automobile', 'ship']
TRAIN_SAMPLES = 2000  # 总训练样本数（每类1000）
TEST_SAMPLES = 500    # 总测试样本数（每类250）

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
# Data Loading (仅保留类别1和8)
# ========================
class Cutout(object):
    def __init__(self, length=18):
        self.length = length

    def __call__(self, img):
        c, h, w = img.shape
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).float()
        mask = mask.unsqueeze(0).expand(c, -1, -1)
        return img * mask

def get_transforms(augment=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.15, 0.15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if augment:
        transform_train.transforms.append(Cutout(18))
    
    return transform_train, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def filter_classes(dataset):
    indices = []
    for idx, (_, label) in enumerate(dataset):
        if label in TARGET_CLASSES:
            mapped_label = 0 if label == TARGET_CLASSES[0] else 1
            # 控制样本数量
            if (dataset.train and len(indices) < TRAIN_SAMPLES) or \
               (not dataset.train and len(indices) < TEST_SAMPLES):
                indices.append((idx, mapped_label))
    return indices

class SubsetWithLabel(torch.utils.data.Dataset):
    def __init__(self, full_dataset, indices):
        self.subset = Subset(full_dataset, [i[0] for i in indices])  # 原始数据集已包含transform
        self.labels = [i[1] for i in indices]                         # 存储映射后的标签

    def __getitem__(self, index):
        img, _ = self.subset[index]  # 直接获取已应用transform的图像
        return img, self.labels[index]

    def __len__(self):
        return len(self.subset)

def load_datasets():
    transform_train, transform_test = get_transforms(augment=True)
    
    # 加载数据集时直接应用transform
    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train  # 应用训练增强
    )
    full_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test    # 应用测试增强
    )

    # 过滤并映射标签
    train_indices = filter_classes(full_train)
    test_indices = filter_classes(full_test)
    
    # 创建带标签映射的子集
    train_subset = SubsetWithLabel(full_train, train_indices)
    test_subset = SubsetWithLabel(full_test, test_indices)
    
    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4
    )
    
    return train_loader, test_loader

# ========================
# Training Components (优化早停和梯度裁剪)
# ========================
class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.005):  # 修改参数
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

def train_model(model, optimizer, criterion, train_loader, test_loader, scheduler=None, epochs=1):
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
    best_weights = model.state_dict()
    
    try:
        for epoch in range(epochs):
            epoch_monitor = ResourceMonitor()
            epoch_start = time.time()
            total_loss = 0.0
            correct = 0
            total = 0
            
            with tqdm(train_loader, unit="batch", leave=False) as pbar:
                pbar.set_description(f"Epoch {epoch+1}/{epochs}")
                for batch_idx, (inputs, labels) in enumerate(pbar):
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # 修改：增大梯度裁剪阈值
                    optimizer.step()
                    
                    if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler.step()
                    
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
            
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
                
            if early_stopper(history['test_acc'][-1]):
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                model.load_state_dict(best_weights)
                break
            if history['test_acc'][-1] > early_stopper.max_accuracy:
                best_weights = model.state_dict().copy()
                
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user!")
        return None
    
    return history

# ========================
# 可视化函数（保持原样）
# ========================
def plot_confusion_matrix(cm, class_names, filename):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', 
                cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_training_curves(full_history, sub_history):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(full_history['train_acc'], label='Full Train')
    plt.plot(full_history['test_acc'], label='Full Test')
    plt.plot(sub_history['train_acc'], label='Sub Train')
    plt.plot(sub_history['test_acc'], label='Sub Test')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(full_history['epoch_loss'], label='Full')
    plt.plot(sub_history['epoch_loss'], label='Sub')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(full_history['avg_cpu_usage'], label='Full CPU')
    plt.plot(full_history['avg_gpu_usage'], label='Full GPU')
    plt.plot(sub_history['avg_cpu_usage'], label='Sub CPU')
    plt.plot(sub_history['avg_gpu_usage'], label='Sub GPU')
    plt.title('Resource Usage')
    plt.xlabel('Epoch')
    plt.ylabel('Usage (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, class_names, top_k=(1,3,5), filename_prefix=''):
    model.eval()
    all_labels = []
    all_outputs = []
    num_classes = len(class_names)
    
    # Filter out invalid top_k values and ensure at least top1 is computed
    valid_top_k = [k for k in top_k if 1 <= k <= num_classes]
    if not valid_top_k:
        valid_top_k = [1]
        logging.warning("No valid k in top_k, defaulting to top1")
    max_k = max(valid_top_k)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.append(outputs.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    _, preds = torch.max(all_outputs, 1)
    
    cm = confusion_matrix(all_labels, preds.numpy())
    plot_confusion_matrix(cm, class_names, f'{filename_prefix}_confusion_matrix.png')
    
    # Compute top-k accuracy for valid k values
    _, topk_preds = torch.topk(all_outputs, max_k, dim=1)
    topk_acc = {}
    
    for k in valid_top_k:
        correct = sum(1 for i in range(len(all_labels)) if all_labels[i] in topk_preds[i, :k].numpy())
        topk_acc[f'top_{k}_acc'] = 100. * correct / len(all_labels)
    
    return accuracy_score(all_labels, preds.numpy()) * 100, cm, topk_acc

def main():
    try:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.benchmark = True
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        logging.info("\n=== Training with 2-class Dataset ===")
        train_loader, test_loader = load_datasets()
        model = SEResNet(Bottleneck, [3,4,6,3], num_classes=2).to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FULL_EPOCHS)
        history = train_model(model, opt, nn.CrossEntropyLoss(),
                            train_loader, test_loader,
                            scheduler=scheduler, epochs=FULL_EPOCHS)
        
        acc, cm, topk = evaluate_model(model, test_loader, CLASS_NAMES, filename_prefix='2class')
        pd.DataFrame(history).to_csv('2class_training.csv', index=False)
        logging.info(f"Final Test Accuracy: {acc:.2f}%")

    except Exception as e:
        logging.error(f"Execution failed: {str(e)}")
        raise
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()