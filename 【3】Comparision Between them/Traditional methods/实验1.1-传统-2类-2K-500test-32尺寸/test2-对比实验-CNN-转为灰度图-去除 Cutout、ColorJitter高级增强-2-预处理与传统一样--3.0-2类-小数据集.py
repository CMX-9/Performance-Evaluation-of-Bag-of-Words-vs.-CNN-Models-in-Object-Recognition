# ========================
# Enhanced Model with SE Blocks and Residual Connections (User Configurable Version)
# ========================
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score
import seaborn as sns
from tqdm import tqdm
import logging
import sys
import psutil
import GPUtil
import warnings
import gc
import json
from torch.utils.data import Subset
from torchvision.models.resnet import ResNet, Bottleneck
from PIL import Image
import cv2
from tensorflow.keras.datasets import cifar10

# Suppress known warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# ======================== 用户配置区域 ========================
# 可配置参数：选择需要的类别索引（0-9）
SELECTED_CLASSES = [1, 8]  # 示例：汽车 vs 船舶分类
# 每个类别的训练样本量
TRAIN_SAMPLES_PER_CLASS = 1000  # 总训练样本量 = 类别数 × 该值
# 每个类别的测试样本量 
TEST_SAMPLES_PER_CLASS = 250    # 总测试样本量 = 类别数 × 该值
# ======================== 配置结束 ========================

CLASS_NAMES_FULL = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_NAMES = [CLASS_NAMES_FULL[i] for i in SELECTED_CLASSES]
NUM_CLASSES = len(SELECTED_CLASSES)

# ======================== 数据预处理（保持不变）=======================
class HistEqualize:
    def __call__(self, img):
        img_np = np.array(img)
        if img_np.ndim == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        img_eq = cv2.equalizeHist(img_gray)
        return Image.fromarray(np.stack([img_eq]*3, axis=2)) if img_np.ndim == 3 else Image.fromarray(img_eq)

class GaussianBlur:
    def __init__(self, kernel_size=5, sigma=1.5):
        self.kernel_size = (kernel_size, kernel_size)
        self.sigma = sigma
        
    def __call__(self, img):
        img_np = np.array(img)
        img_blur = cv2.GaussianBlur(img_np, self.kernel_size, self.sigma)
        return Image.fromarray(img_blur)

def get_transforms():
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda x: x.convert("L")),
        HistEqualize(),
        GaussianBlur(),
        transforms.RandomAffine(0, translate=(0.15, 0.15)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,), (0.2023,))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda x: x.convert("L")),
        HistEqualize(),
        GaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,), (0.2023,))
    ])
    return transform_train, transform_test

# ======================== 数据加载（修改部分）=======================
def load_datasets():
    # 加载完整数据集
    (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()
    
    # 创建类别掩码
    train_mask = np.isin(y_train_full.ravel(), SELECTED_CLASSES)
    test_mask = np.isin(y_test_full.ravel(), SELECTED_CLASSES)
    
    # 应用掩码
    x_train = x_train_full[train_mask]
    y_train = y_train_full[train_mask]
    x_test = x_test_full[test_mask]
    y_test = y_test_full[test_mask]
    
    # 重新映射标签到连续区间
    label_mapping = {orig: idx for idx, orig in enumerate(SELECTED_CLASSES)}
    y_train = np.vectorize(label_mapping.get)(y_train.ravel())
    y_test = np.vectorize(label_mapping.get)(y_test.ravel())
    
    # 平衡采样
    def balanced_subsample(data, labels, samples_per_class):
        indices = []
        for class_idx in range(len(SELECTED_CLASSES)):
            class_indices = np.where(labels == class_idx)[0]
            selected = np.random.choice(class_indices, samples_per_class, replace=False)
            indices.extend(selected)
        return data[indices], labels[indices]
    
    # 二次采样
    x_train, y_train = balanced_subsample(x_train, y_train, TRAIN_SAMPLES_PER_CLASS)
    x_test, y_test = balanced_subsample(x_test, y_test, TEST_SAMPLES_PER_CLASS)
    
    # 应用数据增强
    transform_train, transform_test = get_transforms()
    
    def apply_transforms(images, transform):
        return np.array([
            transform(Image.fromarray(img)) for img in images
        ])
    
    return (
        apply_transforms(x_train, transform_train),
        y_train,
        apply_transforms(x_test, transform_test),
        y_test
    )

# ======================== 模型定义（保持不变）=======================
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
        # 修改输入通道数为1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
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

# ======================== 训练组件（保持不变）=======================
BATCH_SIZE = 256
PATIENCE = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.005):
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
        self.ram_usage = []
        
    def update(self):
        self.ram_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent())
        try:
            gpu = GPUtil.getGPUs()[0]
            self.gpu_usage.append(gpu.load*100)
            self.gpu_mem.append(gpu.memoryUsed)
        except:
            self.gpu_usage.append(0)
            self.gpu_mem.append(0)

# ======================== 模型评估（保持不变）=======================
def evaluate_model(model, test_loader, class_names, top_k=(1,3,5), filename_prefix=''):
    model.eval()
    all_labels = []
    all_outputs = []
    
    def measure_inference_time():
        model.eval()
        random_test_sample = next(iter(test_loader))[0][0].unsqueeze(0).to(device)
        for _ in range(10):
            _ = model(random_test_sample)
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model(random_test_sample)
        torch.cuda.synchronize()
        return (time.time() - start_time) * 1000 / 100
    
    inference_time = measure_inference_time()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.append(outputs.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    y_true = np.array(all_labels)
    y_score = all_outputs.numpy()
    
    # 新增：动态过滤有效k值
    valid_ks = [k for k in top_k if k <= len(class_names)]
    if not valid_ks:  # 至少保留top1
        valid_ks = [1]
    
    # 新增：添加labels参数
    labels = np.arange(len(class_names))
    topk_acc = {
        f'top_{k}_acc': top_k_accuracy_score(
            y_true, 
            y_score, 
            k=k, 
            labels=labels
        ) * 100 for k in valid_ks
    }

# ======================== 训练流程（保持不变）=======================
def train_model(model, train_loader, test_loader, epochs=100):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'top3_acc': [],
        'top5_acc': [],
        'epoch_time': [],
        'cpu_usage': [],
        'gpu_usage': [],
        'ram_usage': []
    }
    
    early_stopper = EarlyStopper(patience=PATIENCE)
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_monitor = ResourceMonitor()
        start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (len(history['epoch'])+1) % 10 == 0:
                epoch_monitor.update()
        
        eval_report = evaluate_model(model, test_loader, CLASS_NAMES)
        
        history['epoch'].append(epoch+1)
        history['train_loss'].append(total_loss/len(train_loader))
        history['train_acc'].append(100.*correct/total)
        history['test_acc'].append(eval_report['accuracy'])
        history['top3_acc'].append(eval_report['topk_acc']['top_3_acc'])
        history['top5_acc'].append(eval_report['topk_acc']['top_5_acc'])
        history['epoch_time'].append(time.time()-start_time)
        history['cpu_usage'].append(np.mean(epoch_monitor.cpu_usage))
        history['gpu_usage'].append(np.mean(epoch_monitor.gpu_usage))
        history['ram_usage'].append(np.mean(epoch_monitor.ram_usage))
        
        if early_stopper(eval_report['accuracy']):
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
        if eval_report['accuracy'] > best_acc:
            best_acc = eval_report['accuracy']
            torch.save(model.state_dict(), 'best_model.pth')

        logging.info(
            f"Epoch {epoch+1} | "
            f"Train Acc: {history['train_acc'][-1]:.2f}% | "
            f"Test Acc: {history['test_acc'][-1]:.2f}% | "
            f"Top-3 Acc: {history['top3_acc'][-1]:.2f}%"
        )
    
    return history

# ======================== 主函数（保持不变）=======================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        # 加载数据
        x_train, y_train, x_test, y_test = load_datasets()
        # 删除permute操作
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).long()
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_test).long()
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE*2,
            shuffle=False,
            num_workers=4
        )
        
        # 初始化模型
        model = SEResNet(Bottleneck, [3, 4, 6, 3], num_classes=NUM_CLASSES).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Model Parameters: {total_params/1e6:.2f}M")
        
        # 训练模型
        history = train_model(model, train_loader, test_loader)
        
        # 生成报告
        final_report = {
            'classes': SELECTED_CLASSES,
            'class_names': CLASS_NAMES,
            'train_samples_per_class': TRAIN_SAMPLES_PER_CLASS,
            'test_samples_per_class': TEST_SAMPLES_PER_CLASS,
            'best_accuracy': max(history['test_acc']),
            'best_top3': max(history['top3_acc']),
            'best_top5': max(history['top5_acc']),
            'total_training_time': sum(history['epoch_time']),
            'avg_inference_time': np.mean([h['inference_time'] for h in history]),
            'resource_usage': {
                'cpu': np.mean(history['cpu_usage']),
                'gpu': np.mean(history['gpu_usage']),
                'ram': np.mean(history['ram_usage'])
            }
        }
        
        pd.DataFrame(history).to_csv('training_history.csv', index=False)
        with open('final_report.json', 'w') as f:
            json.dump(final_report, f, indent=2)
            
        logging.info("\n=== Training Completed ===")
        logging.info(json.dumps(final_report, indent=2))

    except Exception as e:
        logging.error(f"Execution failed: {str(e)}")

if __name__ == "__main__":
    main()