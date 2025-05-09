# ========================
# Enhanced Model with SE Blocks and Residual Connections (Multi-Task Grayscale Version)
# ========================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
from torch.utils.data import Dataset
from torchvision.models.resnet import ResNet, Bottleneck
from PIL import Image
import cv2
from tensorflow.keras.datasets import cifar10

# Suppress known warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# ======================== 用户配置区域 ========================
TASK_TYPE = "binary"  # 修改为"binary"是二分类，"10-class"是10分类
# binary任务配置
SELECTED_CLASSES = [1, 8]          # 使用的类别索引（0-9）
TRAIN_SAMPLES_PER_CLASS = 1000     # 每个类别的训练样本量
TEST_SAMPLES_PER_CLASS = 250       # 每个类别的测试样本量
# 10-class任务配置
TRAIN_SAMPLES_10CLASS = 500       # 每个类别的训练样本量（总5000）
TEST_SAMPLES_10CLASS = 100        # 每个类别的测试样本量（总1000）
# ======================== 配置结束 ========================

# 全局参数
BATCH_SIZE = 256
PATIENCE = 15
CLASS_NAMES_FULL = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 根据任务类型初始化参数
if TASK_TYPE == "binary":
    CLASS_NAMES = [CLASS_NAMES_FULL[i] for i in SELECTED_CLASSES]
    NUM_CLASSES = len(SELECTED_CLASSES)
    IMG_SIZE = 32
    TRAIN_SAMPLES = TRAIN_SAMPLES_PER_CLASS * len(SELECTED_CLASSES)
    TEST_SAMPLES = TEST_SAMPLES_PER_CLASS * len(SELECTED_CLASSES)
else:
    CLASS_NAMES = CLASS_NAMES_FULL
    NUM_CLASSES = 10
    IMG_SIZE = 32
    TRAIN_SAMPLES = TRAIN_SAMPLES_10CLASS * 10
    TEST_SAMPLES = TEST_SAMPLES_10CLASS * 10

# ======================== 数据预处理（统一灰度处理）=======================
def get_transforms():
    # 使用正确的灰度图统计量
    mean = 0.4802
    std = 0.2304
    
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))  # 修正
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))  # 修正
    ])
    return train_transform, test_transform


# ======================== 修复后的Dataset类定义 ========================
class CIFARDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # 处理图像维度：移除单通道维度（H, W, 1）-> (H, W)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
            
        # 转换为PIL Image（灰度模式）
        img_pil = Image.fromarray(img.astype(np.uint8), mode='L')
        
        # 应用转换（关键修复：无论是否有transform都返回PIL Image）
        if self.transform:
            img = self.transform(img_pil)
        else:
            img = img_pil  # 保持PIL Image类型
            
        return img, label

# ======================== 数据加载 ========================
def convert_to_grayscale(images):
    """将RGB图像批量转换为灰度并添加通道维度"""
    grayscale_images = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray[..., np.newaxis]  # (H, W, 1)
        grayscale_images.append(gray)
    return np.array(grayscale_images, dtype=np.uint8)

def load_datasets():
    (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()
    y_train_full = y_train_full.ravel()
    y_test_full = y_test_full.ravel()

    if TASK_TYPE == "binary":
        # ================= 二分类处理 =================
        train_mask = np.isin(y_train_full, SELECTED_CLASSES)
        test_mask = np.isin(y_test_full, SELECTED_CLASSES)
        
        x_train = x_train_full[train_mask]
        y_train = y_train_full[train_mask]
        x_test = x_test_full[test_mask]
        y_test = y_test_full[test_mask]

        # 保存原始标签用于调试
        y_train_original = y_train.copy()  # 新增
        y_test_original = y_test.copy()    # 新增

        # 标签重映射
        label_mapping = {orig: idx for idx, orig in enumerate(SELECTED_CLASSES)}
        y_train = np.vectorize(label_mapping.get)(y_train)
        y_test = np.vectorize(label_mapping.get)(y_test)

        # 均衡采样
        def balanced_subsample(data, labels, samples_per_class):
            indices = []
            for class_idx in range(len(SELECTED_CLASSES)):
                class_indices = np.where(labels == class_idx)[0]
                selected = np.random.choice(class_indices, samples_per_class, replace=False)
                indices.extend(selected)
            # 新增随机打乱
            np.random.shuffle(indices)
            return data[indices], labels[indices]
        
        x_train, y_train = balanced_subsample(x_train, y_train, TRAIN_SAMPLES_PER_CLASS)
        x_test, y_test = balanced_subsample(x_test, y_test, TEST_SAMPLES_PER_CLASS)
        
        # 转换为灰度
        x_train = convert_to_grayscale(x_train)
        x_test = convert_to_grayscale(x_test)
    else:
        # ================= 十分类处理 =================
        def subsample(data, labels, samples_per_class):
            indices = []
            for class_idx in range(10):
                class_indices = np.where(labels == class_idx)[0]
                selected = np.random.choice(class_indices, samples_per_class, replace=False)
                indices.extend(selected)
            return data[indices], labels[indices]
        
        x_train, y_train = subsample(x_train_full, y_train_full, TRAIN_SAMPLES_10CLASS)
        x_test, y_test = subsample(x_test_full, y_test_full, TEST_SAMPLES_10CLASS)
        
        # 转换为灰度
        x_train = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., np.newaxis] for img in x_train], dtype=np.uint8)
        x_test = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., np.newaxis] for img in x_test], dtype=np.uint8)

    # 数据验证
    print(f"训练集类别分布: {np.bincount(y_train)}")
    print(f"测试集类别分布: {np.bincount(y_test)}")
    print(f"训练数据形状: {x_train.shape}, 数据类型: {x_train.dtype}")
    print(f"测试数据形状: {x_test.shape}, 数据类型: {x_test.dtype}")

    # 创建Dataset实例
    transform_train, transform_test = get_transforms()
    train_dataset = CIFARDataset(x_train, y_train, transform_train)
    test_dataset = CIFARDataset(x_test, y_test, transform_test)
    # 数据验证（修改调试输出）
    print("\n[DEBUG] 数据加载验证:")
    print(f"训练集原始类别分布: {np.bincount(y_train_original)}")  # 新增
    print(f"测试集原始类别分布: {np.bincount(y_test_original)}")    # 新增
    print(f"训练集映射后示例标签: {y_train[:10]}")  # 修改输出说明
    print(f"测试集映射后示例标签: {y_test[:10]}")  # 修改输出说明
    print(f"训练集唯一原始标签: {np.unique(y_train_original)} → 映射为: {np.unique(y_train)}")
    print(f"测试集唯一原始标签: {np.unique(y_test_original)} → 映射为: {np.unique(y_test)}")
    
    return train_dataset, test_dataset

# ======================== 模型定义（统一单通道输入）=======================
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

# ======================== 训练组件 ========================
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

# ======================== 模型评估 ========================
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_labels = []
    all_outputs = []
    inference_times = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            start_time = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize()
            inference_times.append(time.time() - start_time)
            
            all_labels.extend(labels.cpu().numpy())
            all_outputs.append(outputs.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    y_true = np.array(all_labels)
    y_score = torch.softmax(all_outputs, dim=1).numpy()
    y_pred = np.argmax(y_score, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算top-k准确率
    valid_ks = [k for k in [1,3,5] if k <= len(class_names)]
    topk_acc = {}
    for k in valid_ks:
        try:
            if k == 1:
                topk_acc[f'top_{k}_acc'] = accuracy
                continue
                
            top_k_preds = np.argsort(y_score, axis=1)[:, -k:]
            correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
            topk_acc[f'top_{k}_acc'] = np.mean(correct) * 100
        except:
            topk_acc[f'top_{k}_acc'] = 0.0
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'topk_acc': topk_acc,
        'inference_time': np.mean(inference_times)*1000
    }


# ======================== 训练流程 ========================
def train_model(model, train_loader, test_loader, epochs):
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
            
            epoch_monitor.update()
        
        eval_report = evaluate_model(model, test_loader, CLASS_NAMES)
        
        # 记录训练指标
        history['epoch'].append(epoch+1)
        history['train_loss'].append(total_loss/len(train_loader))
        history['train_acc'].append(100.*correct/total)
        history['test_acc'].append(eval_report['accuracy'])
        history['top3_acc'].append(eval_report['topk_acc'].get('top_3_acc', 0))
        history['top5_acc'].append(eval_report['topk_acc'].get('top_5_acc', 0))
        history['epoch_time'].append(time.time()-start_time)
        history['cpu_usage'].append(np.mean(epoch_monitor.cpu_usage))
        history['gpu_usage'].append(np.mean(epoch_monitor.gpu_usage))
        history['ram_usage'].append(np.mean(epoch_monitor.ram_usage))
        
        # 早停判断
        if early_stopper(eval_report['accuracy']):
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
        if eval_report['accuracy'] > best_acc:
            best_acc = eval_report['accuracy']
            torch.save(model.state_dict(), f'best_model_{TASK_TYPE}.pth')

        # 统一日志格式
        logging.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Train Acc: {history['train_acc'][-1]:.2f}% | "
            f"Test Acc: {history['test_acc'][-1]:.2f}% | "
            f"Top-3 Acc: {history['top3_acc'][-1]:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        scheduler.step()
    
    return history

# ======================== 可视化与报告 ========================
def plot_confusion_matrix(cm, class_names, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def generate_report(history, eval_report, filename_prefix):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(132)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history.get('val_loss', []), label='Validation')  # 新增
    plt.title('Training Loss')
    plt.legend()  # 新增
    
    plt.subplot(133)
    plt.plot(history['cpu_usage'], label='CPU')
    plt.plot(history['gpu_usage'], label='GPU')
    plt.title('Resource Usage')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_curves.png')
    plt.close()
    
    plot_confusion_matrix(eval_report['confusion_matrix'], 
                         CLASS_NAMES, 
                         f'{filename_prefix}_cm.png')
    
    final_report = {
        'task_type': TASK_TYPE,
        'train_samples': TRAIN_SAMPLES,
        'test_samples': TEST_SAMPLES,
        'best_accuracy': max(history['test_acc']),
        'best_top3': max(history['top3_acc']),
        'best_top5': max(history['top5_acc']),
        'total_training_time': sum(history['epoch_time']),
        'avg_inference_time': eval_report['inference_time'],
        'resource_usage': {
            'cpu': np.mean(history['cpu_usage']),
            'gpu': np.mean(history['gpu_usage']),
            'ram': np.mean(history['ram_usage'])
        }
    }
    
    with open(f'{filename_prefix}_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    pd.DataFrame(history).to_csv(f'{filename_prefix}_history.csv', index=False)
    return final_report

# ======================== 主函数 ========================
def main():
    # 初始化日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f"training_{TASK_TYPE}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        # 数据加载
        train_dataset, test_dataset = load_datasets()
        
        # 创建数据加载器
        num_workers = 0 if sys.platform.startswith('win') else 4
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE*2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # 初始化模型
        model = SEResNet(Bottleneck, [3, 4, 6, 3], num_classes=NUM_CLASSES).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"=== Model Initialized ===")
        logging.info(f"Model Parameters: {total_params/1e6:.2f}M")
        logging.info(f"Input shape: [B, 1, {IMG_SIZE}, {IMG_SIZE}]")
        logging.info(f"Number of classes: {NUM_CLASSES}")
        
        # 训练模型
        logging.info(f"\n=== Training Started ===")
        history = train_model(model, train_loader, test_loader, epochs=100)
        
        # 生成最终报告
        logging.info(f"\n=== Generating Final Report ===")
        eval_report = evaluate_model(model, test_loader, CLASS_NAMES)
        final_report = {
            'task_type': TASK_TYPE,
            'classes': SELECTED_CLASSES if TASK_TYPE == "binary" else list(range(10)),
            'class_names': CLASS_NAMES,
            'train_samples': TRAIN_SAMPLES,
            'test_samples': TEST_SAMPLES,
            'best_accuracy': max(history['test_acc']),
            'best_top3': max(history['top3_acc']),
            'best_top5': max(history['top5_acc']),
            'total_training_time': sum(history['epoch_time']),
            'avg_inference_time': eval_report['inference_time'],
            'resource_usage': {
                'cpu': np.mean(history['cpu_usage']),
                'gpu': np.mean(history['gpu_usage']),
                'ram': np.mean(history['ram_usage'])
            }
        }
        
        # 保存结果
        pd.DataFrame(history).to_csv(f'training_history_{TASK_TYPE}.csv', index=False)
        with open(f'final_report_{TASK_TYPE}.json', 'w') as f:
            json.dump(final_report, f, indent=2)
            
        # 保存混淆矩阵
        plot_confusion_matrix(eval_report['confusion_matrix'], 
                            CLASS_NAMES, 
                            f'confusion_matrix_{TASK_TYPE}.png')
        
        logging.info("\n=== Training Completed ===")
        logging.info(json.dumps(final_report, indent=2))

    except Exception as e:
        logging.error(f"Execution failed: {str(e)}")

def debug_data_pipeline():
    """修复后的数据管道调试函数"""
    (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()
    
    # 创建未应用转换的数据集（确保使用正确的图像格式）
    sample_img_np = cv2.cvtColor(x_train_full[0], cv2.COLOR_RGB2GRAY)  # (32, 32)
    sample_img_np = sample_img_np[..., np.newaxis]  # (32, 32, 1)
    
    dataset = CIFARDataset(
        images=np.array([sample_img_np]), 
        labels=np.array([0]), 
        transform=None
    )
    
    sample_img, _ = dataset[0]
    print(f"样本图像类型: {type(sample_img)}")  # 应该输出PIL.Image.Image
    print(f"转换前图像模式: {sample_img.mode}")  # 应该输出'L'
    
    # 应用转换流程验证
    train_transform, _ = get_transforms()
    tensor_img = train_transform(sample_img)
    print(f"张量形状: {tensor_img.shape}")    # 应该输出torch.Size([1, 32, 32])
    print(f"张量类型: {tensor_img.dtype}")    # 应该输出torch.float32
    print(f"数值范围: [{tensor_img.min()}, {tensor_img.max()}]")  # 应该在[0,1]范围内

if __name__ == "__main__":
    debug_data_pipeline()
    torch.multiprocessing.freeze_support()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main()