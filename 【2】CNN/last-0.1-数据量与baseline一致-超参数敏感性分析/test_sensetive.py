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


# Suppress known warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# ========================
# Neural Network Definition
# ========================
class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extraction layers
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
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
        # Enable gradients
        self.features.requires_grad_(True)
        self.classifier.requires_grad_(True)
        
        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, 
                use_reentrant=False,
                preserve_rng_state=True
            )
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        with torch.set_grad_enabled(self.training):
            x = self.features(x)
            return self.classifier(x)

# ========================
# Experiment Configuration
# ========================
BATCH_SIZE = 128
EPOCHS = 30
BASE_LR = 0.1
MOMENTUM = 0.9
BASE_WD = 5e-4
LR_RANGE = [0.001, 0.01, 0.1, 0.5]
WD_RANGE = [0, 1e-5, 5e-4, 1e-2]
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def filter_classes(dataset, max_samples_per_class=2000):
    """确保每个类别精确采样 max_samples_per_class 个样本"""
    indices = []
    class_counts = {i: 0 for i in range(10)}  # CIFAR-10 共10类
    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < max_samples_per_class:
            indices.append(idx)
            class_counts[label] += 1
    return Subset(dataset, indices)

def load_datasets():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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

    # ✅ 只保留前 5 类（0 到 4）
    allowed_classes = list(range(10))  # 可改为 range(3) 等
    # ✅ 修改：每个类别取 2000 样本（共 20000 训练样本）
    filtered_train = filter_classes(train_set, max_samples_per_class=2000)
    # ✅ 测试集使用完整数据（每个类别 1000 样本，共 10000 样本）
    filtered_test = filter_classes(test_set, max_samples_per_class=1000)  

    # ✅ 控制数据量
    train_subset = Subset(filtered_train, range(min(2000, len(filtered_train))))
    test_subset = Subset(filtered_test, range(min(500, len(filtered_test))))

    # ✅ 移除对数据量的额外截断（删除原有 Subset 操作）
    train_loader = torch.utils.data.DataLoader(
        filtered_train,  # 直接使用过滤后的数据
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        filtered_test,  # 直接使用过滤后的数据
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    
    return train_loader, test_loader

# ========================
# Training Components
# ========================
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

# ========================
# Modified Training Components
# ========================
def train_model(model, optimizer, criterion, train_loader, scheduler=None, test_loader):
    print(f">>> Starting new training with EPOCHS = {EPOCHS}")
    model.train()
    
    # 统一数据结构：每个epoch对应一个值
    history = {
        'epoch_loss': [],
        'epoch_acc': [],
        'epoch_test_acc': [],  # ✅ 添加测试准确度
        'epoch_time': [],
        'avg_cpu_usage': [],
        'avg_gpu_usage': [],
        'avg_gpu_mem': []
    }
    
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
                    # 数据迁移到GPU
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # 梯度清零
                    optimizer.zero_grad(set_to_none=True)
                    
                    # 前向传播
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    # 统计计算
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # 每10个batch记录资源
                    if batch_idx % 10 == 0:
                        epoch_monitor.update()
                    
                    # 显存清理
                    if batch_idx % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 计算epoch指标
            history['epoch_loss'].append(total_loss / len(train_loader))
            history['epoch_acc'].append(100. * correct / total)
            history['epoch_time'].append(time.time() - epoch_start)
            # ✅ 添加：在每个 epoch 后计算测试准确度
            if test_loader is not None:
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                test_acc = accuracy_score(all_labels, all_preds) * 100
                history['epoch_test_acc'].append(test_acc)
            else:
                history['epoch_test_acc'].append(np.nan)
            
            # 计算资源使用平均值
            history['avg_cpu_usage'].append(np.mean(epoch_monitor.cpu_usage) if epoch_monitor.cpu_usage else 0)
            history['avg_gpu_usage'].append(np.mean(epoch_monitor.gpu_usage) if epoch_monitor.gpu_usage else 0)
            history['avg_gpu_mem'].append(np.mean(epoch_monitor.gpu_mem) if epoch_monitor.gpu_mem else 0)
            
            # 学习率调度
            if scheduler:
                scheduler.step()
                
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user!")
        return None
    
    return history

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)*100, confusion_matrix(all_labels, all_preds)

# ========================
# Visualization Functions
# ========================
def plot_resource_usage(history):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # CPU Usage
    ax1.plot(history['avg_cpu_usage'], label='CPU Usage')
    ax1.set_title('Average CPU Utilization per Epoch (%)')
    ax1.set_ylim(0, 100)
    
    # GPU Usage
    ax2.plot(history['avg_gpu_usage'], label='GPU Usage', color='orange')
    ax2.set_title('Average GPU Utilization per Epoch (%)')
    ax2.set_ylim(0, 100)
    
    # GPU Memory
    ax3.plot(history['avg_gpu_mem'], label='GPU Memory', color='green')
    ax3.set_title('Average GPU Memory Usage per Epoch (MB)')
    
    plt.tight_layout()
    plt.savefig('resource_usage.png', bbox_inches='tight')
    plt.close()

def plot_training_and_test_accuracy(history):
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch_acc'], label='Train Accuracy', marker='o')
    plt.plot(history['epoch_test_acc'], label='Test Accuracy', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Test Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_vs_test_accuracy.png', bbox_inches='tight')
    plt.close()


def plot_final_accuracy(history):
    plt.figure(figsize=(10,6))
    plt.plot(history['epoch_acc'], 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Progress')
    plt.grid(True)
    plt.savefig('training_accuracy.png', bbox_inches='tight')
    plt.close()

def plot_hyperparameter_sensitivity(results_df, param_type):
    filtered_df = results_df[results_df['type'] == param_type]
    plt.figure(figsize=(10,6))
    plt.plot(filtered_df['value'], filtered_df['accuracy'], 'o-', markersize=8)
    plt.xscale('log' if param_type == 'LR' else 'linear')
    plt.xlabel(param_type)
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'{param_type} Sensitivity Analysis')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{param_type}_sensitivity.png', bbox_inches='tight')
    plt.close()

def plot_training_curves(histories, param_values, param_name):
    plt.figure(figsize=(12,6))
    for value, hist in zip(param_values, histories):
        plt.plot(hist['loss'], label=f'{param_name}={value}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss Comparison ({param_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'loss_curves_{param_name}.png', bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                square=True,
                cbar_kws={"shrink": 0.8})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()

# ========================
# Modified Main Workflow
# ========================
def main():
    try:
        # Environment setup
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logging.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
        
        # Load data
        train_loader, test_loader = load_datasets()
        
        # Hyperparameter experiments
        hyper_results = []
        
        # Learning rate experiment
        logging.info("\n===== Learning Rate Experiment =====")
        for lr in LR_RANGE:
            try:
                model = CIFARNet().to(device)
                opt = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=BASE_WD)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
                
                hist = train_model(model, opt, nn.CrossEntropyLoss(), train_loader, scheduler)
                acc, _ = evaluate_model(model, test_loader)
                
                # 确保数据类型统一
                hyper_results.append({
                    'type': 'LR',
                    'value': float(lr),
                    'accuracy': float(acc),
                    'time': float(sum(hist['epoch_time'])) if hist else 0.0
                })
                
            except Exception as e:
                logging.error(f"LR experiment failed for lr={lr}: {str(e)}")
                hyper_results.append({
                    'type': 'LR',
                    'value': float(lr),
                    'accuracy': 0.0,
                    'time': 0.0
                })
                
            finally:
                if 'model' in locals(): del model
                if 'opt' in locals(): del opt
                if 'scheduler' in locals(): del scheduler
                torch.cuda.empty_cache()
                gc.collect()
        
        # Weight decay experiment
        logging.info("\n===== Weight Decay Experiment =====")
        for wd in WD_RANGE:
            try:
                model = CIFARNet().to(device)
                opt = optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=wd)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
                
                hist = train_model(model, opt, nn.CrossEntropyLoss(), train_loader, scheduler)
                acc, _ = evaluate_model(model, test_loader)
                
                hyper_results.append({
                    'type': 'WD',
                    'value': float(wd),
                    'accuracy': float(acc),
                    'time': float(sum(hist['epoch_time'])) if hist else 0.0
                })
                
            except Exception as e:
                logging.error(f"WD experiment failed for wd={wd}: {str(e)}")
                hyper_results.append({
                    'type': 'WD',
                    'value': float(wd),
                    'accuracy': 0.0,
                    'time': 0.0
                })
                
            finally:
                if 'model' in locals(): del model
                if 'opt' in locals(): del opt
                if 'scheduler' in locals(): del scheduler
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final training
        logging.info("\n===== Final Training =====")
        try:
            # 创建DataFrame前验证数据一致性
            validated_data = []
            for item in hyper_results:
                validated_data.append({
                    'type': str(item['type']),
                    'value': float(item['value']),
                    'accuracy': float(item['accuracy']),
                    'time': float(item['time'])
                })
            results_df = pd.DataFrame(validated_data)
            
            # 选择最佳参数
            best_lr = results_df[results_df['type'] == 'LR'].sort_values('accuracy', ascending=False).iloc[0]
            best_wd = results_df[results_df['type'] == 'WD'].sort_values('accuracy', ascending=False).iloc[0]
            
            # 最终训练
            final_model = CIFARNet().to(device)
            opt = optim.SGD(
                final_model.parameters(),
                lr=float(best_lr['value']),
                momentum=MOMENTUM,
                weight_decay=float(best_wd['value'])
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            
            final_history = train_model(final_model, opt, nn.CrossEntropyLoss(), train_loader, scheduler)
            final_acc, final_cm = evaluate_model(final_model, test_loader)
            
            # 保存结果前验证数据长度
            max_len = max(len(v) for v in final_history.values())
            for k in final_history:
                final_history[k] = final_history[k] + [np.nan]*(max_len - len(final_history[k]))
            
            # 保存数据
            pd.DataFrame(final_history).to_csv('final_training.csv', index=False)
            results_df.to_csv('hyperparameter_results.csv', index=False)
            
            # 生成可视化
            plot_resource_usage(final_history)
            plot_final_accuracy(final_history)
            plot_confusion_matrix(final_cm)
            plot_hyperparameter_sensitivity(results_df, 'LR')
            plot_hyperparameter_sensitivity(results_df, 'WD')
            plot_training_and_test_accuracy(final_history)

            
            logging.info("\n=== Experiment Complete ===")
            logging.info(f"Best LR: {best_lr['value']}, Accuracy: {best_lr['accuracy']:.2f}%")
            logging.info(f"Best WD: {best_wd['value']}, Accuracy: {best_wd['accuracy']:.2f}%")
            logging.info(f"Final Model Accuracy: {final_acc:.2f}%")
            
        except Exception as e:
            logging.error(f"Final training failed: {str(e)}")
            raise
        
    except Exception as e:
        logging.error(f"Execution failed: {str(e)}")
        raise
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Environment validation
    print("===== System Status =====")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"VRAM Usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Gradient check
    test_model = CIFARNet().to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    y = test_model(x)
    loss = y.sum()
    loss.backward()
    print("Gradient check passed")
    
    # Execute main program
    main()