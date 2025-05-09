import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import sys
import psutil
import GPUtil
import gc
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import os

# ========================
# 配置参数
# ========================
class Config:
    SUBSET_CLASSES = 10
    SUBSET_SAMPLES = 2000
    TEST_SAMPLES = 500
    
    #全20
    OPTIM_STEPS = [
        {'name': 'Baseline', 'residual': False, 'augment': False, 'scheduler': 'cosine', 'epochs': 20},
        {'name': '+ResBlocks', 'residual': True, 'augment': False, 'scheduler': 'cosine', 'epochs': 20},
        {'name': '+Cutout', 'residual': True, 'augment': True, 'scheduler': 'cosine', 'epochs': 20},
        {'name': '+OneCycle', 'residual': True, 'augment': True, 'scheduler': 'onecycle', 'epochs': 20},
        {'name': 'Final', 'residual': True, 'augment': True, 'scheduler': 'onecycle', 'epochs': 20}
    ]
    
    BATCH_SIZE = 128
    BASE_LR = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    
    OUTPUT_DIR = "results"
    FIGURE_SIZE = (12, 6)

# ========================
# 初始化设置
# ========================
def setup_environment():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
        torch.backends.cudnn.benchmark = True
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Config.OUTPUT_DIR, "training.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

# ========================
# 数据增强模块（已修复）
# ========================
class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        """处理张量格式的输入，形状为(C, H, W)"""
        h, w = img.shape[1], img.shape[2]  # 修正为正确的张量维度访问
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length//2, 0, h)
        y2 = np.clip(y + self.length//2, 0, h)
        x1 = np.clip(x - self.length//2, 0, w)
        x2 = np.clip(x + self.length//2, 0, w)
        mask = torch.ones_like(img)
        mask[:, y1:y2, x1:x2] = 0
        return img * mask

def get_transforms(augment=False):
    """返回训练和测试的transform（已修复顺序问题）"""
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if augment:
        # 在ToTensor之后插入Cutout（修正插入位置）
        train_transforms.insert(3, Cutout(16))
    
    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    
    return (
        transforms.Compose(train_transforms),
        transforms.Compose(test_transforms)
    )

def load_datasets(augment=False, use_subset=True):
    transform_train, transform_test = get_transforms(augment)
    
    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    if not use_subset:
        return (
            torch.utils.data.DataLoader(full_train, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4),
            torch.utils.data.DataLoader(full_test, batch_size=Config.BATCH_SIZE*2, shuffle=False, num_workers=4)
        )

    def filter_classes(dataset):
        indices = []
        class_counts = {i:0 for i in range(Config.SUBSET_CLASSES)}
        for idx, (_, label) in enumerate(dataset):
            if label < Config.SUBSET_CLASSES and class_counts[label] < Config.SUBSET_SAMPLES:
                indices.append(idx)
                class_counts[label] += 1
        return Subset(dataset, indices)
    
    return (
        torch.utils.data.DataLoader(filter_classes(full_train), 
        batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4),
        torch.utils.data.DataLoader(filter_classes(full_test), 
        batch_size=Config.BATCH_SIZE*2, shuffle=False, num_workers=4)
    )

# ========================
# 模型定义（保持不变）
# ========================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return nn.functional.relu(out)

class CIFARNet(nn.Module):
    def __init__(self, use_residual=True):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)
        
    def _make_layer(self, channels, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(self.in_channels, channels, stride))
            stride = 1
            self.in_channels = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ========================
# 训练模块（保持不变）
# ========================
class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_acc': {}, 'epoch_time': [],
            'cpu_usage': [], 'gpu_usage': [], 'gpu_mem': []
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        monitor = ResourceMonitor()
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if np.random.rand() < 0.1:
                monitor.update()
        
        return {
            'loss': total_loss / len(train_loader),
            'acc': 100. * correct / total,
            'cpu': np.mean(monitor.cpu_usage),
            'gpu': np.mean(monitor.gpu_usage),
            'gpu_mem': np.mean(monitor.gpu_mem)
        }
    
    def evaluate(self, test_loaders):
        self.model.eval()
        results = {}
        for name, loader in test_loaders.items():
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            results[name] = 100. * correct / total
        return results
    
    def run(self, train_loader, test_loaders, epochs):
        for epoch in range(epochs):
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            test_metrics = self.evaluate(test_loaders)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            for name, acc in test_metrics.items():
                self.history['test_acc'].setdefault(name, []).append(acc)
            self.history['epoch_time'].append(time.time() - start_time)
            self.history['cpu_usage'].append(train_metrics['cpu'])
            self.history['gpu_usage'].append(train_metrics['gpu'])
            self.history['gpu_mem'].append(train_metrics['gpu_mem'])
            
            logging.info(f"Epoch {epoch+1}/{epochs} | "
                         f"Train Acc: {train_metrics['acc']:.2f}% | "
                         f"Test Acc: {test_metrics} | "
                         f"Time: {self.history['epoch_time'][-1]:.1f}s")
        return self.history

# ========================
# 辅助类（保持不变）
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
            self.gpu_usage.append(gpu.load * 100)
            self.gpu_mem.append(gpu.memoryUsed)
        except Exception as e:
            logging.warning(f"GPU监控失败: {str(e)}")
            self.gpu_usage.append(0)
            self.gpu_mem.append(0)

def run_experiment():
    setup_environment()
    full_train, full_test = load_datasets(use_subset=False)
    sub_train, sub_test = load_datasets(use_subset=True)
    
    all_results = []
    full_history = []
    
    for config in Config.OPTIM_STEPS:
        logging.info(f"\n=== 开始阶段: {config['name']} ===")
        is_final = config['name'] == 'Final'
        train_loader = full_train if is_final else sub_train
        #test_loaders = {'full': full_test, 'sub': sub_test} if is_final else {'sub': sub_test}
        # 统一测试集键名
        test_loaders = {'test': sub_test}  # 始终使用子集测试
        if is_final:
            test_loaders['full'] = full_test  # Final阶段额外添加全量测试
        
        model = CIFARNet(config['residual']).to(Config.device)
        optimizer = optim.SGD(model.parameters(), 
                            lr=Config.BASE_LR,
                            momentum=Config.MOMENTUM,
                            weight_decay=Config.WEIGHT_DECAY)
        scheduler = create_scheduler(optimizer, config, len(train_loader))
        
        if config['augment']:
            train_loader, _ = load_datasets(augment=True, use_subset=not is_final)
        
        trainer = Trainer(model, optimizer, nn.CrossEntropyLoss(), scheduler)
        history = trainer.run(train_loader, test_loaders, config['epochs'])
        
        # 显式处理所有可能的键
        stage_result = {
            'stage': config['name'],
            'epochs': config['epochs'],
            'sub_acc': max(history['test_acc'].get('test', [0])),
            'full_acc': max(history['test_acc'].get('full', [0])) if is_final else None
        }
        all_results.append(stage_result)
        full_history.append(history)
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()
    
    save_results(all_results, full_history)

def create_scheduler(optimizer, config, steps_per_epoch):
    if config['scheduler'] == 'onecycle':
        return optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=Config.BASE_LR,
            epochs=config['epochs'],
            steps_per_epoch=steps_per_epoch
        )
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs']
    )

def save_results(results, histories):
    df = pd.DataFrame(results, columns=['stage', 'epochs', 'sub_acc', 'full_acc'])
    df.to_csv(os.path.join(Config.OUTPUT_DIR, "results.csv"), index=False)
    plot_accuracy_comparison(df)
    plot_training_curves(histories)
    plot_resource_usage(histories)

def plot_accuracy_comparison(df):
    plt.figure(figsize=Config.FIGURE_SIZE)
    x = np.arange(len(df))
    plt.bar(x - 0.2, df['sub_acc'], width=0.4, label='Subset Test')
    if 'full_acc' in df.columns:
        final_idx = df.index[df['stage'] == 'Final'][0]
        plt.bar(final_idx + 0.2, df.loc[final_idx, 'full_acc'], 
                width=0.4, label='Full Test', color='orange')
    plt.title("Test Accuracy Comparison")
    plt.xlabel("Optimization Stage")
    plt.ylabel("Accuracy (%)")
    plt.xticks(x, df['stage'])
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "accuracy_comparison.png"), bbox_inches='tight')
    plt.close()

def plot_training_curves(histories):
    plt.figure(figsize=(14, 8))
    
    # 定义高区分度样式
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, hist in enumerate(histories):
        style = {
            'linestyle': line_styles[i % len(line_styles)],
            'marker': markers[i % len(markers)],
            'color': colors[i % len(colors)],
            'markersize': 8,
            'linewidth': 2
        }
        
        # 训练曲线
        plt.plot(
            hist['train_acc'], 
            label=f"{Config.OPTIM_STEPS[i]['name']} Train", 
            **style
        )
        
        # 测试曲线
        for test_name in hist['test_acc']:
            plt.plot(
                hist['test_acc'][test_name], 
                label=f"{Config.OPTIM_STEPS[i]['name']} {test_name.capitalize()} Test",
                **{**style, 'linestyle': '-'}  # 测试曲线统一用实线
            )
    
    plt.title("Training Progress", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=10,
        framealpha=0.9
    )
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_resource_usage(histories):
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    
    # 定义区分度高的样式组合
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, hist in enumerate(histories):
        style = {
            'linestyle': line_styles[i % len(line_styles)],
            'marker': markers[i % len(markers)],
            'markersize': 5,
            'color': colors[i % len(colors)],
            'linewidth': 1.5,
            'label': Config.OPTIM_STEPS[i]['name']
        }
        
        axs[0].plot(hist['cpu_usage'], **style)
        axs[1].plot(hist['gpu_usage'], **style)
        axs[2].plot(hist['gpu_mem'], **style)
    
    # 增强图表可读性
    titles = ["CPU Utilization (%)", "GPU Utilization (%)", "GPU Memory Usage (MB)"]
    for ax, title in zip(axs, titles):
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel(title.split(' ')[0], fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(
            bbox_to_anchor=(1.05, 1), 
            loc='upper left',
            framealpha=0.9
        )
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # 为图例留空间
    plt.savefig(
        os.path.join(Config.OUTPUT_DIR, "resource_usage.png"), 
        bbox_inches='tight', 
        dpi=150
    )
    plt.close()

# ========================
# 主程序
# ========================
if __name__ == "__main__":
    setup_environment()
    
    logging.info("===== 系统状态检查 =====")
    logging.info(f"PyTorch版本: {torch.__version__}")
    logging.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"当前GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    
    try:
        run_experiment()
        logging.info("实验成功完成！")
    except Exception as e:
        logging.error(f"实验失败: {str(e)}")
        raise