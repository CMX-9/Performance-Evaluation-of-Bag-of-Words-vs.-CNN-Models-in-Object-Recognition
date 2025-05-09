import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import psutil
from tensorflow.keras.datasets import cifar10

try:
    import pynvml
    HAS_GPU = True
except:
    HAS_GPU = False

# ####################################
# 可动态修改的配置参数
# ####################################
CLASSES = 2        # 使用的类别数（从CIFAR-10中选择前N类）
SAMPLE_SIZE = 500   # 训练样本量
TEST_SIZE = 100     # 测试样本量
K_VALUES = [100, 300, 500]  # 测试的视觉词典大小
SVM_PARAMS = {
    'C': 10,        # 固定参数
    'gamma': 0.01    # 固定参数
}

# ####################################
# 资源监控工具类
# ####################################
class ResourceMonitor:
    def __init__(self):
        self.cpu_usage = []
        if HAS_GPU:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.gpu_usage = []
    
    def measure(self):
        self.cpu_usage.append(psutil.cpu_percent())
        if HAS_GPU:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.gpu_usage.append(util.gpu)
    
    def get_stats(self):
        cpu_avg = np.mean(self.cpu_usage) if self.cpu_usage else 0
        gpu_avg = np.mean(self.gpu_usage) if self.gpu_usage else 0
        return cpu_avg, gpu_avg

# ####################################
# 1. 数据预处理模块
# ####################################
def load_data():
    """加载并预处理数据"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 筛选指定类别
    train_mask = np.squeeze(y_train < CLASSES)
    test_mask = np.squeeze(y_test < CLASSES)
    
    # 随机采样
    def random_sample(data, labels, size):
        idx = np.random.choice(len(data), size, replace=False)
        return data[idx], labels[idx].ravel()
    
    # 处理图像
    def process_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (64, 64))  # 调整尺寸以适配特征提取
    
    # 并行处理
    x_train_proc = np.array(Parallel(n_jobs=-1)(
        delayed(process_image)(img) for img in x_train[train_mask][:SAMPLE_SIZE]
    ))
    x_test_proc = np.array(Parallel(n_jobs=-1)(
        delayed(process_image)(img) for img in x_test[test_mask][:TEST_SIZE]
    ))
    
    return x_train_proc, y_train[train_mask][:SAMPLE_SIZE].ravel(), \
           x_test_proc, y_test[test_mask][:TEST_SIZE].ravel()

# ####################################
# 2. Harris-SIFT特征提取模块（已修复）
# ####################################
class HarrisSIFTFeatureExtractor:
    def __init__(self):
        self.sift = cv2.SIFT_create()
    
    def extract(self, img):
        """使用Harris角点检测+SIFT描述子提取特征"""
        gray = img.astype(np.uint8)
        
        # Harris角点检测参数
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=3,
            useHarrisDetector=True,
            k=0.035
        )
        
        if corners is not None:
            # 修正点1：使用正确的size参数名
            keypoints = [
                cv2.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), size=10)
                for c in corners
            ]
            # 计算SIFT描述子
            _, descriptors = self.sift.compute(gray, keypoints)
            if descriptors is not None:
                return descriptors
        
        return np.zeros((1, 128))  # 保证至少返回一个描述符

def extract_features(images):
    """并行特征提取"""
    extractor = HarrisSIFTFeatureExtractor()
    features = Parallel(n_jobs=-1, prefer="threads")(
        delayed(extractor.extract)(img) for img in tqdm(images, desc="特征提取")
    )
    return [f for f in features if f.shape[0] > 0]

# ####################################
# 3. 视觉词典构建
# ####################################
def build_vocab(features, k=500):
    """构建视觉词典"""
    all_descriptors = np.vstack(features)
    
    if len(all_descriptors) < k:
        raise ValueError(f"特征不足: 需要至少{k}个描述符，实际得到{len(all_descriptors)}个")
    
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=2048,
        compute_labels=False,
        random_state=42
    )
    
    for i in range(0, len(all_descriptors), 2048):
        kmeans.partial_fit(all_descriptors[i:i+2048])
    
    return kmeans

# ####################################
# 4. BoW特征生成
# ####################################
def bow_transform(features, kmeans):
    """生成词袋特征"""
    histograms = []
    for desc in tqdm(features, desc="生成BoW"):
        if desc.shape[0] == 0:
            hist = np.zeros(kmeans.n_clusters)
        else:
            visual_words = kmeans.predict(desc)
            hist = np.bincount(visual_words, minlength=kmeans.n_clusters)
            hist = hist / (hist.sum() + 1e-6)  # L1归一化
        histograms.append(hist)
    return np.array(histograms)

# ####################################
# 5. 实验流程（已修复预测调用）
# ####################################
def run_experiment():
    """执行完整实验流程"""
    results = {
        'k': [],
        'accuracy': [],
        'time': {'total': [], 'feature': [], 'clustering': [], 'training': []},
        'resource': {'cpu': [], 'gpu': []},
        'confusion_mat': [],
        'classification_report': []
    }
    
    # 加载数据
    start_time = time.time()
    x_train, y_train, x_test, y_test = load_data()
    results['time']['data_loading'] = time.time() - start_time
    
    # 特征提取
    feature_start = time.time()
    train_features = extract_features(x_train)
    test_features = extract_features(x_test)
    results['time']['feature'] = time.time() - feature_start
    
    for k in K_VALUES:
        try:
            print(f"\n{'='*40}\n正在测试 k={k}\n{'='*40}")
            iter_start = time.time()
            monitor = ResourceMonitor()
            
            # 构建词典
            cluster_start = time.time()
            with tqdm(total=100, desc="资源监控") as pbar:
                for _ in range(100):
                    monitor.measure()
                    pbar.update(1)
                    time.sleep(0.01)
            kmeans = build_vocab(train_features, k=k)
            results['time']['clustering'].append(time.time() - cluster_start)
            
            # 记录资源使用
            cpu_avg, gpu_avg = monitor.get_stats()
            results['resource']['cpu'].append(cpu_avg)
            results['resource']['gpu'].append(gpu_avg)
            
            # 生成特征
            X_train = bow_transform(train_features, kmeans)
            X_test = bow_transform(test_features, kmeans)
            
            # 训练SVM（使用固定参数）
            train_start = time.time()
            svm = SVC(
                C=SVM_PARAMS['C'],
                gamma=SVM_PARAMS['gamma'],
                kernel='rbf',
                class_weight='balanced',
                random_state=42
            )
            svm.fit(X_train, y_train)
            results['time']['training'].append(time.time() - train_start)
            
            # 修正点2：使用svm对象进行预测
            y_pred = svm.predict(X_test)  # 替换原来的grid.predict
            
            # 记录结果
            results['k'].append(k)
            results['accuracy'].append(accuracy_score(y_test, y_pred))
            results['confusion_mat'].append(confusion_matrix(y_test, y_pred))
            results['classification_report'].append(
                classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(CLASSES)]))
            
            results['time']['total'].append(time.time() - iter_start)
            
            print(f"\nk={k} 结果:")
            print(f"准确率: {results['accuracy'][-1]:.3f}")
            print(f"CPU使用率: {cpu_avg:.1f}% | GPU使用率: {gpu_avg:.1f}%")
            print(f"总耗时: {results['time']['total'][-1]:.1f}s")
            
        except Exception as e:
            print(f"k={k} 实验失败: {str(e)}")
    
    visualize_results(results)
    return results

def visualize_results(results):
    """生成可视化分析报告"""
    plt.figure(figsize=(20, 12))
    
    # 准确率曲线
    plt.subplot(2, 3, 1)
    plt.plot(results['k'], results['accuracy'], 'o-', markersize=8)
    plt.xlabel('视觉词典大小 (K)')
    plt.ylabel('准确率')
    plt.title('不同词典尺寸的准确率对比')
    plt.grid(alpha=0.3)
    
    # 时间分析
    plt.subplot(2, 3, 2)
    time_data = {
        '聚类耗时': results['time']['clustering'],
        '训练耗时': results['time']['training'],
        '总耗时': results['time']['total']
    }
    for label, data in time_data.items():
        plt.plot(results['k'], data, 'o--', label=label)
    plt.xlabel('视觉词典大小 (K)')
    plt.ylabel('时间 (秒)')
    plt.title('时间消耗分析')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 资源消耗
    plt.subplot(2, 3, 3)
    if any(results['resource']['cpu']):
        plt.plot(results['k'], results['resource']['cpu'], 's-', label='CPU使用率')
    if HAS_GPU and any(results['resource']['gpu']):
        plt.plot(results['k'], results['resource']['gpu'], 'd-', label='GPU使用率')
    plt.xlabel('视觉词典大小 (K)')
    plt.ylabel('利用率 (%)')
    plt.title('资源消耗分析')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 混淆矩阵
    plt.subplot(2, 3, 4)
    best_idx = np.argmax(results['accuracy'])
    sns.heatmap(
        results['confusion_mat'][best_idx],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[f'类别 {i}' for i in range(CLASSES)],
        yticklabels=[f'类别 {i}' for i in range(CLASSES)]
    )
    plt.title(f'混淆矩阵 (K={results["k"][best_idx]}, 准确率={results["accuracy"][best_idx]:.2f})')
    
    # 时间分布
    plt.subplot(2, 3, 5)
    time_labels = ['数据加载', '特征提取', '词典构建', '模型训练']
    time_values = [
        results['time']['data_loading'],
        np.mean(results['time']['feature']),
        np.mean(results['time']['clustering']),
        np.mean(results['time']['training'])
    ]
    plt.pie(time_values, labels=time_labels, autopct='%1.1f%%', startangle=90)
    plt.title('时间分布比例')
    
    plt.tight_layout()
    plt.savefig('性能分析.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = run_experiment()
    with open('实验结果汇总.txt', 'w', encoding='utf-8') as f:
        f.write("实验配置:\n")
        f.write(f"类别数: {CLASSES}\n")
        f.write(f"训练样本量: {SAMPLE_SIZE}\n")
        f.write(f"测试样本量: {TEST_SIZE}\n\n")
        
        f.write("最佳结果:\n")
        best_idx = np.argmax(results['accuracy'])
        f.write(f"K值: {results['k'][best_idx]} | 准确率: {results['accuracy'][best_idx]:.4f}\n\n")
        
        f.write("资源使用统计:\n")
        for k, cpu, gpu in zip(results['k'], results['resource']['cpu'], results['resource']['gpu']):
            f.write(f"K={k}: CPU平均使用率={cpu:.1f}% | GPU平均使用率={gpu:.1f}%\n")
        
        f.write("\n分类性能报告:\n")
        f.write(results['classification_report'][best_idx])