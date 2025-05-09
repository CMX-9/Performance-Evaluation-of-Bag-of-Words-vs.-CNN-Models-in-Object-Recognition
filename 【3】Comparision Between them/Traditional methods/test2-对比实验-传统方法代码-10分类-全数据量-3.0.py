# ========================
# Harris-SIFT-SVM Single-K Analysis
# ========================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OMP_NUM_THREADS"] = "16"
import time
import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, 
                           confusion_matrix, 
                           top_k_accuracy_score)
from joblib import Parallel, delayed
from tqdm import tqdm
from tensorflow.keras.datasets import cifar10
from datetime import datetime
import psutil
import GPUtil
import json

# ======================== 资源监控类 ========================
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

# ======================== 环境配置 ========================
os.environ["OMP_NUM_THREADS"] = "6"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def configure_fonts():
    plt.rcParams.update({
        'figure.dpi': 300,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'font.family': 'sans-serif',
        'axes.unicode_minus': False
    })
    return 'sans-serif'

selected_font = configure_fonts()

# ======================== 参数配置 ========================
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
K = 500  # 固定视觉词典尺寸
IMG_SIZE = (128, 128)
MAX_FEATURES = 1000
HARRIS_THRESHOLD = 0.01

SVM_CONFIG = {
    'C': 2,
    'kernel': 'rbf',
    'gamma': 'scale',
    'class_weight': 'balanced',
    'decision_function_shape': 'ovo'
}

HARRIS_CONFIG = {
    'block_size': 5,
    'sobel_ksize': 3,
    'k': 0.06,
    'nms_size': 7
}

SIFT_CONFIG = {
    'contrastThreshold': 0.04,
    'edgeThreshold': 15
}

# ======================== 日志配置 ========================
def setup_logger():
    logger = logging.getLogger('HarrisAnalysis')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(
        f'logs/harris_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# ======================== 特征提取 ========================
def harris_feature_extraction(img):
    try:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (5,5), 1.5)
        
        dst = cv2.cornerHarris(
            img.astype(np.float32),
            blockSize=HARRIS_CONFIG['block_size'],
            ksize=HARRIS_CONFIG['sobel_ksize'],
            k=HARRIS_CONFIG['k']
        )
        
        dst_norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
        threshold = HARRIS_THRESHOLD * 255
        
        keypoints = []
        nms_size = HARRIS_CONFIG['nms_size']
        height, width = dst_norm.shape
        
        integral = cv2.integral(dst_norm)
        for y in range(height):
            for x in range(width):
                if dst_norm[y, x] > threshold:
                    y1 = max(0, y - nms_size//2)
                    y2 = min(height, y + nms_size//2 + 1)
                    x1 = max(0, x - nms_size//2)
                    x2 = min(width, x + nms_size//2 + 1)
                    
                    local_max = integral[y2, x2] + integral[y1, x1] - integral[y1, x2] - integral[y2, x1]
                    if dst_norm[y, x] >= local_max / ((y2-y1)*(x2-x1)):
                        keypoints.append((x, y, dst_norm[y, x]))
        
        keypoints.sort(key=lambda k: -k[2])
        keypoints = keypoints[:MAX_FEATURES]
        
        sift = cv2.SIFT_create(**SIFT_CONFIG)
        kps = [
            cv2.KeyPoint(
                x=float(x), 
                y=float(y),
                size=20 * (response/255)**0.5,
                response=response,
                angle=-1,
                octave=0,
                class_id=-1
            ) for (x, y, response) in keypoints
        ]
        
        _, descriptors = sift.compute(img, kps)
        return descriptors if descriptors is not None else np.zeros((1,128))
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        return np.zeros((1,128))

# ======================== 并行处理 ========================
def process_batch(images):
    return Parallel(n_jobs=4, prefer="threads")(
        delayed(harris_feature_extraction)(img)
        for img in tqdm(images, desc="Processing images")
    )

# ======================== 主流程 ========================
def main_workflow():
    # 加载完整数据集
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    def preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.medianBlur(img, 3)
        return img
    
    logger.info("===== Preprocessing Started =====")
    x_train = np.array([preprocess(img) for img in x_train])
    x_test = np.array([preprocess(img) for img in x_test])
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    logger.info("===== Feature Extraction Started =====")
    feature_monitor = ResourceMonitor()
    start_feature = time.time()
    
    with tqdm(total=len(x_train)+len(x_test), desc="Processing images") as pbar:
        train_features = process_batch(x_train)
        test_features = process_batch(x_test)
        pbar.update(len(x_train)+len(x_test))
    
    feature_time = time.time() - start_feature
    
    def measure_inference_time():
        sample_img = x_test[0]
        start = time.time()
        for _ in range(100):
            harris_feature_extraction(sample_img)
        return (time.time() - start) * 10  # ms per image

    results = {
        'K': K,
        'train_time': 0.0,
        'test_time': 0.0,
        'train_acc': 0.0,
        'test_acc': 0.0,
        'top1_acc': 0.0,
        'top3_acc': 0.0,
        'top5_acc': 0.0,
        'cpu_usage': 0.0,
        'gpu_usage': 0.0,
        'ram_usage': 0.0,
        'inference_time': 0.0,
        'total_time': 0.0
    }
    
    total_start = time.time()
    monitor = ResourceMonitor()
    iteration_start = time.time()
    
    try:
        # KMeans训练
        kmeans_start = time.process_time()
        kmeans = MiniBatchKMeans(
            n_clusters=K,
            batch_size=4096,
            random_state=42,
            n_init='auto'
        ).fit(np.vstack(train_features))
        kmeans_time = time.process_time() - kmeans_start
        
        # BOW特征生成
        def bow_transform(features):
            return np.array([
                np.bincount(kmeans.predict(f), minlength=K)
                if f.shape[0] > 0 else np.zeros(K)
                for f in features
            ])
        
        X_train = bow_transform(train_features)
        X_test = bow_transform(test_features)
        
        # 特征归一化
        X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
        X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)
        
        # SVM训练
        svm_start = time.process_time()
        svm = SVC(**SVM_CONFIG).fit(X_train, y_train)
        svm_time = time.process_time() - svm_start
        
        # 模型评估
        y_train_pred = svm.predict(X_train)
        y_test_pred = svm.predict(X_test)
        y_test_score = svm.decision_function(X_test)
        
        eval_report = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'top1': top_k_accuracy_score(y_test, y_test_score, k=1),
            'top3': top_k_accuracy_score(y_test, y_test_score, k=3),
            'top5': top_k_accuracy_score(y_test, y_test_score, k=5),
            'cm': confusion_matrix(y_test, y_test_pred)
        }
        
        # 可视化混淆矩阵
        plt.figure(figsize=(20, 18))
        sns.heatmap(eval_report['cm'], 
                    annot=True, fmt='d',
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES,
                    cmap='Blues')
        plt.title(f"Confusion Matrix (K={K}, Acc={eval_report['accuracy']:.2f})")
        plt.savefig(f'cm_k{K}.png', bbox_inches='tight')
        plt.close()
        
        # 记录结果
        results['train_time'] = kmeans_time + svm_time
        results['test_time'] = time.time() - iteration_start
        results['train_acc'] = accuracy_score(y_train, y_train_pred)
        results['test_acc'] = eval_report['accuracy']
        results['top1_acc'] = eval_report['top1']
        results['top3_acc'] = eval_report['top3']
        results['top5_acc'] = eval_report['top5']
        results['cpu_usage'] = np.mean(monitor.cpu_usage)
        results['gpu_usage'] = np.mean(monitor.gpu_usage)
        results['ram_usage'] = np.mean(monitor.ram_usage)
        results['inference_time'] = measure_inference_time()
        
        logger.info(
            f"Test Acc: {eval_report['accuracy']:.4f} | "
            f"Top-3 Acc: {eval_report['top3']:.4f} | "
            f"RAM Usage: {np.mean(monitor.ram_usage):.1f}%"
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return results
    
    # 生成最终报告
    results['total_time'] = time.time() - total_start
    final_report = {
        'K': K,
        'accuracy': results['test_acc'],
        'top3_accuracy': results['top3_acc'],
        'top5_accuracy': results['top5_acc'],
        'avg_cpu_usage': results['cpu_usage'],
        'avg_gpu_usage': results['gpu_usage'],
        'avg_ram_usage': results['ram_usage'],
        'total_time': results['total_time'],
        'inference_time': results['inference_time']
    }
    
    # 保存结果
    pd.DataFrame([results]).to_csv('single_k_results.csv', index=False)
    with open('final_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
        
    return results

# ======================== 执行入口 ========================
if __name__ == "__main__":
    logger.info("===== Single-K Analysis Started =====")
    start_time = time.time()
    
    try:
        results = main_workflow()
        
        logger.info("\n===== Final Results =====")
        logger.info(f"Accuracy: {results['test_acc']:.4f}")
        logger.info(f"Top-3 Accuracy: {results['top3_acc']:.4f}")
        logger.info(f"Total execution time: {results['total_time']:.1f}s")
        
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}", exc_info=True)
    finally:
        logger.info(f"Total execution time: {time.time()-start_time:.1f} seconds")