# ========================
# Robust Harris-SIFT Optimization
# ========================
import os
import time
import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from tqdm import tqdm
from tensorflow.keras.datasets import cifar10
from datetime import datetime

# ========================
# Environment Configuration
# ========================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ========================
# Experiment Parameters
# ========================
CLASSES = 2
SAMPLE_SIZE = 1000
TEST_SIZE = 200
R_VALUES = [0.01, 0.02, 0.03]
FIXED_K = 500
IMG_SIZE = (64, 64)
DEFAULT_SIZE = 10.0  # 关键点默认尺寸

# ========================
# Logging Configuration
# ========================
def setup_logger():
    global logger
    logger = logging.getLogger('HarrisOpt')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(
        f'logs/opt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# ========================
# Enhanced Feature Extractor
# ========================
def safe_feature_extraction(img, r_thresh):
    """
    修复关键点参数问题的特征提取
    """
    try:
        # 确保灰度输入
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape
        
        # Harris检测
        dst = cv2.cornerHarris(img.astype(np.float32), 3, 5, 0.04)
        dst_norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
        
        # 阈值处理
        threshold = r_thresh * dst_norm.max()
        y_coords, x_coords = np.where(dst_norm > threshold)
        
        # 非极大值抑制
        keypoints = []
        for y, x in zip(y_coords, x_coords):
            # 边界安全检测
            y_min = max(0, y-2)
            y_max = min(h, y+3)
            x_min = max(0, x-2)
            x_max = min(w, x+3)
            
            if dst_norm[y, x] == dst_norm[y_min:y_max, x_min:x_max].max():
                keypoints.append((x, y, dst_norm[y, x]))
        
        # 按响应值排序并创建合法关键点
        keypoints.sort(key=lambda k: -k[2])
        sift = cv2.SIFT_create()
        
        # 正确构造KeyPoint参数
        kps = [
            cv2.KeyPoint(
                x=float(x), 
                y=float(y),
                size=DEFAULT_SIZE,  # 必需参数
                response=response,
                angle=-1,
                octave=0,
                class_id=-1
            ) for (x, y, response) in keypoints[:500]  # 限制数量
            if 0 <= x < w and 0 <= y < h
        ]
        
        # 计算描述子
        if kps:
            _, descriptors = sift.compute(img, kps)
            return descriptors
        return np.zeros((1, 128))
        
    except Exception as e:
        logger.error(f"特征提取失败: {str(e)}")
        return np.zeros((1, 128))

# ========================
# Processing Pipeline
# ========================
def process_images(images, r_thresh):
    """稳定的特征生成"""
    try:
        return Parallel(n_jobs=2, prefer="processes")(
            delayed(safe_feature_extraction)(img, r_thresh)
            for img in tqdm(images, desc=f"R={r_thresh:.2f}")
        )
    except Exception as e:
        logger.error(f"并行处理失败: {str(e)}")
        return []

def optimized_workflow():
    """主优化流程"""
    # 数据加载与预处理
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    preprocess = lambda img: cv2.resize(
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 
        IMG_SIZE
    )
    
    x_train = np.array([preprocess(img) for img in x_train[y_train[:,0] < CLASSES][:SAMPLE_SIZE]])
    x_test = np.array([preprocess(img) for img in x_test[y_test[:,0] < CLASSES][:TEST_SIZE]])
    y_train = y_train[y_train[:,0] < CLASSES][:SAMPLE_SIZE]
    y_test = y_test[y_test[:,0] < CLASSES][:TEST_SIZE]
    
    results = {'R': [], 'Accuracy': [], 'Features': []}
    
    for r in R_VALUES:
        logger.info(f"Processing R={r:.2f}")
        
        # 特征提取
        train_features = [f for f in process_images(x_train, r) if f.shape[0] > 0]
        if len(train_features) < 10:
            logger.warning(f"跳过 R={r:.2f} - 特征不足")
            continue
            
        # 构建视觉词典
        kmeans = MiniBatchKMeans(
            n_clusters=FIXED_K, 
            batch_size=2048,
            random_state=42
        ).fit(np.vstack(train_features))
        
        # 生成BOW特征
        bow_transform = lambda features: np.array([
            np.bincount(kmeans.predict(f), minlength=FIXED_K) for f in features
        ])
        
        X_train = bow_transform(train_features)
        X_test = bow_transform([f for f in process_images(x_test, r) if f.shape[0] > 0])
        
        # 分类评估
        try:
            svm = SVC(C=1.0, kernel='linear').fit(X_train, y_train.ravel())
            acc = accuracy_score(y_test[:len(X_test)], svm.predict(X_test))
        except Exception as e:
            logger.error(f"分类失败: {str(e)}")
            continue
            
        # 记录结果
        results['R'].append(r)
        results['Accuracy'].append(acc)
        results['Features'].append(np.mean([f.shape[0] for f in train_features]))
        logger.info(f"R={r:.2f} | Acc={acc:.3f} | Features={results['Features'][-1]:.1f}")
    
    return results

# ========================
# Visualization & Main
# ========================
def plot_results(results):
    """结果可视化"""
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(df['R'], df['Accuracy'], 'bo-')
    plt.xlabel('Harris Threshold')
    plt.ylabel('Classification Accuracy')
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(df['R'], df['Features'], 'rs--')
    plt.xlabel('Harris Threshold')
    plt.ylabel('Average Features per Image')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('harris_optimization.png')
    plt.show()

if __name__ == "__main__":
    setup_logger()
    try:
        logger.info("=== Experiment Start ===")
        start_time = time.time()
        
        results = optimized_workflow()
        if results['R']:
            best_idx = np.argmax(results['Accuracy'])
            logger.info(f"Optimal R: {results['R'][best_idx]:.3f}")
            logger.info(f"Max Accuracy: {results['Accuracy'][best_idx]:.3f}")
            plot_results(results)
        else:
            logger.error("No valid results obtained")
        
        logger.info(f"Total Time: {time.time()-start_time:.1f}s")
    except Exception as e:
        logger.error(f"Main Error: {str(e)}")
        raise