# ========================
# Robust Harris-SIFT Optimization (Final)
#终端需设置环境变量
#set OMP_NUM_THREADS=8
#set TF_ENABLE_ONEDNN_OPTS=0
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
from mpl_toolkits.mplot3d import Axes3D

# ========================
# Environment Configuration
# ========================
os.environ["OMP_NUM_THREADS"] = "8"  # 修复内存泄漏
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ========================
# Enhanced Parameters
# ========================
CLASSES = 2
SAMPLE_SIZE = 500      # 减小样本量加速调试
TEST_SIZE = 200
R_VALUES = [0.01, 0.03, 0.05, 0.07, 0.09]  # 扩展R范围
ADAPTIVE_K = True
BASE_K = 200
IMG_SIZE = (64, 64)
DEFAULT_SIZE = 10.0
MIN_FEATURES = 80      # 提升最小特征阈值
HARRIS_CONFIG = {      # 可调节Harris参数
    'block_size': 5,
    'sobel_ksize': 3,
    'k': 0.06
}

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
# Optimized Feature Extractor
# ========================
def safe_feature_extraction(img, r_thresh):
    """动态响应特征提取"""
    try:
        # 增强图像预处理
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)  # 直方图均衡化
        h, w = img.shape
        
        # 参数化Harris检测
        dst = cv2.cornerHarris(
            img.astype(np.float32),
            blockSize=HARRIS_CONFIG['block_size'],
            ksize=HARRIS_CONFIG['sobel_ksize'],
            k=HARRIS_CONFIG['k']
        )
        dst_norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
        
        # 自适应阈值
        threshold = r_thresh * dst_norm.max()
        y_coords, x_coords = np.where(dst_norm > threshold)
        
        # 优化的非极大值抑制
        keypoints = []
        neighborhood_size = 5  # 增大邻域范围
        for y, x in zip(y_coords, x_coords):
            y_min = max(0, y-neighborhood_size)
            y_max = min(h, y+neighborhood_size+1)
            x_min = max(0, x-neighborhood_size)
            x_max = min(w, x+neighborhood_size+1)
            
            local_max = dst_norm[y_min:y_max, x_min:x_max].max()
            if dst_norm[y, x] >= local_max:
                keypoints.append((x, y, dst_norm[y, x]))
        
        # 动态关键点选择
        keypoints.sort(key=lambda k: -k[2])
        sift = cv2.SIFT_create()
        
        # 动态调整保留比例
        retain_ratio = 0.9 - r_thresh*15  # R越大保留越少
        max_keypoints = max(
            MIN_FEATURES, 
            int(len(keypoints)*retain_ratio)
        )
        kps = [
            cv2.KeyPoint(
                x=float(x), 
                y=float(y),
                size=DEFAULT_SIZE * (response/255)**2,  # 非线性尺寸映射
                response=response,
                angle=-1,
                octave=0,
                class_id=-1
            ) for (x, y, response) in keypoints[:max_keypoints]
            if 0 <= x < w and 0 <= y < h
        ]
        
        # 特征调试日志
        logger.debug(f"R={r_thresh:.2f} | Raw: {len(keypoints)} | Final: {len(kps)}")
        
        if kps:
            _, descriptors = sift.compute(img, kps)
            return descriptors
        return np.zeros((1, 128))
        
    except Exception as e:
        logger.error(f"特征提取失败: {str(e)}")
        return np.zeros((1, 128))

# ========================
# Enhanced Parallel Processing
# ========================
def process_images(images, r_thresh):
    """带资源限制的并行处理"""
    try:
        return Parallel(n_jobs=4, max_nbytes=None, prefer="threads")(  # 改用线程
            delayed(safe_feature_extraction)(img, r_thresh)
            for img in tqdm(images, desc=f"R={r_thresh:.2f}")
        )
    except Exception as e:
        logger.error(f"并行处理失败: {str(e)}")
        return []

# ========================
# Robust Workflow
# ========================
def optimized_workflow():
    """抗噪声优化流程"""
    # 数据加载增强
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    def preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.medianBlur(img, 3)  # 中值滤波去噪
        return img
    
    # 数据筛选优化
    train_mask = (y_train[:,0] < CLASSES).flatten()
    test_mask = (y_test[:,0] < CLASSES).flatten()
    
    x_train = np.array([preprocess(img) for img in x_train[train_mask][:SAMPLE_SIZE]])
    x_test = np.array([preprocess(img) for img in x_test[test_mask][:TEST_SIZE]])
    y_train = y_train[train_mask][:SAMPLE_SIZE].ravel()
    y_test = y_test[test_mask][:TEST_SIZE].ravel()
    
    results = {'R': [], 'Accuracy': [], 'Features': [], 'K': []}
    
    for r in R_VALUES:
        logger.info(f"\n===== Processing R={r:.2f} =====")
        
        # 特征提取增强
        train_features = [f for f in process_images(x_train, r) if f.shape[0] > 0]
        if not train_features:
            logger.warning(f"跳过 R={r:.2f} - 无有效特征")
            continue
            
        # 动态K值计算（带有效性检查）
        valid_features = [f for f in train_features if f.shape[0] > 0]
        avg_features = np.mean([f.shape[0] for f in valid_features]) if valid_features else 0
        logger.debug(f"有效特征样本数: {len(valid_features)}")
        
        if ADAPTIVE_K:
            current_k = min(BASE_K, int(avg_features//1.5))  # 调整系数
            current_k = max(current_k, 50, int(avg_features*0.3))  # 多重保护
            current_k = min(current_k, 500)  # 上限控制
        else:
            current_k = BASE_K
        
        logger.info(f"平均特征数: {avg_features:.1f} | 动态K值: {current_k}")
        
        # 视觉词典构建（修复内存泄漏）
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=current_k,
                batch_size=5120,  # 增大batch_size
                random_state=42,
                n_init='auto'     # 自动选择初始化次数
            ).fit(np.vstack(train_features))
        except Exception as e:
            logger.error(f"KMeans训练失败: {str(e)}")
            continue
        
        # 增强BOW特征生成
        def bow_transform(features):
            histograms = []
            for f in features:
                if f.shape[0] == 0:
                    hist = np.zeros(current_k)
                else:
                    hist = np.bincount(
                        kmeans.predict(f),
                        minlength=current_k
                    )
                histograms.append(hist)
            return np.array(histograms)
        
        # 特征归一化增强
        X_train = bow_transform(train_features)
        with np.errstate(divide='ignore', invalid='ignore'):
            X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
        
        # 测试集处理
        test_features = [f for f in process_images(x_test, r) if f.shape[0] > 0]
        X_test = bow_transform(test_features)
        with np.errstate(divide='ignore', invalid='ignore'):
            X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)
        
        # 分类模型增强
        try:
            svm = SVC(
                C=0.7, 
                kernel='rbf',     # 改用RBF核
                class_weight='balanced',
                gamma='scale'
            ).fit(X_train, y_train)
            acc = accuracy_score(y_test[:len(X_test)], svm.predict(X_test))
            logger.info(f"分类准确率: {acc:.3f}")
        except Exception as e:
            logger.error(f"分类失败: {str(e)}")
            continue
            
        results['R'].append(r)
        results['Accuracy'].append(acc)
        results['Features'].append(avg_features)
        results['K'].append(current_k)
    
    return results

# ========================
# Robust R Selection
# ========================
def select_optimal_R(results):
    """带安全机制的优化选择"""
    df = pd.DataFrame(results)
    
    # 空结果处理
    if df.empty or df['Accuracy'].isnull().all():
        logger.warning("无有效结果，返回默认R值")
        return R_VALUES[0] if R_VALUES else 0.05
    
    # 安全标准化
    eps = 1e-8
    metrics = ['Accuracy', 'Features', 'K']
    for col in metrics:
        col_min = df[col].min()
        col_max = df[col].max()
        span = col_max - col_min + eps
        df[f'{col}_norm'] = (df[col] - col_min) / span
    
    # 动态权重调整
    accuracy_weight = 0.6
    features_weight = 0.3 if df['Features_norm'].mean() < 0.7 else 0.2  # 特征充足时降低权重
    complexity_weight = 0.1
    
    # 复合评分（带对数变换）
    df['score'] = (
        accuracy_weight * df['Accuracy_norm'] +
        features_weight * np.log1p(df['Features_norm']*10) +
        complexity_weight * (1 - df['K_norm']))
    
    # NaN处理
    if df['score'].isnull().all():
        logger.warning("评分全为NaN，返回最佳Accuracy")
        return df.loc[df['Accuracy'].idxmax(), 'R']
    
    # 可视化
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(df['R'], df['K'], df['Accuracy'], c=df['score'], cmap='viridis', s=100)
    ax.set_xlabel('R Value', fontsize=10)
    ax.set_ylabel('K Value', fontsize=10)
    ax.set_zlabel('Accuracy', fontsize=10)
    
    ax = fig.add_subplot(132)
    df.plot(x='R', y='score', kind='line', ax=ax, marker='o', color='m')
    ax.set_title('Composite Score Trend')
    ax.grid(True)
    
    ax = fig.add_subplot(133)
    parts = ax.stackplot(df['R'], 
                       df['Accuracy_norm']*accuracy_weight,
                       np.log1p(df['Features_norm']*10)*features_weight,
                       (1-df['K_norm'])*complexity_weight,
                       labels=['Accuracy','Features','Complexity'])
    ax.legend(loc='upper right')
    ax.set_title('Score Composition')
    
    plt.tight_layout()
    plt.savefig('optimization_analysis.png')
    plt.close()
    
    return df.loc[df['score'].idxmax(), 'R']

# ========================
# Enhanced Visualization
# ========================
def plot_results(results):
    """交互式可视化"""
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(15,6))
    
    plt.subplot(131)
    plt.plot(df['R'], df['Accuracy'], 'bo-', markersize=8, label='Accuracy')
    plt.fill_between(df['R'], df['Accuracy'], alpha=0.1, color='blue')
    plt.xlabel('R Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(132)
    plt.bar(df['R'], df['Features'], width=0.02, color='orange', alpha=0.6, label='Features')
    plt.plot(df['R'], df['K'], 'g^--', markersize=8, label='K Value')
    plt.xlabel('R Value')
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(133)
    plt.scatter(df['Features'], df['Accuracy'], c=df['R'], cmap='viridis', s=100)
    plt.colorbar(label='R Value')
    plt.xlabel('Average Features')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_dashboard.png')
    plt.close()

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    setup_logger()
    logger.info("===== 实验开始 =====")
    start_time = time.time()
    
    try:
        results = optimized_workflow()
        
        if results and results['R']:
            logger.info("\n===== 结果汇总 =====")
            for r, acc, feat, k in zip(results['R'], results['Accuracy'], results['Features'], results['K']):
                logger.info(f"R={r:.2f} | Acc={acc:.3f} | Features={feat:.1f} | K={k}")
            
            optimal_R = select_optimal_R(results)
            logger.info(f"\n===== 最优参数 =====")
            logger.info(f"* Optimal R: {optimal_R:.3f}")
            logger.info(f"* Max Accuracy: {max(results['Accuracy']):.3f}")
            
            plot_results(results)
        else:
            logger.error("未获得有效结果")
        
    except Exception as e:
        logger.error(f"主流程错误: {str(e)}", exc_info=True)
    finally:
        logger.info(f"总耗时: {time.time()-start_time:.1f}秒")