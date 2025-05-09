# ========================
# Harris-SIFT-SVM Analysis with Enhanced Performance Profiling
# ========================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OMP_NUM_THREADS"] = "16"  # Limit threads to prevent memory issues
import time
import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import Parallel, delayed
from tqdm import tqdm
from tensorflow.keras.datasets import cifar10
from datetime import datetime
from matplotlib import font_manager
import psutil
from scipy.interpolate import make_interp_spline

# ========================
# Environment Configuration
# ========================
os.environ["OMP_NUM_THREADS"] = "6"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Font configuration
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

# ========================
# Parameter Configuration
# ========================
CLASSES = [1, 8]
SAMPLE_SIZE = 2000
TEST_SIZE = 500
K_VALUES = [50,100, 200, 300, 500, 800, 1000]  #50, 150, 300, 500, 800, 1000, 1300
IMG_SIZE = (128, 128)
MAX_FEATURES = 800
HARRIS_THRESHOLD = 0.02  # 新增：固定角点筛选阈值

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

# ========================
# Logging Configuration
# ========================
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

# ========================
# Feature Extraction
# ========================
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
        #threshold = 0.035 * dst_norm.max()
        threshold = HARRIS_THRESHOLD * 255  # 统一为固定绝对阈值
        
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

# ========================
# Parallel Processing
# ========================
def process_batch(images):
    return Parallel(n_jobs=4, prefer="threads")(
        delayed(harris_feature_extraction)(img)
        for img in tqdm(images, desc="Processing images")
    )

# ========================
# Main Workflow
# ========================
def main_workflow():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    def preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.medianBlur(img, 3)
        return img
    
    # Filter data for binary classification
    train_mask = np.isin(y_train, CLASSES).flatten()
    test_mask = np.isin(y_test, CLASSES).flatten()

    x_train = np.array([preprocess(img) for img in x_train[train_mask][:SAMPLE_SIZE]])
    x_test = np.array([preprocess(img) for img in x_test[test_mask][:TEST_SIZE]])
    y_train = (y_train[train_mask][:SAMPLE_SIZE] == CLASSES[1]).astype(int).ravel()
    y_test = (y_test[test_mask][:TEST_SIZE] == CLASSES[1]).astype(int).ravel()

    logger.info("===== Feature Extraction Started =====")
    start_feature = time.time()
    train_features = process_batch(x_train)
    test_features = process_batch(x_test)
    feature_time = time.time() - start_feature
    
    results = {
        'K': [],
        'Feature_Time': [],
        'KMeans_Time': [],
        'SVM_Train_Time': [],
        'Total_Time': [],
        'Train_Accuracy': [],
        'Test_Accuracy': [],
        'Confusion_Matrix': [],
        'CPU_Time': [],
        'RAM_MB': []
    }
    
    for current_k in K_VALUES:
        logger.info(f"\n===== Evaluating K={current_k} =====")
        iteration_start = time.time()
        process = psutil.Process()
        
        try:
            # KMeans timing
            kmeans_start = time.process_time()
            kmeans = MiniBatchKMeans(
                n_clusters=current_k,
                batch_size=4096,
                random_state=42,
                n_init='auto'
            ).fit(np.vstack(train_features))
            kmeans_time = time.process_time() - kmeans_start
            
            # BOW feature generation
            def bow_transform(features):
                return np.array([
                    np.bincount(kmeans.predict(f), minlength=current_k)
                    if f.shape[0] > 0 else np.zeros(current_k)
                    for f in features
                ])
            
            X_train = bow_transform(train_features)
            X_test = bow_transform(test_features)
            
            # Feature normalization
            X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
            X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)
            
            # SVM training timing
            svm_start = time.process_time()
            svm = SVC(
                C=1,
                kernel='linear',
                class_weight='balanced',
                gamma='scale'
            ).fit(X_train, y_train)
            svm_time = time.process_time() - svm_start
            
            # Model evaluation
            y_train_pred = svm.predict(X_train)
            y_test_pred = svm.predict(X_test)
            
            # Resource monitoring
            mem_info = process.memory_info()
            results['K'].append(current_k)
            results['Feature_Time'].append(feature_time)
            results['KMeans_Time'].append(kmeans_time)
            results['SVM_Train_Time'].append(svm_time)
            results['Total_Time'].append(time.time() - iteration_start)
            results['Train_Accuracy'].append(accuracy_score(y_train, y_train_pred))
            results['Test_Accuracy'].append(accuracy_score(y_test, y_test_pred))
            results['Confusion_Matrix'].append(confusion_matrix(y_test, y_test_pred))
            results['CPU_Time'].append(kmeans_time + svm_time)
            results['RAM_MB'].append(mem_info.rss / (1024 ** 2))
            
            logger.info(
                f"K={current_k} | "
                f"Feature: {feature_time:.1f}s | "
                f"KMeans: {kmeans_time:.1f}s (CPU) | "
                f"SVM: {svm_time:.1f}s (CPU) | "
                f"RAM: {mem_info.rss/(1024**2):.1f}MB | "
                f"Total: {results['Total_Time'][-1]:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for K={current_k}: {str(e)}")
            continue
            
    return results

# ========================
# Enhanced Visualization
# ========================
def plot_results(results):
    df = pd.DataFrame(results)
    os.makedirs('analysis', exist_ok=True)
    
    # 1. 时间消耗曲线（平滑处理）
    plt.figure(figsize=(12, 6))
    x_new = np.linspace(min(df['K']), max(df['K']), 300)
    spl = make_interp_spline(df['K'], df['Total_Time'], k=3)
    y_smooth = spl(x_new)
    
    plt.plot(x_new, y_smooth, color='purple', linewidth=2)
    plt.scatter(df['K'], df['Total_Time'], color='darkred', zorder=5)
    plt.title('Total Time Consumption Curve')
    plt.xlabel('Visual Dictionary Size (K)')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis/time_consumption_curve.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. 时间分布饼图
    time_data = {
        'Feature Extraction': df['Feature_Time'].mean(),
        'KMeans Clustering': df['KMeans_Time'].sum(),
        'SVM Training': df['SVM_Train_Time'].sum()
    }
    
    plt.figure(figsize=(8, 8))
    plt.pie(time_data.values(), labels=time_data.keys(),
            autopct='%1.1f%%', startangle=90,
            colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Time Distribution Across Stages')
    plt.savefig('analysis/time_distribution_pie.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. 资源消耗曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df['K'], df['CPU_Time'], label='CPU Time (s)', marker='o')
    plt.plot(df['K'], df['RAM_MB'], label='RAM Usage (MB)', marker='s')
    plt.title('Resource Consumption by K-value')
    plt.xlabel('Visual Dictionary Size (K)')
    plt.ylabel('Usage')
    plt.legend()
    plt.grid(True)
    plt.savefig('analysis/resource_consumption.pdf', bbox_inches='tight')
    plt.close()
    
    # 5. Accuracy vs K 曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df['K'], df['Train_Accuracy'], label='Train Accuracy', marker='o')
    plt.plot(df['K'], df['Test_Accuracy'], label='Test Accuracy', marker='s')
    plt.title('Classification Accuracy vs K')
    plt.xlabel('Visual Dictionary Size (K)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('analysis/accuracy_vs_k.pdf', bbox_inches='tight')
    plt.close()

    # 4. 混淆矩阵对比
    max_test_idx = df['Test_Accuracy'].idxmax()
    min_test_idx = df['Test_Accuracy'].idxmin()
    
    cases = {
        'optimal': df.loc[max_test_idx, 'K'],
        'underfit': df.loc[min_test_idx, 'K']
    }
    
    for case_name, k in cases.items():
        idx = df.index[df['K'] == k].tolist()[0]
        cm = df.at[idx, 'Confusion_Matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    cbar=False, annot_kws={'size': 12})
        plt.title(f'Confusion Matrix (K={k})\nTest Accuracy: {df.at[idx, "Test_Accuracy"]:.3f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'analysis/confusion_matrix_{case_name}_k{k}.pdf', bbox_inches='tight')
        plt.close()

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    logger.info("===== Analysis Started =====")
    start_time = time.time()
    
    try:
        results = main_workflow()
        
        if results['K']:
            pd.DataFrame(results).to_csv('harris_results.csv', index=False)
            plot_results(results)
            
            best_idx = np.argmax(results['Test_Accuracy'])
            logger.info("\n===== Optimal Results =====")
            logger.info(f"Optimal K-value: {results['K'][best_idx]}")
            logger.info(f"Maximum Test Accuracy: {results['Test_Accuracy'][best_idx]:.3f}")
            
            if results['Test_Accuracy'][-1] < results['Test_Accuracy'][best_idx]:
                logger.warning(f"Overfitting detected at K={results['K'][-1]} "
                              f"(Accuracy drop: {results['Test_Accuracy'][best_idx]-results['Test_Accuracy'][-1]:.3f})")
        else:
            logger.error("No valid results obtained")
            
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}", exc_info=True)
    finally:
        logger.info(f"Total execution time: {time.time()-start_time:.1f} seconds")