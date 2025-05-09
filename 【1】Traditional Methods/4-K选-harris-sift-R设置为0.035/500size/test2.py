# ========================
# Harris-SIFT-SVM Analysis with Overfitting Detection
# ========================
import os
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

# ========================
# Environment Configuration
# ========================
os.environ["OMP_NUM_THREADS"] = "6"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Font configuration
def configure_fonts():
    # Get system available fonts
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
    
    # Font priority list (platform independent)
    font_priority = [
        'DejaVu Sans',       # Linux/Unix
        'Arial',             # Windows/Mac
        'Liberation Sans',   # Red Hat
        'Tahoma',            # Windows fallback
        'Verdana',           # Universal
        'sans-serif'         # Generic fallback
    ]
    
    # Select first available font
    selected_font = next((f for f in font_priority if f in available_fonts), 'sans-serif')
    
    # Configure matplotlib
    plt.rcParams.update({
        'figure.dpi': 300,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'font.family': selected_font,
        'axes.unicode_minus': False
    })
    
    return selected_font

# Initialize font configuration
selected_font = configure_fonts()

# ========================
# Parameter Configuration
# ========================
CLASSES = 2
#SAMPLE_SIZE = 2000
SAMPLE_SIZE = 500
#TEST_SIZE = 500
TEST_SIZE = 100
#K_VALUES = [50, 150, 300, 500, 800, 1000, 1300]
K_VALUES = [50, 150, 300, 500, 800]
IMG_SIZE = (128, 128)
MAX_FEATURES = 800

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
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
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
        threshold = 0.035 * dst_norm.max()
        
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
    train_mask = (y_train[:,0] < CLASSES).flatten()
    test_mask = (y_test[:,0] < CLASSES).flatten()
    
    x_train = np.array([preprocess(img) for img in x_train[train_mask][:SAMPLE_SIZE]])
    x_test = np.array([preprocess(img) for img in x_test[test_mask][:TEST_SIZE]])
    y_train = y_train[train_mask][:SAMPLE_SIZE].ravel()
    y_test = y_test[test_mask][:TEST_SIZE].ravel()
    
    logger.info("===== Feature Extraction Started =====")
    train_features = process_batch(x_train)
    test_features = process_batch(x_test)
    
    results = {
        'K': [],
        'Train_Accuracy': [],
        'Test_Accuracy': [],
        'Training_Time': [],
        'Confusion_Matrix': []
    }
    
    for current_k in K_VALUES:
        logger.info(f"\n===== Evaluating K={current_k} =====")
        start_time = time.time()
        
        try:
            # Visual vocabulary construction
            kmeans = MiniBatchKMeans(
                n_clusters=current_k,
                batch_size=4096,
                random_state=42,
                n_init='auto'
            ).fit(np.vstack(train_features))
            
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
            
            # Model training
            train_start = time.time()
            svm = SVC(
                C=0.7,
                kernel='rbf',
                class_weight='balanced',
                gamma='scale'
            ).fit(X_train, y_train)
            train_time = time.time() - train_start
            
            # Model evaluation
            y_train_pred = svm.predict(X_train)
            y_test_pred = svm.predict(X_test)
            
            results['K'].append(current_k)
            results['Train_Accuracy'].append(accuracy_score(y_train, y_train_pred))
            results['Test_Accuracy'].append(accuracy_score(y_test, y_test_pred))
            results['Training_Time'].append(train_time)
            results['Confusion_Matrix'].append(confusion_matrix(y_test, y_test_pred))
            
            logger.info(
                f"K={current_k} | "
                f"Train Acc: {results['Train_Accuracy'][-1]:.3f} | "
                f"Test Acc: {results['Test_Accuracy'][-1]:.3f} | "
                f"Train Time: {train_time:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for K={current_k}: {str(e)}")
            continue
            
    return results

# ========================
# Visualization
# ========================
def plot_results(results):
    df = pd.DataFrame(results)
    os.makedirs('analysis', exist_ok=True)
    
    # 新增时间消耗曲线 ---------------------------------
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(x='K', y='Training_Time', data=df, 
                     marker='D', linewidth=2, color='purple')
    
    # 标注最大时间点
    max_time_idx = df['Training_Time'].idxmax()
    max_k = df.loc[max_time_idx, 'K']
    max_time = df.loc[max_time_idx, 'Training_Time']
    
    ax.annotate(f'Max Training Time: {max_time:.1f}s\nat K={max_k}',
                xy=(max_k, max_time),
                xytext=(max_k+50, max_time+5),
                arrowprops=dict(arrowstyle="->", color='darkred'))
    
    plt.title('Training Time by Visual Dictionary Size')
    plt.xlabel('Visual Dictionary Size (K)')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.savefig('analysis/training_time_curve.pdf', bbox_inches='tight')
    plt.close()
    
    # Accuracy curves
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='K', y='Train_Accuracy', data=df, 
                 label='Training Accuracy', marker='o', linewidth=2)
    sns.lineplot(x='K', y='Test_Accuracy', data=df,
                 label='Test Accuracy', marker='s', linewidth=2)
    
    max_test_idx = df['Test_Accuracy'].idxmax()
    plt.axvline(x=df.loc[max_test_idx, 'K'], color='r', linestyle='--', alpha=0.7)
    plt.annotate(f'Peak Performance at K={df.loc[max_test_idx, "K"]}\nTest Accuracy: {df.loc[max_test_idx, "Test_Accuracy"]:.3f}',
                 xy=(df.loc[max_test_idx, 'K'], df.loc[max_test_idx, 'Test_Accuracy']),
                 xytext=(20, -40), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='r'))
    
    plt.title('Training vs Test Accuracy by Vocabulary Size')
    plt.xlabel('Visual Dictionary Size (K)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis/accuracy_curves.pdf', bbox_inches='tight')
    plt.close()
    
    # Confusion matrices
    key_points = {
        'best_performance': df.loc[max_test_idx, 'K'],
        'overfitting_case': 1300 if 1300 in df['K'].values else None
    }
    
    for case_name, k in key_points.items():
        if k is None:
            continue
            
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