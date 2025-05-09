# ========================
# Harris-SIFT Optimal K Analysis (CIFAR-10)
# ========================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OMP_NUM_THREADS"] = "8"  # Limit threads to prevent memory issues
import time
import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
from datetime import datetime
import psutil
from tqdm import tqdm

# ========================
# Experiment Configuration
# ========================
EXPERIMENT = {
    'name': 'Optimal K-value Analysis',
    'K_values': [100, 200, 300, 500, 1000],  # Test range
    'fixed_R': 0.03,                   # Harris threshold
    'N_train': 2000,                   # Training samples
    'N_test': 500,                     # Test samples
    'classes': [1, 8],                 # automobile=1, truck=8
    'n_init': 10,                      # K-means initializations
    'max_features': 800                # Max features per image
}

# ========================
# Core Components
# ========================
class HarrisSIFTFeatureExtractor:
    def __init__(self, R=0.03):
        self.R = R
        self.img_size = (128, 128)
        
    def __call__(self, img):
        """Feature extractor with NMS implementation"""
        try:
            # Image preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            gray = cv2.resize(gray, self.img_size)
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (5,5), 1.5)
            
            # Harris corner detection
            dst = cv2.cornerHarris(gray.astype(np.float32), 3, 5, 0.04)
            
            # NMS implementation
            dst_norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(dst_norm, kernel)
            local_max = (dst_norm == dilated)
            valid_points = np.logical_and(dst_norm > self.R*255, local_max)
            y, x = np.where(valid_points)
            
            # Keypoint generation
            keypoints = sorted([(x[i], y[i], dst_norm[y[i], x[i]]) 
                              for i in range(len(x))], key=lambda k: -k[2])
            keypoints = keypoints[:EXPERIMENT['max_features']]
            
            # SIFT descriptor extraction
            sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=15)
            kps = [cv2.KeyPoint(float(x), float(y), float(20 * (r / 255) ** 0.5)) for (x, y, r) in keypoints]
            _, desc = sift.compute(gray, kps)
            return desc if desc is not None else np.zeros((1,128))
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {str(e)}")
            return np.zeros((1,128))

class OptimalKAnalyzer:
    def __init__(self):
        self.logger = self._setup_logger()
        self.feature_extractor = HarrisSIFTFeatureExtractor(EXPERIMENT['fixed_R'])
        
    def _setup_logger(self):
        """Configure logging system"""
        logger = logging.getLogger('KAnalysis')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(
            f"logs/k_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
    
    def load_cifar_data(self):
        """Load and preprocess CIFAR-10 data"""
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Create binary classification masks
        class_mask = lambda y: np.isin(y[:,0], EXPERIMENT['classes'])
        train_mask = class_mask(y_train)
        test_mask = class_mask(y_test)
        
        return (
            x_train[train_mask][:EXPERIMENT['N_train']],
            (y_train[train_mask][:EXPERIMENT['N_train']] == EXPERIMENT['classes'][1]).astype(int).ravel(),
            x_test[test_mask][:EXPERIMENT['N_test']],
            (y_test[test_mask][:EXPERIMENT['N_test']] == EXPERIMENT['classes'][1]).astype(int).ravel()
        )
    
    def run_analysis(self):
        """Execute full analysis pipeline"""
        results = []
        x_train, y_train, x_test, y_test = self.load_cifar_data()
        
        # Feature extraction phase
        feature_start = time.time()
        train_features = []
        for img in tqdm(x_train, desc="Feature Extraction"):
            feat = self.feature_extractor(img)
            if feat.shape[0] > 0:
                train_features.append(feat)
        feature_time = time.time() - feature_start
        
        if len(train_features) == 0:
            raise ValueError("No valid features extracted! Check parameters.")
        
        for K in EXPERIMENT['K_values']:
            self.logger.info(f"\n===== Processing K={K} =====")
            iteration_start = time.time()
            
            # K-Means clustering
            kmeans_start = time.time()
            # kmeans = KMeans(
                # n_clusters=K,
                # random_state=42,
                # n_init=EXPERIMENT['n_init']
            # ).fit(np.vstack(train_features))
            # 替换聚类方式：
            kmeans = MiniBatchKMeans(
                n_clusters=K,
                random_state=42,
                batch_size=4096,
                n_init=1,
                max_iter=100
            ).fit(np.vstack(train_features))
            kmeans_time = time.time() - kmeans_start
            
            # BOW feature generation
            bow_start = time.time()
            def bow_transform(features):
                return np.array([
                    np.bincount(kmeans.predict(f), minlength=K) if f.shape[0] > 0 else np.zeros(K)
                    for f in features
                ])
            X_train = bow_transform(train_features)
            X_test = bow_transform([self.feature_extractor(img) for img in x_test])
            bow_time = time.time() - bow_start
            
            # SVM training
            svm_start = time.time()
            svm = SVC(C=1.0, kernel='linear').fit(X_train, y_train)
            svm_time = time.time() - svm_start
            
            # Model evaluation
            eval_start = time.time()
            test_acc = accuracy_score(y_test, svm.predict(X_test))
            train_acc = accuracy_score(y_train, svm.predict(X_train))
            eval_time = time.time() - eval_start
            
            # Resource monitoring
            process = psutil.Process(os.getpid())
            mem_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            results.append({
                'K': K,
                'Feature_Time': feature_time,
                'KMeans_Time': kmeans_time,
                'BOW_Time': bow_time,
                'SVM_Train_Time': svm_time,
                'Eval_Time': eval_time,
                'Total_Time': time.time() - iteration_start,
                'Full_Total_Time': feature_time + (time.time() - iteration_start),
                'Memory_MB': mem_usage,
                'Test_Accuracy': test_acc,
                'Train_Accuracy': train_acc,
                'Avg_Features': np.mean([f.shape[0] for f in train_features])
            })
            
            self.logger.info(
                f"K={K} | "
                f"Features: {feature_time:.1f}s | "
                f"MiniBatchKMeans: {kmeans_time:.1f}s | "
                f"BOW: {bow_time:.1f}s | "
                f"SVM: {svm_time:.1f}s | "
                f"Total: {results[-1]['Total_Time']:.1f}s"
            )
        
        return pd.DataFrame(results)

# ========================
# Visualization
# ========================
def plot_full_analysis(results):
    """Generate comprehensive analysis plots"""
    plt.figure(figsize=(18, 12))
    
    # Accuracy Analysis
    plt.subplot(2, 2, 1)
    plt.plot(results['K'], results['Test_Accuracy'], 'o-', label='Test')
    plt.plot(results['K'], results['Train_Accuracy'], 's--', label='Train')
    plt.xlabel('Visual Dictionary Size (K)')
    plt.ylabel('Accuracy')
    plt.title('Classification Performance')
    plt.legend()
    
    # Time Breakdown
    plt.subplot(2, 2, 2)
    plt.stackplot(results['K'], 
                 results['KMeans_Time'], 
                 results['BOW_Time'],
                 results['SVM_Train_Time'],
                 labels=['K-Means', 'BOW', 'SVM Training'])
    plt.xlabel('Visual Dictionary Size (K)')
    plt.ylabel('Time (seconds)')
    plt.title('Time Consumption Breakdown')
    plt.legend()
    
    # Memory Usage
    plt.subplot(2, 2, 3)
    plt.plot(results['K'], results['Memory_MB'], 'o-', color='purple')
    plt.xlabel('Visual Dictionary Size (K)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Consumption Analysis')
    
    # Feature Efficiency
    plt.subplot(2, 2, 4)
    sc = plt.scatter(results['Avg_Features'], results['Test_Accuracy'], 
                    c=results['K'], cmap='viridis', s=100)
    plt.xlabel('Average Features per Image')
    plt.ylabel('Test Accuracy')
    plt.title('Feature Efficiency Analysis (Color = K-value)')
    plt.colorbar(sc, label='K-value')
    
    plt.tight_layout()
    plt.savefig('full_analysis.png', dpi=300)
    plt.close()

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    analyzer = OptimalKAnalyzer()
    
    print(f"\n=== Starting Experiment: {EXPERIMENT['name']} ===")
    print(f"Classes: {EXPERIMENT['classes']}")
    print(f"Testing K-values: {EXPERIMENT['K_values']}")
    print(f"Training Samples: {EXPERIMENT['N_train']}")
    print(f"Test Samples: {EXPERIMENT['N_test']}\n")
    
    try:
        results = analyzer.run_analysis()
        results.to_csv('optimal_k_results.csv', index=False)
        plot_full_analysis(results)
        
        best_k = results.loc[results['Test_Accuracy'].idxmax()]
        print(f"\n=== Final Results ===")
        print(f"Optimal K: {int(best_k['K'])}")
        print(f"Test Accuracy: {best_k['Test_Accuracy']:.4f}")
        print(f"Total Time: {best_k['Full_Total_Time']:.1f}s")
        print(f"Peak Memory: {best_k['Memory_MB']:.1f} MB")
        
    except Exception as e:
        print(f"\n!!! Experiment Terminated: {str(e)}")
    finally:
        print("\n=== Experiment Completed ===")