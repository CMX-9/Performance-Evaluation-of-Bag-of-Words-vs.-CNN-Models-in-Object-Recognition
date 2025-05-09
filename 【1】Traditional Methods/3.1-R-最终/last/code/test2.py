import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["OMP_NUM_THREADS"] = "8"  # 限制线程数，防止线程过多引发内存问题
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
from datetime import datetime
import sys
import json
from mpl_toolkits.mplot3d import Axes3D

# ==================== Global Configurations ====================
FIXED_K = 200  # Fixed vocabulary size
EXP_CONFIGS = [
    {'SAMPLE_SIZE': 500,  'MAX_K': FIXED_K},
    {'SAMPLE_SIZE': 1000,  'MAX_K': FIXED_K},
    {'SAMPLE_SIZE': 1500,  'MAX_K': FIXED_K},
    {'SAMPLE_SIZE': 2000,  'MAX_K': FIXED_K}
]
R_VALUES = [ 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
TEST_SIZE = 500
BASE_IMG_SIZE = (128, 128)

# ==================== Utility Functions ====================
class DualLogger:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log_file = file
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def setup_logger():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/fixed_k{FIXED_K}_exp_{timestamp}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = DualLogger(log_file)
    print(f"📄 Logging started: {log_path}")
    return log_path

# ==================== Core Logic ====================
def enhanced_feature_extraction(img, r_thresh):
    """Feature extraction with Harris filtering"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Harris corner detection
    dst = cv2.cornerHarris(gray.astype(np.float32), 3, 5, 0.04)
    y, x = np.where(dst > r_thresh * dst.max())
    
    # Non-Maximum Suppression
    dst = cv2.dilate(dst, None)
    mask = (dst == cv2.erode(dst, None))  # Local maxima
    valid_points = np.logical_and(dst > r_thresh * dst.max(), mask)
    y, x = np.where(valid_points)

    # Keypoint generation
    kps = [cv2.KeyPoint(float(x[i]), float(y[i]), 10)
           for i in range(min(200, len(x)))]
    
    if not kps:
        return np.zeros((1, 128))

    # SIFT descriptor extraction
    _, desc = cv2.SIFT_create().compute(gray, kps)
    return desc if desc is not None else np.zeros((1, 128))

def execute_experiment(config):
    """Execute single experiment with fixed K"""
    print(f"\n🚀 Starting experiment: SampleSize={config['SAMPLE_SIZE']}, FixedK={FIXED_K}")
    
    # Data loading and preprocessing
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_mask = y_train[:, 0] < 2
    test_mask = y_test[:, 0] < 2
    
    train_data = [cv2.resize(img, BASE_IMG_SIZE) 
                 for img in x_train[train_mask][:config['SAMPLE_SIZE']]]
    test_data = [cv2.resize(img, BASE_IMG_SIZE)
                for img in x_test[test_mask][:TEST_SIZE]]
    
    train_labels = y_train[train_mask][:config['SAMPLE_SIZE']].ravel()
    test_labels = y_test[test_mask][:TEST_SIZE].ravel()

    results = []
    
    for r in R_VALUES:
        print(f"\n🔍 Evaluating R = {r:.3f}...")
        
        # Feature extraction
        train_features = [enhanced_feature_extraction(img, r) for img in train_data]
        train_features = [f for f in train_features if f.shape[0] > 0]
        
        # Vocabulary construction
        all_descriptors = np.vstack(train_features)
        actual_k = min(len(all_descriptors), config['MAX_K'])
        
        if actual_k < FIXED_K:
            print(f"⚠️ Warning: Insufficient features ({len(all_descriptors)}) for K={FIXED_K}")
        
        #kmeans = MiniBatchKMeans(n_clusters=actual_k, batch_size=4096)
        kmeans = KMeans(n_clusters=actual_k, random_state=42)
        kmeans.fit(all_descriptors)
        
        # Feature encoding
        def bow_encoder(feat):
            return np.bincount(kmeans.predict(feat), minlength=actual_k)
        
        X_train = np.array([bow_encoder(f) for f in train_features])
        X_test = np.array([bow_encoder(enhanced_feature_extraction(img, r)) for img in test_data])
        
        # Classification
        svm = SVC(kernel='linear').fit(X_train, train_labels[:len(X_train)])
        y_pred = svm.predict(X_test)
        acc = accuracy_score(test_labels[:len(X_test)], y_pred)
        
        results.append({
            'R': r,
            'Accuracy': acc,
            'AvgFeatures': np.mean([f.shape[0] for f in train_features]),
            'TotalFeatures': len(all_descriptors),
            'ActualK': actual_k
        })
        print(f"✅ R={r:.3f} | Accuracy={acc:.4f} | AvgFeatures={results[-1]['AvgFeatures']:.1f}")

    # Save results
    filename = f"fixed_k{FIXED_K}_N{config['SAMPLE_SIZE']}_results.json"
    with open(filename, 'w') as f:
        json.dump({'config': config, 'results': results}, f)
    
    return config, results

# ==================== Visualization ====================
def plot_comparison(all_results):
    """Comparative analysis of different sample sizes"""
    plt.figure(figsize=(12, 6))
    color_map = plt.get_cmap('tab10')
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    
    for idx, (config, results) in enumerate(all_results):
        r_values = [r['R'] for r in results]
        accuracies = [r['Accuracy'] for r in results]
        
        # 获取最高准确度的点
        max_idx = np.argmax(accuracies)
        best_r = r_values[max_idx]
        best_acc = accuracies[max_idx]
        
        # 绘制曲线
        plt.plot(
            r_values, accuracies,
            color=color_map(idx),
            linestyle=line_styles[idx % len(line_styles)],
            marker=markers[idx % len(markers)],
            label=f"N={config['SAMPLE_SIZE']} (K={FIXED_K})"
        )

        # 标记最佳点
        plt.scatter(best_r, best_acc, color=color_map(idx), edgecolors='k', zorder=5)
        plt.text(best_r, best_acc, f"★ {best_acc:.3f}", fontsize=10, ha='left', va='bottom')

    plt.xlabel("R Threshold Value", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.title(f"Fixed K={FIXED_K} Performance Comparison", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"fixed_k{FIXED_K}_comparison.png", dpi=300)
    plt.close()

def plot_r_feature_relationship(all_results):
    """R value vs Average Features analysis"""
    plt.figure(figsize=(12, 6))
    
    for config, results in all_results:
        r_values = [r['R'] for r in results]
        avg_features = [r['AvgFeatures'] for r in results]
        plt.plot(r_values, avg_features, 
                 marker='o',
                 linestyle='--',
                 label=f"N={config['SAMPLE_SIZE']}")
    
    plt.xlabel("R Threshold Value", fontsize=12)
    plt.ylabel("Average Features per Image", fontsize=12)
    plt.title(f"Feature Extraction Efficiency (Fixed K={FIXED_K})", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"fixed_k{FIXED_K}_feature_analysis.png", dpi=300)
    plt.close()

def plot_3d_analysis(all_results):
    """3D parameter space visualization"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data arrays
    sample_sizes = []
    r_values = []
    accuracies = []
    
    for config, results in all_results:
        for result in results:
            sample_sizes.append(config['SAMPLE_SIZE'])
            r_values.append(result['R'])
            accuracies.append(result['Accuracy'])
    
    # Create surface plot
    surf = ax.plot_trisurf(sample_sizes, r_values, accuracies,
                          cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.set_xlabel("Sample Size", fontsize=12)
    ax.set_ylabel("R Threshold", fontsize=12)
    ax.set_zlabel("Accuracy", fontsize=12)
    ax.set_title(f"3D Parameter Space Analysis (Fixed K={FIXED_K})", fontsize=14)
    plt.savefig(f"fixed_k{FIXED_K}_3d_analysis.png", dpi=300)
    plt.close()

# ==================== Main Execution ====================
if __name__ == "__main__":
    log_file = setup_logger()
    all_results = []
    
    try:
        # Run all experiments
        for config in EXP_CONFIGS:
            results = execute_experiment(config)
            all_results.append(results)
        
        # Generate visualizations
        plot_comparison(all_results)
        plot_r_feature_relationship(all_results)
        plot_3d_analysis(all_results)
        
    except Exception as e:
        print(f"❌ Experiment terminated: {str(e)}")
    finally:
        print(f"\n📝 All outputs saved to: {log_file}")
