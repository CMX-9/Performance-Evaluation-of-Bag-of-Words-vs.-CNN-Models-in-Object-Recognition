import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
from datetime import datetime
import sys
##set TF_ENABLE_ONEDNN_OPTS=0

# 配置
PHASE2_CONFIG = {
    'SAMPLE_SIZE': 2000,
    'R_VALUES': [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1],
    'MAX_K': 300
}

# 设置日志保存路径
def setup_logger():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/phase2_评估日志_{timestamp}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    class Logger(object):
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = Logger(log_file)
    print(f"📄 日志记录启动：{log_path}")
    return log_path

def safe_extract_features(img, r_thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dst = cv2.cornerHarris(gray.astype(np.float32), 3, 5, 0.04)
    y, x = np.where(dst > r_thresh * dst.max())
    h, w = gray.shape

    kps = [cv2.KeyPoint(float(x[i]), float(y[i]), 10)
           for i in range(min(200, len(x))) if 0 <= x[i] < w and 0 <= y[i] < h]

    if not kps:
        return np.zeros((1, 128))

    _, desc = cv2.SIFT_create().compute(gray, kps)
    return desc if desc is not None else np.zeros((1, 128))

def phase2_evaluate():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_data = [cv2.resize(img, (64, 64)) for img in x_train[y_train[:, 0] < 2][:PHASE2_CONFIG['SAMPLE_SIZE']]]
    test_data = [cv2.resize(img, (64, 64)) for img in x_test[y_test[:, 0] < 2][:500]]
    train_labels = y_train[y_train[:, 0] < 2][:PHASE2_CONFIG['SAMPLE_SIZE']].ravel()
    test_labels = y_test[y_test[:, 0] < 2][:500].ravel()

    results = []

    for r in PHASE2_CONFIG['R_VALUES']:
        print(f"\n🔎 评估 R = {r:.3f} ...")

        train_features = [safe_extract_features(img, r) for img in train_data]
        train_features = [f for f in train_features if f.shape[0] > 0]
        all_train_descriptors = np.vstack(train_features)
        total_k = min(len(all_train_descriptors), PHASE2_CONFIG['MAX_K'])

        kmeans = MiniBatchKMeans(n_clusters=total_k, batch_size=2048)
        kmeans.fit(all_train_descriptors)

        def bow(feat):
            return np.bincount(kmeans.predict(feat), minlength=total_k)

        X_train = np.array([bow(f) for f in train_features])
        X_test = np.array([bow(safe_extract_features(img, r)) for img in test_data])

        svm = SVC(kernel='linear').fit(X_train, train_labels[:len(X_train)])
        y_pred = svm.predict(X_test)
        acc = accuracy_score(test_labels[:len(X_test)], y_pred)

        results.append({
            'R': r,
            'Accuracy': acc,
            'AvgFeatures': np.mean([f.shape[0] for f in train_features]),
            'FeatureCount': len(train_features),
            'RealK': total_k
        })

        print(f"✅ R={r:.3f} | 准确率={acc:.4f} | 平均特征数={results[-1]['AvgFeatures']:.1f} | 实际K={total_k}")

    best = max(results, key=lambda x: x['Accuracy'])
    print(f"\n🏆 最优R值: {best['R']:.3f}，最高准确率: {best['Accuracy']:.4f}")
    return results,best

def plot_phase2_results(results,best):
    r_vals = [r['R'] for r in results]
    accs = [r['Accuracy'] for r in results]
    avg_feats = [r['AvgFeatures'] for r in results]
    feat_counts = [r['FeatureCount'] for r in results]
    real_ks = [r['RealK'] for r in results]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(r_vals, accs, 'bo-')
    axs[0].fill_between(r_vals, accs, alpha=0.1, color='blue')
    axs[0].set_xlabel("R Value")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy vs R")
    axs[0].legend()
    
    # 标注最优点
    axs[0].annotate(f"Max R = {best['R']:.3f}\nAccuracy = {best['Accuracy']:.4f}",
                    xy=(best['R'], best['Accuracy']),  # 标注点的坐标
                    xytext=(best['R'] + 0.01, best['Accuracy'] - 0.02),  # 注释文字的位置
                    arrowprops=dict(facecolor='red', shrink=0.05),  # 箭头的属性
                    fontsize=10, color='red')  # 标注文字的样式

    axs[1].plot(r_vals, avg_feats, 'ms--', label='Avg Features')
    axs[1].set_xlabel("R Value")
    axs[1].set_ylabel("Average Features per Image")
    axs[1].set_title("R vs Average Features")
    axs[1].legend()


    sc = axs[2].scatter(avg_feats, accs, c=r_vals, cmap='viridis', s=100)
    axs[2].set_xlabel("Average Features")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_title("Accuracy vs Avg Features (Color=R)")
    fig.colorbar(sc, ax=axs[2], label='R Value')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_path = setup_logger()
    results,best = phase2_evaluate()
    plot_phase2_results(results,best)
    print(f"\n📝 所有输出已保存至日志文件：{log_path}")
