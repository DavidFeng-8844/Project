import pandas as pd

df = pd.read_csv('/home/david/srtp/mcnn-pytorch-train/src/oil_leak/runs/detect/train10-csdn/results.csv')
df.columns = df.columns.str.strip()

p = df['metrics/precision(B)'].values
r = df['metrics/recall(B)'].values
f1 = 2 * (p * r) / (p + r + 1e-16)
best_idx = f1.argmax()

print(f'Best Epoch: {best_idx+1}')
print(f'Precision: {p[best_idx]:.4f}')
print(f'Recall: {r[best_idx]:.4f}')
print(f'mAP50: {df["metrics/mAP50(B)"].values[best_idx]:.4f}')
print(f'mAP50-95: {df["metrics/mAP50-95(B)"].values[best_idx]:.4f}')
print(f'Best F1: {f1[best_idx]:.4f}')
