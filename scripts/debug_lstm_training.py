"""
Deep LSTM training debugging: check weight updates, loss trajectory, gradient flow.

Usage:
    python scripts/debug_lstm_training.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "backend"))

DATA_DIR = _ROOT / "data" / "processed"

# Load data
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

# ============================================================================
# 1. CHECK INITIALIZATION
# ============================================================================
print("=" * 70)
print("1. MODEL INITIALIZATION")
print("=" * 70)

from app.ml.prediction.models import PyTorchLSTMClassifierNet

device = "cpu"
net = PyTorchLSTMClassifierNet(n_sensors=6, hidden_dim=64, num_layers=2, dropout=0.25).to(device)

print("\nInitial weight statistics:")
for name, param in net.named_parameters():
    print(f"  {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, min={param.data.min():.6f}, max={param.data.max():.6f}")

# ============================================================================
# 2. FORWARD PASS ON BATCH BEFORE TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("2. FORWARD PASS (BEFORE TRAINING)")
print("=" * 70)

X_batch = torch.tensor(X_train[:128], dtype=torch.float32).to(device)
y_batch = torch.tensor(y_train[:128], dtype=torch.float32).to(device)

net.eval()
with torch.no_grad():
    logits_before = net(X_batch).cpu().numpy()
    print(f"\nLogits BEFORE training:")
    print(f"  Shape: {logits_before.shape}")
    print(f"  min={logits_before.min():.6f}, max={logits_before.max():.6f}, mean={logits_before.mean():.6f}, std={logits_before.std():.6f}")
    print(f"  First 10 logits: {logits_before[:10]}")
    probs_before = 1.0 / (1.0 + np.exp(-logits_before))
    print(f"  Probabilities min={probs_before.min():.6f}, max={probs_before.max():.6f}, mean={probs_before.mean():.6f}")

# ============================================================================
# 3. TRAIN FOR 1 EPOCH WITH DETAILED LOGGING
# ============================================================================
print("\n" + "=" * 70)
print("3. TRAINING (1 EPOCH WITH DETAILED LOGGING)")
print("=" * 70)

dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
loader = DataLoader(dataset, batch_size=128, shuffle=True)

pos_count = float(np.sum(y_train == 1))
neg_count = float(np.sum(y_train == 0))
pos_weight = torch.tensor([max(1.0, neg_count / max(pos_count, 1.0))]).to(device)

print(f"\nClass imbalance: pos={pos_count:.0f}, neg={neg_count:.0f}, pos_weight={pos_weight.item():.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(net.parameters(), lr=0.002)

net.train()

print(f"\nBatch-by-batch loss for epoch 1:")
total_loss = 0.0
num_batches = 0
losses_per_batch = []

for batch_idx, (bx, by) in enumerate(loader):
    bx, by = bx.to(device), by.to(device)
    
    # Before optimizer step
    optimizer.zero_grad()
    logits = net(bx)
    loss = criterion(logits, by)
    
    # Check gradients
    loss.backward()
    
    # Inspect gradient norms
    total_grad_norm = 0.0
    for param in net.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    optimizer.step()
    
    batch_loss = loss.item()
    total_loss += batch_loss * len(bx)
    num_batches += 1
    losses_per_batch.append(batch_loss)
    
    if batch_idx % 50 == 0:
        print(f"  Batch {batch_idx:3d}: loss={batch_loss:.6f}, grad_norm={total_grad_norm:.6f}, logits_mean={logits.mean():.6f}, logits_std={logits.std():.6f}")

avg_loss = total_loss / len(X_train)
print(f"\nEpoch 1 complete: avg_loss={avg_loss:.6f}, first_batch_loss={losses_per_batch[0]:.6f}, last_batch_loss={losses_per_batch[-1]:.6f}")

# ============================================================================
# 4. FORWARD PASS AFTER TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("4. FORWARD PASS (AFTER 1 EPOCH TRAINING)")
print("=" * 70)

net.eval()
with torch.no_grad():
    logits_after = net(X_batch).cpu().numpy()
    print(f"\nLogits AFTER training:")
    print(f"  Shape: {logits_after.shape}")
    print(f"  min={logits_after.min():.6f}, max={logits_after.max():.6f}, mean={logits_after.mean():.6f}, std={logits_after.std():.6f}")
    print(f"  First 10 logits: {logits_after[:10]}")
    probs_after = 1.0 / (1.0 + np.exp(-logits_after))
    print(f"  Probabilities min={probs_after.min():.6f}, max={probs_after.max():.6f}, mean={probs_after.mean():.6f}")
    
    logit_change = np.abs(logits_after - logits_before).mean()
    print(f"\nMean absolute logit change: {logit_change:.6f}")

# ============================================================================
# 5. WEIGHT STATISTICS AFTER TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("5. WEIGHT STATISTICS (AFTER TRAINING)")
print("=" * 70)

print("\nUpdated weight statistics:")
for name, param in net.named_parameters():
    print(f"  {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, min={param.data.min():.6f}, max={param.data.max():.6f}")

# ============================================================================
# 6. TRAIN FOR FULL 15 EPOCHS
# ============================================================================
print("\n" + "=" * 70)
print("6. TRAINING (15 FULL EPOCHS)")
print("=" * 70)

net.train()
for epoch in range(2, 16):
    total_loss = 0.0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = net(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(bx)
    
    avg_loss = total_loss / len(X_train)
    if epoch % 3 == 0:
        print(f"Epoch {epoch:2d}: avg_loss={avg_loss:.6f}")

# ============================================================================
# 7. FINAL EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("7. FINAL EVALUATION (AFTER 15 EPOCHS)")
print("=" * 70)

net.eval()
with torch.no_grad():
    X_full = torch.tensor(X_train, dtype=torch.float32).to(device)
    logits_final = net(X_full).cpu().numpy()
    print(f"\nFinal logits on full training set:")
    print(f"  min={logits_final.min():.6f}, max={logits_final.max():.6f}, mean={logits_final.mean():.6f}, std={logits_final.std():.6f}")
    
    probs_final = 1.0 / (1.0 + np.exp(-logits_final))
    print(f"\nFinal probabilities on full training set:")
    print(f"  min={probs_final.min():.6f}, max={probs_final.max():.6f}, mean={probs_final.mean():.6f}, std={probs_final.std():.6f}")
    
    preds = (probs_final >= 0.5).astype(int)
    print(f"\nFinal predictions (threshold=0.5):")
    print(f"  Predicted distribution: 0={np.sum(preds==0)}, 1={np.sum(preds==1)}")
    print(f"  True distribution: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
    print(f"  Accuracy: {(preds == y_train.astype(int)).mean():.4f}")

print("\n" + "=" * 70)
print("DEBUGGING COMPLETE")
print("=" * 70)
