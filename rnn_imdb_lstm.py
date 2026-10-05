"""TECH 405 RNN Hackathon - IMDB sentiment classification with nn.LSTM (PyTorch)."""
import os, re, json, time, random, urllib.request
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

SEED, VOCAB, MAXLEN = 42, 10000, 200
EMB, HIDDEN, BATCH, EPOCHS, LR = 32, 64, 64, int(os.environ.get("EPOCHS", 6)), 2e-3
BASELINE, TARGET = 50.0, 82.0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs("out", exist_ok=True)
open("out/log.txt", "w").close()
def log(text, name="log"):
    print(text, flush=True)
    with open(f"out/{name}.txt", "a") as f: f.write(text + "\n")

# 1. Load data (public GitHub CSV)
URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
if not os.path.exists("IMDB-Dataset.csv"):
    req = urllib.request.Request(URL, headers={"User-Agent": "M"})
    open("IMDB-Dataset.csv", "wb").write(urllib.request.urlopen(req).read())
df = pd.read_csv("IMDB-Dataset.csv")                      # columns: review, sentiment
y_all = (df["sentiment"] == "positive").astype(int).values
tok = lambda s: re.findall(r"[a-z0-9']+", s.lower().replace("<br />", " "))
texts = [tok(s) for s in df["review"]]

# 2. Fixed 80/20 split (seed 42). Vocabulary is built from the training part only.
idx_tr, idx_te = train_test_split(np.arange(len(texts)), test_size=0.2, stratify=y_all, random_state=SEED)
counts = Counter(w for i in idx_tr for w in texts[i])
vocab = {w: k + 2 for k, (w, _) in enumerate(counts.most_common(VOCAB - 2))}   # 0 = pad, 1 = unknown
def encode(ix):
    X = np.zeros((len(ix), MAXLEN), dtype=np.int64)
    for r, i in enumerate(ix):
        s = [vocab.get(w, 1) for w in texts[i]][-MAXLEN:]   # keep last 200 tokens
        if s: X[r, -len(s):] = s                            # pre-padding
    return torch.tensor(X), torch.tensor(y_all[ix])
X_tr, y_tr = encode(idx_tr); X_te, y_te = encode(idx_te)
# hold out 10% of the TRAINING data for validation (test set is never used for tuning)
i_fit, i_val = train_test_split(np.arange(len(X_tr)), test_size=0.1, stratify=y_tr.numpy(), random_state=SEED)
X_fit, y_fit, X_val, y_val = X_tr[i_fit], y_tr[i_fit], X_tr[i_val], y_tr[i_val]
log("Dataset: IMDB movie reviews (sentiment: positive / negative)", "data")
log(f"Total reviews: {len(df)}  (positive: {y_all.sum()}, negative: {(y_all==0).sum()})", "data")
log(f"Train (fit): X {tuple(X_fit.shape)}  y {tuple(y_fit.shape)}", "data")
log(f"Validation : X {tuple(X_val.shape)}  y {tuple(y_val.shape)}", "data")
log(f"Test       : X {tuple(X_te.shape)}  y {tuple(y_te.shape)}", "data")
log(f"Vocabulary: {VOCAB}   MAXLEN: {MAXLEN}   Test positive share: {y_te.float().mean():.2f}", "data")
log(f"Baseline (majority class): {BASELINE:.1f}%   Target: {TARGET:.1f}%", "data")

# 3. Model: Embedding -> nn.LSTM -> nn.Linear
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, EMB, padding_idx=0)
        self.lstm = nn.LSTM(EMB, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))     # h: (1, batch, HIDDEN) = last hidden state
        return self.fc(h[-1])
model = LSTMClassifier()
torch.set_num_threads(1)
log(str(model), "model")
log(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}", "model")
opt = torch.optim.Adam(model.parameters(), lr=LR); lossf = nn.CrossEntropyLoss()

def evaluate(X, y):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(X), 512):
            correct += (model(X[i:i+512]).argmax(1) == y[i:i+512]).sum().item()
    return 100 * correct / len(X)

# 4. Train (best epoch chosen by VALIDATION accuracy)
hist = dict(loss=[], train_acc=[], val_acc=[]); best_val, best_state = -1, None
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); perm = torch.randperm(len(X_fit)); tot, n, corr = 0.0, 0, 0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i+BATCH]; xb, yb = X_fit[b], y_fit[b]
        opt.zero_grad(); out = model(xb); loss = lossf(out, yb); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item() * len(b); n += len(b); corr += (out.argmax(1) == yb).sum().item()
    va = evaluate(X_val, y_val)
    hist["loss"].append(tot / n); hist["train_acc"].append(100 * corr / n); hist["val_acc"].append(va)
    if va > best_val: best_val, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    log(f"Epoch {ep}/{EPOCHS}  train_loss={tot/n:.4f}  train_acc={100*corr/n:.2f}%  val_acc={va:.2f}%  ({time.time()-t0:.0f}s)", "train")
json.dump(hist, open("out/history.json", "w"))

# 5. Final TEST accuracy (once, using the epoch with the best validation accuracy)
model.load_state_dict(best_state)
test_acc = evaluate(X_te, y_te)
log(f"Best validation accuracy: {best_val:.2f}% (epoch {int(np.argmax(hist['val_acc']))+1})", "final")
log(f"TEST ACCURACY: {test_acc:.2f}%", "final")
log(f"Baseline: {BASELINE:.1f}%   Target: {TARGET:.1f}%   Ours: {test_acc:.2f}%", "final")
log(f"Result: {'target reached' if test_acc >= TARGET else 'between baseline and target' if test_acc > BASELINE else 'at or below baseline'}", "final")

# 6. Plots
ep = range(1, EPOCHS + 1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ep, hist["loss"], marker="o", color="#c0392b"); ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
ax.set_title("Training loss per epoch"); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig("out/loss_curve.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Baseline", "Our LSTM", "Target"], [BASELINE, test_acc, TARGET], color=["#190404", "#2e75b6", "#27ae60"])
for b in bars: ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{b.get_height():.1f}%", ha="center")
ax.set_ylim(0, 100); ax.set_ylabel("Test accuracy (%)"); ax.set_title("Test accuracy vs baseline and target")
plt.tight_layout(); plt.savefig("out/accuracy_bar.png", dpi=150); plt.close()
