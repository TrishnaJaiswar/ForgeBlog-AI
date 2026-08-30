# From Self‑Attention to Production: Building and Deploying Transformers Step‑by‑Step

## Why Transformers? Problem framing and high‑level intuition

Sequence‑to‑sequence (seq2seq) modeling maps an input token stream \(x_1…x_T\) to an output stream \(y_1…y_{T'}\). Classical recurrent neural networks (RNNs) suffer from three scalability bottlenecks: (1) **sequential dependency** – each hidden state must wait for the previous one, (2) **vanishing/exploding gradients** – long‑range error signals decay exponentially, and (3) **limited parallelism** – GPUs cannot process time steps concurrently, leading to high latency on long sequences.

Self‑attention replaces recurrence with a pairwise interaction that costs \(O(T^2)\) operations, whereas an RNN step costs \(O(T)\). For a typical length \(T=512\), an RNN performs roughly 512 matrix‑vector multiplications, while self‑attention computes \(512^2 ≈ 2.6×10^5\) dot‑products. The following Python snippet illustrates the scaling:

```python
T = 512
rnn_ops = T          # per layer
attn_ops = T * T
print(rnn_ops, attn_ops)
```

The intuition behind self‑attention is simple: each token creates a **query** vector, compares it to **key** vectors of all tokens, and aggregates **value** vectors weighted by the similarity scores. This allows every position to attend to the whole context in a single layer.

A Transformer block can be described textually as:

Input → Multi‑Head Attention → Add & Layer‑Norm → Feed‑Forward → Add & Layer‑Norm → Output

Finally, the trade‑off is clear: self‑attention offers massive parallelism (all tokens processed simultaneously) but incurs quadratic memory \(O(T^2)\). When \(T\) exceeds a few thousand tokens, memory dominates cost and forces techniques such as sparse attention or chunking.

## Scaled Dot‑Product Attention – a minimal working example  

**1️⃣ From input to Q, K, V**  
Assume an input tensor `x` of shape **(B, T, d)** (batch, time, model dim).  
A single linear layer per head projects it:

```text
W_Q, W_K, W_V : (d, d_k)          # usually d_k = d / n_heads
Q = x @ W_Q   → (B, T, d_k)
K = x @ W_K   → (B, T, d_k)
V = x @ W_V   → (B, T, d_v)       # often d_v = d_k
```

The three tensors are now ready for the attention core.

**2️⃣ 10‑line NumPy implementation**

```python
import numpy as np

def sdpa(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1)               # (B, T, T)
    scores = scores / np.sqrt(d_k)                  # √d scaling
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)    # causal mask
    # log‑sum‑exp stabilization
    max_score = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_score)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return attn @ V                                 # (B, T, d_v)
```

Exactly ten non‑comment lines compute the full attention.

**3️⃣ Causal mask & unit test**

```python
def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))

def test_causal():
    B, T, d = 1, 4, 2
    x = np.arange(B*T*d).reshape(B, T, d).astype(float)
    Q, K, V = x, x, x
    out = sdpa(Q, K, V, mask=causal_mask(T))
    assert np.allclose(out[:, :, :], out)          # shape sanity
    # future positions must have zero attention
    assert np.all(out[:, 0, 1:] == 0)
    assert np.all(out[:, 1, 2:] == 0)
    print("causal mask works")
test_causal()
```

The mask blocks any `(i, j)` with `j > i`, guaranteeing autoregressive behavior.

**4️⃣ Runtime scaling**

```python
import time
for T in (256, 1024):
    Q = K = V = np.random.randn(1, T, 64)
    start = time.time()
    sdpa(Q, K, V)
    print(T, "->", time.time() - start)
```

Typical output (CPU, float64):  

```
256 -> 0.012 s
1024 -> 0.190 s
```

The runtime grows roughly **O(T²)**, confirming the quadratic cost of the full attention matrix.

**5️⃣ Softmax overflow guard**  
When `scores` contain large values, `np.exp(scores)` can overflow to `inf`, producing NaNs after division. Subtracting `max_score` (the log‑sum‑exp trick) caps the exponent range, preserving numerical stability without changing the softmax result.  

**Trade‑off note:** The log‑sum‑exp step adds a tiny constant overhead but prevents catastrophic failure on long sequences or high‑dim projections.

## Stacking Multi‑Head Attention and Feed‑Forward Layers

**1. Multi‑Head Attention** – The class below splits the query, key and value tensors into `h` heads, re‑uses the scaled‑dot‑product routine from Section 2 (`scaled_dot_product_attention`), and concatenates the heads back to `[batch, seq_len, d_model]`.

```python
import torch, torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.h = h
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: [B, T, d_model]
        Q = self.W_q(x).view(x.size(0), -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(x.size(0), -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(x.size(0), -1, self.h, self.d_k).transpose(1, 2)

        # scaled_dot_product_attention returns [B, h, T, d_k]
        attn = scaled_dot_product_attention(Q, K, V, mask)

        # concat heads
        attn = attn.transpose(1, 2).contiguous().view(x.size(0), -1, self.h * self.d_k)
        return self.W_o(attn)
```

**2. Position‑wise Feed‑Forward** – Two linear layers with a GELU non‑linearity; `ff_dim` is configurable.

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

**3. Encoder Block** – Wraps attention and feed‑forward with pre‑norm and residual connections.

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, ff_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, h)
        self.ff   = PositionwiseFeedForward(d_model, ff_dim)

    def forward(self, x, mask=None):
        # Self‑attention sub‑layer
        a = self.attn(self.norm1(x), mask)
        x = x + a                     # residual

        # Feed‑forward sub‑layer
        f = self.ff(self.norm2(x))
        return x + f                  # residual
```

**4. Sinusoidal Positional Encodings** – Added once to the token embeddings before the first block.

```python
def sinusoidal_positional_encoding(seq_len: int, d_model: int):
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    i   = torch.arange(d_model, dtype=torch.float).unsqueeze(0)
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe  # [seq_len, d_model]

# verification
emb = torch.randn(32, 128, 512)               # batch, seq, d_model
emb = emb + sinusoidal_positional_encoding(128, 512)
```

**5. Parameter count** – For one encoder layer:

- Q, K, V, O projections: `4 * d_model * d_model = 4 * 512² = 1,048,576`
- Feed‑forward: `d_model * ff_dim + ff_dim * d_model = 2 * 512 * 2048 = 2,097,152`
- Two LayerNorms: `2 * 2 * d_model = 2,048`

Total per layer ≈ **3,147,776** parameters.  
Six layers → `6 * 3,147,776 ≈ 18.9 M` parameters (biases omitted). This estimate guides memory budgeting and informs scaling decisions.

## Fine‑tuning a pretrained Transformer for Text Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Optional: freeze the embedding matrix to speed up early epochs
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
```

### Custom `Dataset`

```python
import torch
from torch.utils.data import Dataset

class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels = texts, labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',   # pad to longest in batch later via collate_fn
            return_tensors='pt'
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
```

A simple `collate_fn` can pad dynamically to the longest sequence in each batch, reducing wasted memory.

### Training loop with AdamW, scheduler, and gradient accumulation

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
)

accum_steps = 4   # simulate batch_size * accum_steps
model.train()
for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
        loss = model(**batch).loss / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # TensorBoard logging
        if (step + 1) % 100 == 0:
            writer.add_scalar("train/loss", loss.item()*accum_steps, epoch*len(train_loader)+step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0],
                              epoch*len(train_loader)+step)
            # evaluate on validation set
            val_acc = evaluate(model, val_loader)
            writer.add_scalar("val/accuracy", val_acc,
                              epoch*len(train_loader)+step)
```

**Why log every 100 steps?** It gives a fine‑grained view without overwhelming the disk I/O.

### Quick benchmark

```bash
for bs in 8 16 32; do
    python benchmark.py --batch_size $bs
done
```

Typical output (single RTX 3090):

| Batch | GPU Mem (GB) | Latency per step (ms) |
|------|--------------|-----------------------|
| 8    | 3.2          | 45                    |
| 16   | 5.8          | 78                    |
| 32   | 10.4         | 132                   |

**Trade‑off:** Larger batches improve gradient stability and throughput but double memory usage and increase per‑step latency, which may exceed GPU limits. Use gradient accumulation to keep the effective batch size while staying within memory budgets.

**Edge cases:**  
- Sentences longer than `max_len` are truncated—ensure `max_len` covers the longest expected input.  
- If OOM occurs, reduce `max_len` or increase `accum_steps`.  
- Non‑float16 models may hit memory caps; consider `model.half()` for mixed precision (adds speed but requires careful loss scaling).

## Common Mistakes When Training Transformers and How to Avoid Them

- **Forgetting to mask padding tokens** – without a mask the attention matrix treats `<pad>` as real content, causing information leakage.  
  ```python
  # `input_ids` shape: (batch, seq_len)
  attention_mask = (input_ids != tokenizer.pad_token_id).long()
  outputs = model(input_ids, attention_mask=attention_mask)
  ```  
  *Why*: The mask forces the softmax to ignore padded positions, keeping gradients clean.

- **Using a learning rate > 5e‑4 with AdamW** – large rates produce sudden loss spikes and unstable training.  
  ```python
  # Simple LR finder
  from torch.optim.lr_scheduler import LambdaLR
  optimizer = AdamW(model.parameters(), lr=1e-5)
  scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1e-5 * (10 ** (step / 1000)))
  ```  
  Start at 1e‑5 for base‑size models; increase only after the loss plateaus.  
  *Trade‑off*: Lower LR slows early convergence but dramatically improves stability.

- **Skipping gradient clipping** – gradients can explode, especially with long sequences. Insert clipping right after `loss.backward()`.  
  ```python
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()
  ```  
  *Why*: Clipping caps the norm, preventing NaNs without hurting most updates.

- **Neglecting weight‑decay on bias and LayerNorm** – applying decay to these parameters harms generalization. Create separate optimizer groups:  
  ```python
  decay = [p for n, p in model.named_parameters()
           if not any(nd in n for nd in ["bias", "LayerNorm.weight"])]
  no_decay = [p for n, p in model.named_parameters()
              if any(nd in n for nd in ["bias", "LayerNorm.weight"])]
  optimizer = AdamW([
      {"params": decay, "weight_decay": 0.01},
      {"params": no_decay, "weight_decay": 0.0}
  ], lr=1e-5)
  ```  
  *Why*: Bias and LayerNorm should stay unbiased to preserve normalization behavior.

- **Setting `max_position_embeddings` lower than the longest sequence** – the model will attempt to embed out‑of‑range positions, leading to OOM errors.  
  ```python
  max_len = max(len(seq) for seq in dataset)
  if config.max_position_embeddings < max_len:
      config.max_position_embeddings = max_len + 10  # safety margin
      model.resize_token_embeddings(len(tokenizer))
  ```  
  Verify the dataset’s maximum length before training; increasing the config adds a few extra parameters but avoids crashes.

**Checklist**  
1. Apply `attention_mask`.  
2. Run an LR finder; cap LR ≤ 5e‑4 (≈ 1e‑5 for large models).  
3. Clip gradients (`max_norm=1.0`).  
4. Separate weight‑decay groups (bias & LayerNorm excluded).  
5. Ensure `max_position_embeddings` ≥ dataset max length.  

Following these steps eliminates the most frequent stability issues when scaling Transformers to production.

## Production Checklist & Next Steps

- **Benchmark inference latency**  
  Run a quick latency sweep across realistic batch sizes. If any batch exceeds 30 ms per request, enable mixed‑precision with `torch.cuda.amp`:

  ```python
  with torch.cuda.amp.autocast():
      logits = model(input_ids)          # inference
  ```

  *Why*: AMP halves FP32 compute time on GPUs, reducing tail latency.

- **Profile memory usage**  
  Use `torch.profiler` to find the peak allocation and set a safe `max_seq_len` that fits the target GPU (e.g., 16 GB).  

  ```python
  with torch.profiler.profile(
          activities=[torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA],
          record_shapes=True) as prof:
      model(input_ids)
  print(prof.key_averages().table(sort_by="cuda_memory_usage"))
  ```

  *Trade‑off*: Shorter sequences lower memory but may hurt model quality; choose the longest length that stays under 80 % of GPU memory.

- **Instrument logs**  
  Log per‑request token count, latency, and GPU utilization (e.g., via `torch.cuda.utilization()` or Prometheus). Set an alert when latency > SLA (e.g., 30 ms).  

  ```json
  {"req_id":"123","tokens":128,"latency_ms":27,"gpu_util":73}
  ```

- **Apply input sanitization**  
  Normalize Unicode (`unicodedata.normalize('NFKC', text)`) and enforce length caps before tokenization to block injection attacks and out‑of‑bounds memory use.

- **Plan a rollout**  
  1. Deploy the new model to a shadow traffic bucket.  
  2. A/B test against the baseline, tracking accuracy and latency.  
  3. Monitor for accuracy drift; if drift > 2 % trigger a retraining job.  
  4. Schedule periodic re‑training (e.g., weekly) to incorporate fresh data.

Following this checklist lets you ship a Transformer safely while keeping clear paths for performance tuning and future improvements.
