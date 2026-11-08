# ChengAI

## The ChengAI Model Weight and Train data

ChengAI mainly focuses on narrow AI

ChengAI's reasoning is super cheap and can run on **devices with very low specs or even no GPU.**

# The Gorb-1 Running Code

```python
#!/usr/bin/env python3

# Install the dependent software packages :  pip install torch transformers  Optional Install numpy

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "best_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

MAX_LEN = 16

def predict(name):
    name = name.strip().lower()
    enc = tokenizer(name, truncation=True, max_length=MAX_LEN,
                    padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    label = "Girl" if prob[1] > prob[0] else "Boy"
    conf = float(max(prob))
    return label, conf, {"Boy": float(prob[0]), "Girl": float(prob[1])}

if __name__ == "__main__":
    tests = ["小明", "Mike", "Mr Wang", "Mr John", "Parker", "小红", "小乐", "Patricia", "Cain", "Addy"]
    for n in tests:
        label, conf, probs = predict(n)
        print(f"{n:<10} -> {label}  conf={conf:.4f}  {probs}")
```

# About reasoning Very ChengAI Model
  Inference ChengAI Recommended Configuration Table

  | Item | Minimum | Recommended |
|---|---|---|
| GPU | Not required | Optional |
| CPU | 1 core (64-bit) | 2–4 cores |
| RAM | 512 MB | 1–2 GB |
| Disk | 700 MB (fp32) | 340 MB (fp16) |
| Latency per name | 50 ms | 15–30 ms |
| Throughput | 20 names/s | 100–1000 names/s |


  We used this setup when training the latest ChengAI.

  | Item | Number |
  | ---  |  ---   |
  | GPU  | NVIDIA V100 (VRAM 32G) |
  | CPU | 10vGPU |
  | RAM | 40 GB |
  | Disk Size | 120 GB (HDD) |
  | Computing Power | ~15 TFlops |



