"""Evaluate the batch-512 checkpoint and merge into ablation_batch.csv."""
import numpy as np
import pandas as pd
import torch

from diffusion_corners import pick_device, SP, load_dataset
from ablate_batch import evaluate

device = pick_device()
ids, X, F, M, y = load_dataset()
split = pd.read_csv(f"{SP}/diffusion_split.csv")
hold_idx = np.where(split.split.to_numpy() == "holdout")[0]
train_idx = np.where(split.split.to_numpy() == "train")[0]

res = evaluate("_b512", ids, X, F, M, y, hold_idx, train_idx, device)
res["batch"], res["steps"] = 512, 30000
df = pd.read_csv(f"{SP}/ablation_batch.csv")
df = pd.concat([df[df.batch != 512], pd.DataFrame([res])], ignore_index=True)
df = df.sort_values("batch")
df.to_csv(f"{SP}/ablation_batch.csv", index=False)
print(df.round(3).to_string(index=False))
