"""
predict_dl.py - run the trained network over the whole scan and write a DXF.

  python predict_dl.py --model model.pt --data data --out plan_dl.dxf [--thr 0.5]

Needs PyTorch (CPU is fine for prediction).  The DXF has layers Murs / mur bas / Zellige and uses the same
coordinates as your scan, so it can be opened in the review tool on the first PC and corrected there.
"""
import argparse
import json
import os

import numpy as np
import torch

import dl_vectorize as dv
from train_dl import UNet, load, predict_full


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="plan_dl.dxf")
    ap.add_argument("--thr", type=float, default=0.5, help="probability threshold (lower = more walls)")
    ap.add_argument("--save-prob", action="store_true")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = UNet(cfg["cin"], cfg["ncls"], cfg["base"]).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    stack, _labels, meta = load(a.data)
    print(f"model trained with held-out wall F1 {ck.get('val_f1', float('nan')):.3f}; predicting on {dev} ...", flush=True)
    prob = predict_full(model, stack, ck["mean"], ck["std"], dev).numpy()
    if a.save_prob:
        np.savez_compressed(os.path.splitext(a.out)[0] + "_prob.npz", prob=prob.astype(np.float16))
    dv.prob_to_dxf(prob, meta["x0"], meta["y0"], meta["cell"], a.out, thr=a.thr)


if __name__ == "__main__":
    main()
