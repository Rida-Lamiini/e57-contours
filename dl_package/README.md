# Wall detection - deep learning package

Train a small U-Net on the scan image layers and your drawings, then turn its output into a DXF.
Copy this whole folder to the powerful PC.

## What is inside
| file | purpose |
|---|---|
| `data/stack.npz` | the scan as 9 image layers at 10 cm (height above ground, 3 slice bands, density, colour warmth / brightness / greenness, valid mask) - 4594 x 3441 cells |
| `data/labels.npz` | your drawings as labels: 0 background, 1 wall, 2 low wall, 3 zellige, **255 = not drawn (ignored)** |
| `data/meta.json` | coordinates of the image (origin, cell size, channel names) |
| `train_dl.py` | trains the network (PyTorch) |
| `predict_dl.py` | runs the trained network over the whole area and writes a DXF |
| `dl_vectorize.py`, `outline.py` | turn the wall map into clean straight CAD lines (no PyTorch needed) |

## 1. Install (once)
Python 3.10 - 3.12, then:
```
pip install -r requirements.txt
```
For an NVIDIA GPU, install PyTorch with CUDA from https://pytorch.org first (pick your CUDA version), then the rest.

## 2. Train
```
python train_dl.py --data data --epochs 80
```
- Prints `device: cuda (...)` or `cpu`. On a GPU this takes a few minutes; on a CPU roughly 30-90 minutes.
- Use `--base 32` on a good GPU for a bigger network.
- The right-hand quarter of the labelled area is **held out** and never trained on. The line
  `held-out wall F1` is measured there, so it tells you how well it works on ground it has not seen.
- The best model is saved as `model.pt`.

## 3. Predict (a DXF)
```
python predict_dl.py --model model.pt --data data --out plan_dl.dxf
```
- `--thr 0.4` finds more walls (more false lines), `--thr 0.6` fewer.
- The DXF has layers `Murs`, `mur bas`, `Zellige`, in the same coordinates as the scan.

## 4. Bring it back
Copy `model.pt` and `plan_dl.dxf` back to the first PC. Open `plan_dl.dxf` in the review tool
(`review_tool.py plan_dl.dxf`) to fix what is wrong, then save: the corrected drawing is kept for the next round.

## Important limits (read before trusting the result)
- **Very little labelled data.** Only about 5,800 m2 around the outline of one building complex is labelled, and
  only its OUTER outline. The network can learn the outline well but has seen no inner walls, no rooms, no zellige
  areas (except the few drawn). It will NOT reliably find inner walls until you draw some.
- **One site.** A model trained on this complex may not work on other buildings. Test on a different area before trusting it.
- **The held-out F1 is the number to look at.** If it is low (< 0.6), the data is not enough; draw more walls.

## How to improve it (the loop)
1. Open `plan_dl.dxf` (or the earlier plan) in the review tool on the first PC and **draw the inner walls** of
   the main complex, then save (a copy goes to the `training` folder).
2. On the first PC run `python dl_export.py` (rebuilds `data/` with the new, fuller labels).
3. Copy the new `data/` here and train again. More complete drawings = better results, especially for inner walls.
