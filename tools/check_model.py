from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "mahalanobis_model_full_500.npz"

if not path.exists():
    print("NOT FOUND:", path)
    print("FusionGuard V3 will use its software-demo identity-covariance fallback.")
    raise SystemExit(1)

with np.load(path, allow_pickle=False) as model:
    keys = sorted(model.files)
    print("FOUND:", path)
    print("Keys:", keys)
    for required in ("mean", "inv_cov", "std"):
        if required not in model:
            raise SystemExit(f"Missing required array: {required}")
    print("mean shape:", model["mean"].shape)
    print("inv_cov shape:", model["inv_cov"].shape)
    print("std shape:", model["std"].shape)
    print("Model structure looks compatible with FusionGuard V3.")
