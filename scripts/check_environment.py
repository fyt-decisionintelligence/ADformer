"""Verify the installed packages and FlashAttention forward/backward on one GPU."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=range(4), default=3)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from flash_attn import flash_attn_func
    import geopandas as gpd
    import h3
    from fastdtw import _fastdtw

    assert torch.cuda.is_available(), "CUDA is unavailable in this process"
    query, key, value = [
        torch.randn(2, 135, 3, 4, device="cuda:0", dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    ]
    output = flash_attn_func(query, key, value)
    assert output.shape == query.shape and torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all()
               for tensor in (query, key, value))
    torch.cuda.synchronize()

    root = Path(__file__).resolve().parents[1]
    cell = (root / "data" / "xian_hexAddr.txt").read_text().splitlines()[0]
    assert h3.h3_is_valid(cell) and cell in h3.k_ring(cell, 1)
    zones = gpd.read_file(root / "data" / "geo" / "taxi_zones" / "taxi_zones.shp")
    projected = zones.to_crs(epsg=32618)
    assert projected.geometry.notna().all()
    distance, _ = _fastdtw.fastdtw([[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]])
    assert distance == 0

    packages = ["torch", "flash-attn", "numpy", "scipy", "pandas", "geopandas",
                "pyogrio", "pyproj", "shapely", "geopy", "h3", "tqdm", "fastdtw", "einops"]
    report = {
        "python": platform.python_version(),
        "packages": {package: version(package) for package in packages},
        "cuda_runtime": torch.version.cuda, "physical_gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "flash_attention_forward_backward": "passed",
        "h3_legacy_api": "passed", "fastdtw_compiled_extension": "passed",
        "nyc_geographic_rows": len(zones),
        "nyc_unique_location_ids": int(zones.LocationID.nunique()),
    }
    target = root / "log" / "environment.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
