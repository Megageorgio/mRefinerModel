"""
Make a model folder (for refine.py, the Python API and releases) from a training run.

  python export.py runs/myrefiner models/mrefinermodel-xx [--zip mrefinermodel-xx.zip]

Writes model.pt (weights only, fp16), config.yaml (without the training state) and phonemes.txt.
"""
import argparse
import os
import shutil
import zipfile

import torch
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--zip", help="also pack the folder into this zip (files at its root)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sd = torch.load(os.path.join(args.run, args.checkpoint), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save({"state_dict": sd}, os.path.join(args.out, "model.pt"))
    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml"), encoding="utf-8"))
    cfg.pop("training", None)
    here = os.path.dirname(os.path.abspath(__file__))
    defaults = yaml.safe_load(open(os.path.join(here, "config", "default.yaml"), encoding="utf-8")).get("inference", {})
    cfg["inference"] = {**defaults, **(cfg.get("inference") or {})}  # older runs: add the newer settings
    with open(os.path.join(args.out, "config.yaml"), "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    shutil.copyfile(os.path.join(args.run, "phonemes.txt"), os.path.join(args.out, "phonemes.txt"))
    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("config.yaml", "phonemes.txt", "model.pt"):
                z.write(os.path.join(args.out, name), name)
    print(f"Exported -> {args.out}" + (f" and {args.zip}" if args.zip else ""))


if __name__ == "__main__":
    main()
