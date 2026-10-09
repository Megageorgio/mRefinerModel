"""
Train a mRefinerModel boundary refiner.

  python train.py --train data/train --val data/val --run myrefiner
  python train.py --train data/train --val data/val --pred-train pred/train --pred-val pred/val --run myrefiner

--train / --val: folders with .wav + .lab (HTK: "start end phoneme", 100 ns units) of the reference labels.
--pred-train / --pred-val (optional, recommended): the same files labelled by your first stage (WFL-ASR, SOFA...),
    same names, one or more folders. The refiner then learns the real errors of that labeller; the training
    part should come from a model that did not see those files (e.g. cross-validation folds). Without them
    it learns from random shifts of the reference only (set training.real_frac: 0, see config/synthetic.yaml).
File names "voice__anything.wav" give per-voice averages in the metrics (optional).

Run again with the same --run to continue. The best epoch is chosen on --pred-val when given (the real case),
otherwise on randomly shifted validation boundaries. Then: python export.py runs/<run> models/<name>
"""
import argparse
import csv
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mrefinermodel.data import BoundaryDataset, collate, load_phoneset, read_lab  # noqa: E402
from mrefinermodel.model import BoundaryRefiner  # noqa: E402


def wavs(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(".wav") and os.path.exists(os.path.join(folder, os.path.splitext(f)[0] + ".lab"))]


def evaluate(net, dl, half, hop_s, device):
    """Errors before/after in ms on the true boundary time (mean over voices; one boundary counts at most 100 ms)."""
    net.eval()
    per_voice = {}
    with torch.no_grad():
        for b in dl:
            if len(b["tgt"]) == 0:
                continue
            p = torch.softmax(net(b["win"].to(device), b["ctx"].to(device), b["dur"].to(device)), -1).cpu().numpy()
            pos = np.arange(p.shape[1])
            for k in range(len(p)):
                best = int(p[k].argmax())
                lo, hi = max(best - 2, 0), min(best + 3, p.shape[1])
                frac = (p[k, lo:hi] * pos[lo:hi]).sum() / p[k, lo:hi].sum()
                t_after = (int(b["center"][k]) + frac - half) * hop_s
                tt = float(b["t_true"][k])
                d = per_voice.setdefault(b["voice"][k], [[], []])
                d[0].append(abs(float(b["t_pred"][k]) - tt) * 1000)
                d[1].append(abs(t_after - tt) * 1000)
    net.train()
    if not per_voice:
        return None

    def agg(f):
        return float(np.mean([f(np.array(v)) for v in per_voice.values()]))
    r = {}
    for name, i in (("before", 0), ("after", 1)):
        r[f"{name}_mae"] = agg(lambda v: np.minimum(v[i], 100).mean())
        r[f"{name}_in10"] = agg(lambda v: (v[i] <= 10).mean() * 100)
        r[f"{name}_in20"] = agg(lambda v: (v[i] <= 20).mean() * 100)
    return r


def fmt(r):
    return (f"before {r['before_mae']:5.1f} ms, <=10 {r['before_in10']:4.1f}%, <=20 {r['before_in20']:4.1f}% -> "
            f"after {r['after_mae']:5.1f} ms, <=10 {r['after_in10']:4.1f}%, <=20 {r['after_in20']:4.1f}%")


def collect_phones(files):
    seen = {}
    for w in files:
        for _, _, p in read_lab(os.path.splitext(w)[0] + ".lab"):
            seen[p] = seen.get(p, 0) + 1
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="folder with reference .wav + .lab")
    ap.add_argument("--val", required=True)
    ap.add_argument("--pred-train", nargs="*", default=[], help="first-stage labels of the training files")
    ap.add_argument("--pred-val", nargs="*", default=[], help="first-stage labels of the validation files")
    ap.add_argument("--phonemes", help="phoneme list, one per line (default: every phoneme of the training labels)")
    ap.add_argument("--run", default="refiner")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "default.yaml"))
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--fresh", action="store_true", help="start over (the old run folder is renamed)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    run_dir = os.path.abspath(os.path.join(HERE, "runs", args.run))
    if args.fresh and os.path.exists(run_dir):
        k = 1
        while os.path.exists(f"{run_dir}_old_{k}"):
            k += 1
        os.rename(run_dir, f"{run_dir}_old_{k}")
    os.makedirs(run_dir, exist_ok=True)
    state_path = os.path.join(run_dir, "last_state.pt")
    resume = os.path.exists(state_path)
    train_files, val_files = wavs(args.train), wavs(args.val)
    if not train_files or not val_files:
        sys.exit("No .wav + .lab pairs in --train or --val")
    if resume:
        cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml"), encoding="utf-8"))
    else:
        cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
        if args.phonemes:
            shutil.copyfile(args.phonemes, os.path.join(run_dir, "phonemes.txt"))
        else:
            with open(os.path.join(run_dir, "phonemes.txt"), "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(collect_phones(train_files)) + "\n")
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    if args.max_epochs:
        cfg["training"]["max_epochs"] = args.max_epochs
    phones = load_phoneset(os.path.join(run_dir, "phonemes.txt"))
    p2i = {p: i for i, p in enumerate(phones)}
    tr, fr = cfg["training"], cfg["frontend"]
    half, hop_ms = fr["half_window_frames"], fr["hop_ms"]
    if tr.get("real_frac", 0) > 0 and not args.pred_train:
        print("No --pred-train: learning from random shifts only (real_frac -> 0)")
        tr["real_frac"] = 0.0
    train_ds = BoundaryDataset(train_files, p2i, cfg, train=True, pred_dirs=args.pred_train)
    val_ds = BoundaryDataset(val_files, p2i, cfg, train=False, real_frac=0.0)
    val_real = BoundaryDataset(val_files, p2i, cfg, train=False, real_frac=1.0, pred_dirs=args.pred_val[:1]) \
        if args.pred_val else None
    chunks_per_batch = max(1, tr["batch_boundaries"] // tr["per_chunk"])
    nw = args.num_workers
    train_dl = DataLoader(train_ds, batch_size=chunks_per_batch, shuffle=True, num_workers=nw, collate_fn=collate,
                          persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=nw, collate_fn=collate, persistent_workers=nw > 0)
    real_dl = DataLoader(val_real, batch_size=4, shuffle=False, num_workers=nw, collate_fn=collate,
                         persistent_workers=nw > 0) if val_real else None

    torch.manual_seed(42)
    net = BoundaryRefiner(len(phones), cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, tr["lr_decay_gamma"])
    epoch, best, bad, hist = 0, (math.inf, -1), 0, []
    if resume:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        epoch, best, bad, hist = st["epoch"], st["best"], st["bad"], st["hist"]
        print(f"Continuing {run_dir} from epoch {epoch}")
    print(f"Network: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters; train {len(train_ds)} files, "
          f"val {len(val_ds)}; window ±{half * hop_ms:.0f} ms, hop {hop_ms} ms; "
          f"real first-stage errors in training: {tr.get('real_frac', 0) * 100:.0f}%")
    try:
        while epoch < tr["max_epochs"]:
            t0, tot, n = time.time(), 0.0, 0
            for b in train_dl:
                if (b["tgt"] >= 0).sum() == 0:
                    continue
                logits = net(b["win"].to(device), b["ctx"].to(device), b["dur"].to(device))
                loss = F.cross_entropy(logits, b["tgt"].to(device), label_smoothing=0.05, ignore_index=-100)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += loss.item(); n += 1
            sched.step()
            r_syn = evaluate(net, val_dl, half, hop_ms / 1000, device)
            r_real = evaluate(net, real_dl, half, hop_ms / 1000, device) if real_dl else None
            key = (r_real or r_syn)["after_mae"]
            r = {"epoch": epoch, "loss": tot / max(n, 1), "minutes": (time.time() - t0) / 60, "score": key}
            r.update({f"shifted_{k}": v for k, v in r_syn.items()})
            if r_real:
                r.update({f"first_stage_{k}": v for k, v in r_real.items()})
            hist.append(r)
            sd = {k: (v.half() if v.is_floating_point() else v) for k, v in net.state_dict().items()}
            torch.save({"state_dict": sd, "epoch": epoch}, os.path.join(run_dir, "last.pt"))
            mark = ""
            if key < best[0]:
                best, bad, mark = (key, epoch), 0, "  *"
                shutil.copyfile(os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt"))
            else:
                bad += 1
            print(f"epoch {epoch:3d}  loss {r['loss']:.3f}  {r['minutes']:.1f} min{mark}")
            print(f"    random shifts: {fmt(r_syn)}")
            if r_real:
                print(f"    first stage:   {fmt(r_real)}")
            with open(os.path.join(run_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(hist[-1]))
                w.writeheader(); w.writerows([{k: h.get(k) for k in hist[-1]} for h in hist])
            epoch += 1
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                        "best": best, "bad": bad, "hist": hist}, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
            if bad >= tr["patience"]:
                print(f"Early stop: {tr['patience']} epochs without improvement.")
                break
    except KeyboardInterrupt:
        print("\nStopped; run again to continue.")
        return
    print(f"Done: {os.path.join(run_dir, 'best.pt')} (epoch {best[1]}, {best[0]:.2f} ms)")


if __name__ == "__main__":
    main()
