import os
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .model import MelFrontend


def read_lab(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        p = line.split()
        if len(p) >= 3:
            try:
                out.append((float(p[0]) / 1e7, float(p[1]) / 1e7, p[2]))
            except ValueError:
                pass
    return out


def load_phoneset(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def touching(segs, gap=0.002):
    """Indices i of internal boundaries (between segs[i-1] and segs[i]) where the two segments touch."""
    return [i for i in range(1, len(segs)) if abs(segs[i][0] - segs[i - 1][1]) <= gap]


def context(segs, i, t, p2i, unk, ctx):
    """Phoneme ids [ctx on the left ..., ctx on the right] and log-durations of the neighbours for boundary i at time t."""
    ids = [p2i.get(segs[i - k][2], unk) if i - k >= 0 else unk for k in range(ctx, 0, -1)]
    ids += [p2i.get(segs[i + k][2], unk) if i + k < len(segs) else unk for k in range(ctx)]
    dl = max(t - segs[i - 1][0], 0.001)
    dr = max(segs[i][1] - t, 0.001)
    return ids, [float(np.log(dl)), float(np.log(dr))]


def boundaries(segs, gap=0.002):
    """(time, left phone, right phone) for touching neighbours (kept for compatibility)."""
    return [(segs[i][0], segs[i - 1][2], segs[i][2]) for i in touching(segs, gap)]


def align_pairs(a, b):
    """Levenshtein alignment -> list of (i, j) where a[i] == b[j] matched."""
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    out, i, j = [], m, n
    while i > 0 and j > 0:
        c = int(a[i - 1] != b[j - 1])
        if dp[i, j] == dp[i - 1, j - 1] + c:
            if not c:
                out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dp[i, j] == dp[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def matched_boundaries(gt, pred):
    """Boundaries of the prediction whose both neighbours match the reference: (i_pred, t_pred, t_true)."""
    if [p for _, _, p in gt] == [p for _, _, p in pred]:
        pairs = [(k, k) for k in range(len(gt))]
    else:
        pairs = align_pairs([p for _, _, p in gt], [p for _, _, p in pred])
    m = {j: i for i, j in pairs}
    tp = set(touching(pred))
    tg = set(touching(gt))
    out = []
    for j in range(1, len(pred)):
        if j in tp and j in m and (j - 1) in m and m[j - 1] == m[j] - 1 and m[j] in tg:
            out.append((j, pred[j][0], gt[m[j]][0]))
    return out


def windows(mel, centers_frames, half):
    """mel [n_mels, F]; centers int[N] -> [N, n_mels, 2*half+1] (zero padded at the edges)."""
    F_ = mel.shape[1]
    padded = torch.nn.functional.pad(mel, (half, half))
    idx = torch.as_tensor(centers_frames, dtype=torch.long).clamp(0, F_ - 1)
    rng = torch.arange(-half, half + 1)
    gather = (idx[:, None] + rng[None, :] + half)  # into padded
    return padded[:, gather].permute(1, 0, 2).contiguous()


def laplace(rng, scale, mx):
    u = rng.random() - 0.5
    return float(np.clip(-scale * np.sign(u) * np.log(1 - 2 * abs(u) + 1e-12), -mx, mx))


class BoundaryDataset(Dataset):
    """Each item: boundaries of one chunk. The window is centred where the first stage put the boundary:
    either a real prediction (pred_dirs: same-named .lab of the first stage) or the reference moved by a random error.
    Target = position of the reference boundary in the window (-100 if outside: not learnable, still measured)."""

    def __init__(self, wavs, phone2id, cfg, train=True, seed=0, pred_dirs=None, real_frac=None):
        self.wavs, self.p2i, self.cfg, self.train = wavs, phone2id, cfg, train
        fr = cfg["frontend"]
        self.fe = MelFrontend(fr["sample_rate"], fr["hop_ms"], fr["n_mels"], fr["n_fft"], fr["win_ms"])
        self.hop_s = self.fe.hop / fr["sample_rate"]
        self.half = fr["half_window_frames"]
        self.ctx = cfg["model"].get("context", 1)
        self.seed = seed
        self.pred_dirs = [d for d in (pred_dirs or []) if d and os.path.isdir(d)]
        tr = cfg["training"]
        self.real_frac = tr.get("real_frac", 0.0) if real_frac is None else real_frac

    def __len__(self):
        return len(self.wavs)

    def _sources(self, path):
        base = os.path.splitext(os.path.basename(path))[0] + ".lab"
        return [os.path.join(d, base) for d in self.pred_dirs if os.path.exists(os.path.join(d, base))]

    def __getitem__(self, i):
        tr = self.cfg["training"]
        rng = random.Random() if self.train else random.Random(self.seed * 100003 + i)
        path = self.wavs[i]
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(1)
        wav = wav / (np.abs(wav).max() + 1e-8)
        if self.train and rng.random() < 0.5:
            wav = np.clip(wav * rng.uniform(0.5, 1.5) + np.random.normal(0, 0.005, wav.shape).astype(np.float32), -1, 1)
        with torch.no_grad():
            mel = self.fe(torch.from_numpy(wav))
        gt = read_lab(os.path.splitext(path)[0] + ".lab")
        unk = len(self.p2i)
        scale, mx = tr["jitter_ms"] / 1000, tr["max_jitter_ms"] / 1000
        srcs = self._sources(path)
        items = []  # (t_pred, t_true, ids, dur)
        if srcs and rng.random() < self.real_frac:
            pred = read_lab(srcs[rng.randrange(len(srcs))] if self.train else srcs[0])
            for j, tp, tt in matched_boundaries(gt, pred):
                if self.train:
                    tp += laplace(rng, tr.get("real_extra_jitter_ms", 3.0) / 1000, 0.02)
                ids, dur = context(pred, j, tp, self.p2i, unk, self.ctx)
                items.append((tp, tt, ids, dur))
        else:
            for j in touching(gt):
                tt = gt[j][0]
                tp = tt + laplace(rng, scale, mx)
                tp = min(max(tp, gt[j - 1][0] + 0.005), gt[j][1] - 0.005) if gt[j][1] - gt[j - 1][0] > 0.011 else tt
                ids, dur = context(gt, j, tp, self.p2i, unk, self.ctx)
                items.append((tp, tt, ids, dur))
        if self.train and len(items) > tr["per_chunk"]:
            items = rng.sample(items, tr["per_chunk"])
        n = 2 * self.half + 1
        if not items:
            return {"win": torch.zeros(0, mel.shape[0], n), "tgt": torch.zeros(0, dtype=torch.long),
                    "ctx": torch.zeros(0, 2 * self.ctx, dtype=torch.long), "dur": torch.zeros(0, 2),
                    "t_pred": torch.zeros(0, dtype=torch.float64), "t_true": torch.zeros(0, dtype=torch.float64),
                    "center": torch.zeros(0, dtype=torch.long), "voice": []}
        centers, targets = [], []
        for tp, tt, _, _ in items:
            c = int(round(tp / self.hop_s))
            tgt = int(round(tt / self.hop_s)) - c + self.half
            centers.append(c)
            targets.append(tgt if 0 <= tgt <= 2 * self.half else -100)
        return {"win": windows(mel, centers, self.half), "tgt": torch.tensor(targets),
                "ctx": torch.tensor([x[2] for x in items]), "dur": torch.tensor([x[3] for x in items], dtype=torch.float32),
                "t_pred": torch.tensor([x[0] for x in items], dtype=torch.float64),
                "t_true": torch.tensor([x[1] for x in items], dtype=torch.float64),
                "center": torch.tensor(centers), "voice": [os.path.basename(path).split("__")[0]] * len(items)}


def collate(items):
    out = {k: torch.cat([x[k] for x in items]) for k in ("win", "tgt", "ctx", "dur", "t_pred", "t_true", "center")}
    out["voice"] = [v for x in items for v in x["voice"]]
    return out
