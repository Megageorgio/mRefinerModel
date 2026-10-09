"""mRefinerModel: refine phoneme boundaries of a ready segmentation (WFL-ASR, SOFA, HubertFA, MFA or by hand).

    from mrefinermodel import Refiner
    r = Refiner("models/mrefinermodel-ru")
    segments = r.refine(audio, sr, segments)            # [(start, end, phoneme), ...] in seconds

Only boundaries between touching segments move; phonemes, their order and a minimum length stay as they are.
A boundary is moved only when the refiner is sure enough (see `mode`), so with an unfamiliar language or
labelling style it leaves the input mostly as it was instead of making it worse.
"""
import os

import numpy as np
import torch
import yaml

from .data import context, load_phoneset, touching, windows
from .model import BoundaryRefiner, MelFrontend

MODES = ("auto", "normal", "safe")


class Refiner:
    def __init__(self, model_dir, checkpoint=None, device=None):
        self.cfg = yaml.safe_load(open(os.path.join(model_dir, "config.yaml"), encoding="utf-8"))
        self.phones = load_phoneset(os.path.join(model_dir, "phonemes.txt"))
        self.p2i = {p: i for i, p in enumerate(self.phones)}
        checkpoint = checkpoint or next((c for c in ("model.pt", "best.pt") if os.path.exists(os.path.join(model_dir, c))),
                                        "model.pt")
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(model_dir, checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        fr = self.cfg["frontend"]
        self.fe = MelFrontend(fr["sample_rate"], fr["hop_ms"], fr["n_mels"], fr["n_fft"], fr["win_ms"]).to(self.device)
        self.sr = fr["sample_rate"]
        self.hop_s = self.fe.hop / self.sr
        self.half = fr["half_window_frames"]
        self.net = BoundaryRefiner(len(self.phones), self.cfg)
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        self.net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
        self.net.to(self.device).eval()
        inf = self.cfg.get("inference", {})
        self.min_dur = inf.get("min_phone_ms", 10.0) / 1000
        self.conf_normal = inf.get("min_confidence", 0.0)
        self.conf_safe = inf.get("safe_min_confidence", 0.2)
        self.max_shift = inf.get("max_shift_ms", 60.0) / 1000
        self.max_shift_safe = inf.get("safe_max_shift_ms", 40.0) / 1000
        self.unknown_limit = inf.get("auto_safe_unknown_share", 0.5)
        self.passes = inf.get("passes", 1)

    def unknown_share(self, segments, phone_map=None):
        pm = phone_map or {}
        names = [pm.get(p, p) for _, _, p in segments]
        return sum(1 for p in names if p not in self.p2i) / max(len(names), 1)

    @torch.no_grad()
    def refine(self, audio, sr, segments, mode="auto", phone_map=None, min_confidence=None, max_shift_ms=None,
               passes=None, return_info=False):
        """segments: [(start, end, phone)] in seconds -> the same list with refined boundaries.

        mode: "normal" - the settings the model was tuned with (labels of its own language and style);
              "safe"   - move only confident boundaries and not far: almost never makes things worse, gains
                         less (other languages, labels of another style);
              "auto"   - "safe" when most phonemes are unknown to the model (after phone_map), else "normal".
        phone_map: {other_name: our_name} for labels with another phoneme set.
        min_confidence / max_shift_ms: override the threshold / the largest move of the mode.
        return_info=True -> (segments, {"mode", "unknown_share", "boundaries", "moved" (by more than 1 ms)})."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        segs = [tuple(s) for s in segments]
        share = self.unknown_share(segs, phone_map)
        used = mode if mode != "auto" else ("safe" if share > self.unknown_limit else "normal")
        conf = min_confidence if min_confidence is not None else (self.conf_safe if used == "safe" else self.conf_normal)
        limit = (self.max_shift_safe if used == "safe" else self.max_shift) if max_shift_ms is None else max_shift_ms / 1000
        info = {"mode": used, "unknown_share": share, "boundaries": len(touching(segs)), "moved": 0}
        if len(segs) >= 2:
            a = np.asarray(audio, dtype=np.float32)
            if a.ndim > 1:
                a = a.mean(1)
            if sr != self.sr:
                import torchaudio
                a = torchaudio.functional.resample(torch.from_numpy(a), sr, self.sr).numpy()
            a = a / (np.abs(a).max() + 1e-8)
            mel = self.fe(torch.from_numpy(a).to(self.device))
            orig = [s[0] for s in segs]
            out = [list(s) for s in segs]
            for _ in range(passes or self.passes):
                out = self._pass(mel, out, segs, conf, limit, phone_map or {})
            info["moved"] = sum(1 for i in touching(segs) if abs(out[i][0] - orig[i]) > 0.001)  # by more than 1 ms
            segs = [tuple(s) for s in out]
        return (segs, info) if return_info else segs

    def _pass(self, mel, segs, original, min_conf, limit, pm):
        idx = touching(segs)
        if not idx:
            return segs
        unk = len(self.phones)
        mapped = [(s, e, pm.get(p, p)) for s, e, p in segs]
        centers, ctx, dur = [], [], []
        for i in idx:
            ids, d = context(mapped, i, segs[i][0], self.p2i, unk, self.net.ctx)
            centers.append(int(round(segs[i][0] / self.hop_s)))
            ctx.append(ids)
            dur.append(d)
        ctx = torch.tensor(ctx, device=self.device)
        dur = torch.tensor(dur, dtype=torch.float32, device=self.device)
        out = []
        for k in range(0, len(idx), 1024):
            w = windows(mel, centers[k:k + 1024], self.half)
            out.append(torch.softmax(self.net(w, ctx[k:k + 1024], dur[k:k + 1024]), -1).cpu())
        prob = torch.cat(out).numpy()
        pos = np.arange(prob.shape[1])
        new = [list(s) for s in segs]
        for j, i in enumerate(idx):
            p = prob[j]
            best = int(p.argmax())
            if p[best] < min_conf:
                continue
            lo, hi = max(best - 2, 0), min(best + 3, len(p))  # sub-frame: weighted mean around the peak
            frac = float((p[lo:hi] * pos[lo:hi]).sum() / p[lo:hi].sum())
            t = (centers[j] + frac - self.half) * self.hop_s
            t0 = original[i][0]
            t = min(max(t, t0 - limit), t0 + limit)
            lo_t, hi_t = new[i - 1][0] + self.min_dur, segs[i][1] - self.min_dur
            if lo_t <= hi_t:
                t = min(max(t, lo_t), hi_t)
                new[i - 1][1] = t
                new[i][0] = t
        return new
