"""Boundary refiner: looks at a log-mel window (2.5 ms hop) around one boundary, knows the phonemes on
both sides, and says where in the window the boundary really is."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MelFrontend(nn.Module):
    def __init__(self, sr=16000, hop_ms=2.5, n_mels=80, n_fft=512, win_ms=15.0):
        super().__init__()
        self.sr = sr
        self.hop = int(round(sr * hop_ms / 1000))
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=int(sr * win_ms / 1000),
                                                        hop_length=self.hop, n_mels=n_mels, center=True, power=2.0)

    def forward(self, wav):  # [S] -> [n_mels, frames], normalised per file
        m = torch.log(self.mel(wav).clamp_min(1e-5))
        return (m - m.mean()) / (m.std() + 1e-5)


class Block(nn.Module):
    def __init__(self, ch, k, d, p):
        super().__init__()
        self.norm = nn.BatchNorm1d(ch)
        self.conv = nn.Conv1d(ch, 2 * ch, k, padding=d * (k - 1) // 2, dilation=d)
        self.out = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(self.out(F.gelu(F.glu(self.conv(self.norm(x)), dim=1))))


class BoundaryRefiner(nn.Module):
    def __init__(self, n_phones, cfg):
        super().__init__()
        m = cfg["model"]
        ch, emb = m.get("channels", 128), m.get("phone_emb", 32)
        self.n_mels = cfg["frontend"].get("n_mels", 80)
        self.half = cfg["frontend"]["half_window_frames"]
        self.ctx = m.get("context", 1)              # phonemes on each side
        self.use_dur = m.get("use_durations", False)  # log-durations of the two neighbours (as the first stage put them)
        self.emb = nn.Embedding(n_phones + 1, emb)  # last id = unknown phoneme
        self.inp = nn.Conv1d(self.n_mels + 2 * self.ctx * emb + 1 + (2 if self.use_dur else 0), ch, 5, padding=2)
        k = m.get("kernel", 5)
        self.blocks = nn.ModuleList([Block(ch, k, d, m.get("dropout", 0.1)) for d in m.get("dilations", [1, 2, 4, 8, 1, 2])])
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, win, ctx, dur=None):
        """win [B, n_mels, 2*half+1]; ctx [B, 2*context] phoneme ids (left..., right...); dur [B, 2]
        -> logits over positions [B, 2*half+1]."""
        B, _, T = win.shape
        pos = torch.linspace(-1, 1, T, device=win.device).expand(B, 1, T)
        feats = [self.emb(ctx).reshape(B, -1)]
        if self.use_dur:
            feats.append(dur)
        cond = torch.cat(feats, -1).unsqueeze(-1).expand(-1, -1, T)
        x = self.inp(torch.cat([win, cond, pos], 1))
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(1)
