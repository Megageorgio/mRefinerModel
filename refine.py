"""
Refine the boundaries of ready labels (WFL-ASR, SOFA, HubertFA, MFA, by hand).

  python refine.py -i folder_with_wav_and_lab -m models/mrefinermodel-ru -o output_folder
  python refine.py -i ... -m ... -o ... --map other_to_ours.txt   # other phoneme names: lines "theirs -> ours"
  python refine.py -i ... -m ... -o ... --mode safe               # other language / labels of another style

Labels (.lab, HTK: "start end phoneme" in 100 ns units) are taken next to the wav or from --labs.
Only boundaries between touching phonemes move; the phonemes, their order and a minimum length stay as they are.
--mode auto (default) switches to "safe" for a file when most of its phonemes are unknown to the model.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import soundfile as sf  # noqa: E402

from mrefinermodel.api import MODES, Refiner  # noqa: E402
from mrefinermodel.data import read_lab  # noqa: E402


def read_map(path):
    pm = {}
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0]
        if "->" in line:
            a, b = (x.strip() for x in line.split("->", 1))
            if a and b:
                pm[a] = b
    return pm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="folder with .wav (and .lab next to them)")
    ap.add_argument("-m", "--model", required=True, help="model folder (config.yaml, phonemes.txt, model.pt)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--labs", help="folder with the .lab files if they are not next to the wav")
    ap.add_argument("-c", "--checkpoint")
    ap.add_argument("--map", help='file with lines "their_phoneme -> our_phoneme"')
    ap.add_argument("--mode", default="auto", choices=MODES)
    ap.add_argument("--min-confidence", type=float, help="override the threshold of the mode")
    ap.add_argument("--max-shift", type=float, help="ms, a boundary never moves farther (default from the model)")
    ap.add_argument("--passes", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()
    ref = Refiner(args.model, args.checkpoint, args.device)
    pm = read_map(args.map) if args.map else {}
    os.makedirs(args.output, exist_ok=True)
    files = [f for f in sorted(os.listdir(args.input)) if f.lower().endswith(".wav")]
    t0, sec, moved, total, safe = time.time(), 0.0, 0, 0, 0
    for w in files:
        base = os.path.splitext(w)[0]
        lab = os.path.join(args.labs or args.input, base + ".lab")
        if not os.path.exists(lab):
            print(f"  no {lab}, skipped")
            continue
        a, sr = sf.read(os.path.join(args.input, w), dtype="float32")
        segs, info = ref.refine(a, sr, read_lab(lab), mode=args.mode, phone_map=pm, min_confidence=args.min_confidence,
                                max_shift_ms=args.max_shift, passes=args.passes, return_info=True)
        sec += len(a) / sr
        moved += info["moved"]
        total += info["boundaries"]
        safe += info["mode"] == "safe"
        with open(os.path.join(args.output, base + ".lab"), "w", encoding="utf-8", newline="\n") as f:
            for s, e, p in segs:
                f.write(f"{int(round(s * 1e7))} {int(round(e * 1e7))} {p}\n")
    print(f"Done: {len(files)} files, {sec / 60:.1f} min of audio in {time.time() - t0:.1f} s -> {args.output}")
    print(f"  boundaries moved: {moved} of {total}; files in safe mode: {safe}")


if __name__ == "__main__":
    main()
