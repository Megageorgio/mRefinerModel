"""
Compare labels with reference labels: how far the boundaries are, and whether refining helped.

  python evaluate.py --ref reference_labs --pred first_stage_labs [--refined refined_labs]

All folders hold .lab files with the same names (HTK, 100 ns units). Boundaries are compared where the phoneme
sequences match (Levenshtein alignment), so labels with a few different phonemes can be compared too.
With --refined it also tells how many boundaries got better / worse by more than 5 ms.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from mrefinermodel.data import align_pairs, read_lab  # noqa: E402


def stats(err_ms):
    e = np.asarray(err_ms)
    return (f"{len(e)} boundaries: mean {e.mean():.1f} ms, capped at 100 ms {np.minimum(e, 100).mean():.1f}, "
            f"median {np.median(e):.1f}; <=10 ms {np.mean(e <= 10) * 100:.1f}%, <=20 ms {np.mean(e <= 20) * 100:.1f}%, "
            f">50 ms {np.mean(e > 50) * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--refined")
    args = ap.parse_args()
    before, after, skipped = [], [], 0
    for f in sorted(os.listdir(args.ref)):
        if not f.endswith(".lab") or not os.path.exists(os.path.join(args.pred, f)):
            continue
        ref, pred = read_lab(os.path.join(args.ref, f)), read_lab(os.path.join(args.pred, f))
        rf = None
        if args.refined:
            path = os.path.join(args.refined, f)
            rf = read_lab(path) if os.path.exists(path) else None
            if rf is None or [p for _, _, p in rf] != [p for _, _, p in pred]:
                skipped += 1
                continue
        m = {j: i for i, j in align_pairs([p for _, _, p in ref], [p for _, _, p in pred])}
        for j in range(1, len(pred)):
            i = m.get(j)
            if i is not None and i > 0 and m.get(j - 1) == i - 1:
                before.append(abs(pred[j][0] - ref[i][0]) * 1000)
                if rf is not None:
                    after.append(abs(rf[j][0] - ref[i][0]) * 1000)
    if not before:
        sys.exit("No matching files")
    print("first stage:", stats(before))
    if after:
        b, a = np.array(before), np.array(after)
        print("refined:    ", stats(a))
        print(f"  better by >5 ms: {np.mean(b - a > 5) * 100:.1f}%, worse by >5 ms: {np.mean(a - b > 5) * 100:.1f}%")
    if skipped:
        print(f"  ({skipped} files skipped: no refined file or different phonemes)")


if __name__ == "__main__":
    main()
