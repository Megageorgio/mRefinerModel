# mRefinerModel

**English** | [Русский](README.ru.md)

mRefinerModel is a small phoneme **boundary refiner** for singing labels. It runs after any labeller (WFL-ASR, SOFA,
HubertFA, MFA, a rough manual draft): for every boundary between two touching phonemes it looks at a fine
log-mel spectrogram around it (±120 ms, 2.5 ms hop), knows the phonemes on both sides and how long the first stage
made them, and moves the boundary to where it really is. It never changes the phonemes, their order or a minimum
length; a boundary only moves when the refiner is sure enough.

- ~3.5 M parameters, a CNN — much faster than the first stage; on CPU too (about 30× real time on 2 cores).
- Input/output: HTK `.lab` (`start end phoneme`, 100 ns units) or a Python list of `(start, end, phoneme)`.
- A first model for **Russian singing** (see [Models](#models)); you can train one for your own data.

## Results (Russian model)

First stage: a Russian WFL-ASR model, on held-out recordings (every voice represented; mean over voices).
"Capped" is the mean error where one boundary counts at most 100 ms.

| | mean error | capped | ≤ 20 ms | misses > 50 ms |
|---|---|---|---|---|
| WFL-ASR, alignment with lyrics | 13.2 ms | 10.3 ms | 88.4 % | 4.2 % |
| **+ mRefinerModel** | **11.5 ms** | **8.9 ms** | **90.2 %** | **3.1 %** |
| WFL-ASR, recognition without lyrics | 17.7 ms | | 85.6 % | |
| **+ mRefinerModel** | **16.5 ms** | | **87.9 %** | |

Running it twice does not help. Most of the remaining error is between pauses and breaths (SP/AP) and between the
same vowel sung on two notes (a–a), where there is little to see in the spectrum.

### Other languages and labelling styles

The refiner learns *where the annotator of its training data puts boundaries*. On labels of another style (a
different dataset or labelling convention) or with phonemes it does not know, `mode="normal"` can move boundaries
away from that style. `mode="safe"` (used automatically when most phonemes are unknown) moves only confident
boundaries and not far: in our tests on a public multilingual singing set (Russian, English, Japanese and more,
with all phonemes unknown to the model) it practically never made labels worse — correct boundaries moved by
1–5 ms on average, a shifted set of boundaries came out equal or slightly better. It gains little there, though.
For a real gain on another language or style, train a refiner on that data.

## Install

```
pip install -r requirements.txt     # torch, torchaudio, numpy, soundfile, PyYAML
```

## Use

```
python refine.py -i folder_with_wav_and_lab -m models/mrefinermodel-ru -o output_folder
python refine.py -i ... -m ... -o ... --labs folder_with_lab       # labels in another folder
python refine.py -i ... -m ... -o ... --map theirs_to_ours.txt     # other phoneme names: lines "theirs -> ours"
python refine.py -i ... -m ... -o ... --mode safe                  # other language / labels of another style
```

```python
from mrefinermodel import Refiner
r = Refiner("models/mrefinermodel-ru")          # device="cpu" / "cuda"
segments = r.refine(audio, sr, segments)        # [(start, end, phoneme), ...] in seconds
segments, info = r.refine(audio, sr, segments, mode="safe", return_info=True)
# info: {"mode": "safe", "unknown_share": 0.0, "boundaries": 120, "moved": 37}
```

Modes: `auto` (default: `safe` when more than half of the phonemes are unknown to the model, else `normal`),
`normal`, `safe`. `min_confidence` and `max_shift_ms` override the mode's settings.

Check against reference labels:

```
python evaluate.py --ref reference_labs --pred first_stage_labs --refined refined_labs
```

## Train your own

```
python train.py --train data/train --val data/val --run my_refiner
python train.py --train data/train --val data/val --pred-train pred/train --pred-val pred/val --run my_refiner
python export.py runs/my_refiner models/my-refiner --zip my-refiner.zip
```

`--train` / `--val`: `.wav` + reference `.lab`. Better results come with `--pred-train` / `--pred-val`: the same
files labelled by your first stage, so that the refiner learns its real errors. Labels for the training files should
come from a model that did not see them (e.g. train the first stage on 2/3 of the data three times and label the
remaining third each time). Without them it learns from random shifts of the reference (`config/synthetic.yaml`).
File names `voice__anything.wav` give per-voice averages in the training log. Training takes minutes on a GPU.

## Models

| model | language | phonemes | download |
|---|---|---|---|
| mrefinermodel-ru v0.1.0 | Russian (singing) | 51: `a i u e o y ax x ex exh b v g d z k l m n p r s t f h sh ts zh by vy gy dy zy ky ly my ny py ry sy ty fy hy shy ch j SP AP vf cl` | [releases](../../releases) |

Unzip a model into a folder (`config.yaml`, `phonemes.txt`, `model.pt`) and pass it with `-m`.

## License

MIT, see [LICENSE](LICENSE).
