# KAGGLE RUNBOOK — every click, in order

Written for the **free tier**: 30 GPU-hours/week, 12-hour session cap, 20 GB `/kaggle/working`,
4 vCPU, ~30 GB RAM, T4 ×2 or P100.

Read §0 once. Then follow §1 → §7 in order. §8 is what to do when something breaks.

---

## 0. One-time setup

### 0.1 Get the code onto Kaggle

Two routes. **Route A is better** — you can push fixes and re-clone.

**Route A — GitHub (recommended)**

1. Push this folder to a GitHub repo (public or private).
2. In each notebook's first code cell, set:
   ```python
   REPO_URL = "https://github.com/YOUR-USERNAME/speaker-count-separate.git"
   ```
3. In the notebook's right-hand panel: **Settings → Internet → On**.
   *Internet requires a phone-verified Kaggle account.* If you cannot enable it, use Route B.

**Route B — repo as a dataset (no internet needed)**

1. Zip this folder (excluding `store/`, `runs/`, `*.pt`).
2. Kaggle → **Datasets → New Dataset → Upload** the zip. Title it exactly
   `speaker-count-separate`.
3. In every notebook: **+ Add Input → Datasets →** your `speaker-count-separate`.

The bootstrap cell tries the dataset first, then the clone, so both work unchanged.

### 0.2 Add the corpus

**+ Add Input → Datasets →** search `libri2mix-8khz-min` (by *unconscious*, 9.96 GB, CC0).

Kaggle has changed where attached datasets appear, and may again. Both shapes are handled —
`autodetect_libri2mix()` searches for the *contents* (`<split>/s1`) rather than assuming a path:

| Layout | Mount point |
|---|---|
| older | `/kaggle/input/<slug>/Libri2Mix/wav8k/min` |
| current | `/kaggle/input/datasets/<owner>/<slug>/Libri2Mix/wav8k/min` |

So leave `LIBRI2MIX_DIR = autodetect_libri2mix()` alone. If it ever returns `None`, print the
tree with the diagnostic in §8 and pass the path explicitly with `--libri2mix_dir`.

### 0.3 Optional — real background noise

Without this, noise is synthetic (white / pink / brown / babble from held-out speakers), which
is fine and is the default. To add recorded noise, attach any audio dataset and set `NOISE_DIR`
in notebook 00. Good choices, small first:

| Dataset | Size |
|---|---|
| `mmoreaux/environmental-sound-classification-50` (ESC-50) | ~600 MB |
| `chrisfilo/urbansound8k` | ~6 GB |

Do **not** try to attach WHAM! (~50 GB) — it will not fit alongside everything else.

### 0.4 Upload the notebooks

Kaggle → **Create → New Notebook → File → Import Notebook →** upload from `notebooks/`.
Do all seven now; it takes two minutes and saves confusion later.

---

## 1. Notebook 00 — build the dataset · **CPU** · ~10 min

> **Settings → Accelerator → None.** This notebook must not touch the GPU. Running it on a GPU
> burns quota on `soundfile` and buys you nothing.

**Inputs to attach:** `libri2mix-8khz-min` (+ your noise dataset, if any).

Run all cells. It will:

1. pack ~27,800 train + 6,000 dev + 6,000 test utterances into flat `int16` files (~4 GB),
2. write `data/recipes_dev.csv` and `data/recipes_test.csv`,
3. run the test suite,
4. print the speaker-disjointness audit and the `mix == Σsources + noise` check,
5. play a few mixtures so you can hear that a "3-speaker" mixture has three people in it.

**Expected output near the end:**

```
split      utts   speakers  babble spk  hours  median s  size
train-100  27800  201       50          ~57    8         ~3.1 GB
dev         ~2250  32        8          ~3.7   ~5.9      ~0.2 GB
test        ~2073  32        8          ~3.3   ~5.5      ~0.2 GB

speaker disjointness across splits (must all be 0):
   train-100 & dev        shared speakers:    0  OK
   ...
max |mix - (sum(sources) + noise)| over the test set: <1e-05   OK
```

**Read those numbers before moving on**, because two of them surprise people:

* **The `speakers` column counts *target* speakers only.** The 20 % held out for babble noise
  are the next column. 201 + 50 = 251 and 32 + 8 = 40 are LibriMix's real speaker counts.
* **dev and test hold far fewer utterances than train.** Libri2Mix's dev and test each use
  3,000 mixtures x 2 = 6,000 source *slots*, but LibriSpeech dev-clean and test-clean only
  contain 2,703 and 2,620 distinct utterances — so utterances repeat across mixtures there.
  The packer de-duplicates, which is why you get roughly 2,250 and 2,073. That is correct.
  train-100 is sampled without replacement, so it lands on exactly 27,800.
* **`median s` of 8.0 in train-100** is `CAP_SECONDS` working: over half of train-clean-100's
  utterances are longer than 8 s and were cropped to their highest-energy 8-second window.

Sanity check the size against the hours: `hours x 3600 x 8000 x 2 bytes` should equal the
reported size, because the store is raw int16 at 8 kHz with no header.

### Then publish it — two routes, pick one

Nothing else in the repo cares which you pick; `autodetect_store()` just looks for
`manifest.json` anywhere under `/kaggle/input`.

| | **Route 1 — Save Version** | **Route 2 — a real Dataset** |
|---|---|---|
| Setup | none | an API token, once |
| Cost of publishing | **re-runs the whole notebook** (~45 min) | uploads the files (~10 min) |
| Mounts at | `/kaggle/input/<notebook-slug>/store` | `/kaggle/input/csnet-store` |
| Attach with | + Add Input → Notebook Output | + Add Input → Datasets |
| Versioning | one per notebook run | explicit, with a message |

**Route 1 — Save Version (safe default).**
**Save Version → Save & Run All (Commit).** Wait for it to finish (the header turns green).
`/kaggle/working` becomes this notebook's dataset output. Be aware the commit **re-executes
every cell**, so the 45-minute packing runs a second time. That is normal.

**Route 2 — publish as a real Kaggle Dataset.** Worth the one-time token if you will attach
this store to several notebooks over several weeks: it is created once and never recomputed,
and the mount path is a name you chose.

1. <https://www.kaggle.com/settings> → **API** → **Create New Token** (downloads `kaggle.json`).
2. In the notebook: **Add-ons → Secrets → Add a new secret**, twice —
   `KAGGLE_USERNAME` and `KAGGLE_KEY`, using the two values from inside `kaggle.json`.
   Tick both so they attach.
3. Set `PUBLISH = True` in the Route 2 cell and run it. It calls:

   ```bash
   python scripts/09_publish_dataset.py --dir /kaggle/working/store --slug csnet-store \
       --include_recipes /kaggle/working/data
   ```

4. **Verify it**, because whether subdirectories survive an upload is Kaggle's behaviour, not
   ours. Attach the new dataset, then:

   ```bash
   python scripts/09_publish_dataset.py --verify /kaggle/input/csnet-store
   ```

   It must print `OK` for `manifest.json` and all three splits. If not, re-upload with
   `--dir_mode tar`, or fall back to Route 1.

To push a corrected store later: `--update --message "what changed"`.

From then on every script takes `--store /kaggle/input/csnet-store`.

### Also commit the recipes to git

Download `data/recipes_dev.csv` and `data/recipes_test.csv` from the notebook output and commit
them to your repo. They are a few hundred kB and **they are the evaluation protocol** — the
frozen test set is a list of recipes plus a seed, not audio.

> **Run this notebook once.** Rebuilding the store later changes the frozen sets and makes
> every number you have already reported incomparable.

---

## 2. Notebook 01 — EDA and the leakage audit · **CPU** · 10–20 min

> **Settings → Accelerator → None.** Still no GPU quota.

**Inputs:** **+ Add Input → Notebook Output →** the `00_build_dataset` notebook.

Produces two mandated report sections at once — Phase 1 (EDA) and Phase 4 (leakage audit,
whose output is also the Phase 5 naive-predictor floor).

**The number to write down** is the mitigated tree's accuracy. That is the floor your counting
head must beat. If the raw ARTEFACT-only row is well above chance and the mitigated row is at
chance, the mitigation is working and you can say so with a measurement instead of a claim.

Six figures land in `/kaggle/working/eda/`. Download `leak_before_after.png` — it is the single
figure that justifies the whole data design.

---

## 3. Notebook 02 — train · **GPU T4 ×2** · 11 h per session, resumable

> **Settings → Accelerator → GPU T4 x2.**

**Inputs:** `00_build_dataset` output. *From the second session onward, also this notebook's own
previous output.*

### 3.1 Decide the epoch count before you start

A free weekly quota buys roughly **60–130 epochs** of the pooled N=1..5 set. Plan **60–100, not
200**, and state it in the report as a budget decision. Your numbers will sit a decibel or two
below published baselines. That is expected and defensible; half a joint model plus half an
interpretability study is not.

```python
CONFIG = "configs/paper.yaml"   # or small.yaml if quota is tight
EPOCHS = 60
BATCH_SIZE = 12                 # drop to 8 on CUDA OOM
TIME_BUDGET_H = 11.0
```

### 3.2 The dry run is not optional

The notebook runs `--dry_run` first: 5 steps and one eval batch, about 30 seconds. Never
discover a wrong path an hour into a GPU session.

### 3.3 Read the throughput block

Before training commits, it times 20 real steps:

```
  measured throughput (this replaces the FLOP table)
     420.3 ms per step (batch 12, 36 s of audio)
        8565 steps per hour   ~2.31 TFLOP/s
     budget remaining 10.92 h -> about 93.5 epochs of 1000 steps
```

If that says you will not reach `EPOCHS` this session, that is fine — it resumes. But it is
also your cue to decide, now, whether the quota goes into one joint model or into a clean
fixed-N=3 result plus the interpretability work.

### 3.4 The resume loop

When the budget expires you get:

```
==========================================================================
                                  RESUME
==========================================================================
This session stopped on its time budget, not on an error. Progress is saved.
...
```

**This is the normal, healthy ending.** Then:

1. **Save Version → Save & Run All (Commit).** Wait for it to finish.
2. **+ Add Input → Notebook Output →** *this* notebook, latest version.
3. Re-run.

It finds `last.pt` in the attached dataset by itself. Epoch, optimizer moments, scheduler
position, AMP scaler and RNG streams all resume exactly. Repeat until `EPOCHS` is reached.

> Each session costs one Save Version and nothing else. You do not edit a single path.

### 3.5 Gate 6 — the count head alone

If joint training struggles, freeze the separator and train only the counter (~2 h):

```
--set train.freeze_separator=True loss.w_sep=0.0 loss.w_sil=0.0 loss.w_noise=0.0
```

If that cannot beat ~85 % with the leak mitigated, fall back to fixed-N=3 plus the
interpretability work — and say so in the report.

---

## 4. Notebook 03 — hyperparameter search · **GPU** · 1.5–2.5 h, resumable

**Inputs:** `00_build_dataset` output (+ this notebook's own previous output to resume).

The Phase 3 deliverable. Folds are over **speakers**, not mixtures — a random k-fold over
mixtures would put the same speaker on both sides.

What K-fold buys here is a **variance estimate**, since train and dev are already speaker-disjoint.
The script prints the verdict for you:

```
gap to runner-up: 0.41 dB against a fold sd of 0.83 dB -> WITHIN FOLD NOISE -- do not claim a winner
```

If it says that, report it. A negative result stated clearly beats a winner asserted on one fold.

Results are written after **every** (config, fold) run, so killing the notebook costs at most
one run.

> Optional. If quota is tight, skip this, use `configs/paper.yaml`, and say in the report that
> the search was cut for budget reasons.

---

## 5. Notebook 04 — evaluate · **GPU** · 10–25 min

**Inputs:** `00_build_dataset` output, `02_train` output, and `libri2mix-8khz-min` (for gate 2).

Produces `eval_report.md`, which pastes straight into the report, plus `confusion.png`.

### Gate 2 is the number that matters

```
gate 2 - official Libri2Mix test set (the literature-comparable number)
SI-SDRi        : 14.02 dB (sd 3.41)
published      : 14.76 dB (asteroid)
gap            : -0.74 dB   PASS (within 1 dB)
```

If it fails, check in this order **before** touching the architecture:

1. **Is the assignment permuting?** → `python -m csnet.losses`
2. **Are the sources distinct speakers?** → notebook 01's audit table
3. **Did training converge, or stop on the budget?** → `train_log.csv`
4. Only then: more epochs, or a larger config.

---

## 6. Notebook 05 — interpretability · **GPU or CPU** · 15–30 min

**Inputs:** `00_build_dataset` and `02_train` outputs.

Forward passes only, no training. This is the project's actual contribution — **protect this
time.** If the schedule slips, cut epochs from notebook 02, not this.

Produces the Phase 2 deliverable (the learned filterbank, replacing the STFT) and the Phase 5b
deliverable (mask geometry vs N, count-confidence correlation, filter ablation).

---

## 7. Notebook 06 — the demo · any accelerator · seconds

**Inputs:** `02_train` output (+ `00_build_dataset` to demo on test mixtures).

Any audio file in → "there are K people" + K clean tracks + the isolated noise. Upload your own
recording via **+ Add Input → Upload → New Dataset**.

Worth trying: a single speaker (does it say **1**?), two people talking over each other, a clip
with music underneath.

---

## 8. When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `could not find a packed store` | notebook 00's output is not attached | **+ Add Input → Notebook Output →** `00_build_dataset` |
| `could not find Libri2Mix` | corpus not attached, or a different path | attach `libri2mix-8khz-min`; check the printed path |
| `no frozen dev set at ...` | notebook 00 not finished | run notebook 00 to completion, then Save Version |
| `CUDA out of memory` | batch too large | `BATCH_SIZE = 8`, then 6. Or `--set model.kernel=32` (halves the frames) |
| Training crawls | dataloader starved, or a GPU-less session | check `num_workers=2` and that the Accelerator really is GPU |
| Session died with no checkpoint | ran less than `ckpt_every_steps` | lower it: `--set train.ckpt_every_steps=100` |
| Resume starts from epoch 0 | previous output not attached | **+ Add Input → Notebook Output →** the *training* notebook |
| `git clone` fails | Internet is off | **Settings → Internet → On**, or use Route B in §0.1 |
| Disk full at ~20 GB | store + checkpoints + renders | drop `--render_wav`, or `--set train.keep_last_k=1` |
| Counting is stuck at one class | count term too weak, or too early | raise `loss.w_count` to 1.0; check the loss is falling at all |
| `nan` in the loss | LR too high, or fp16 overflow | `--set train.lr=5e-4`, or `train.amp=False` to confirm the cause |
| Everything looks wrong | — | `python tools/run_all_tests.py` — 35 s, and it localises the problem |

### Quota management

- Notebooks 00 and 01 are **CPU-only**. Never run them with a GPU attached.
- Check your remaining quota at the top-right of any notebook editor before starting a long run.
- Quota resets weekly, not daily. A failed 11-hour run costs 11 hours.
- `--dry_run` costs seconds. Use it every time you change a path.

### The rules that are easy to break

1. **Run notebook 00 exactly once.** Rebuilding the store changes the frozen evaluation sets.
2. **Commit `recipes_dev.csv` and `recipes_test.csv`.** They are the protocol, not a byproduct.
3. **Never evaluate on dev and report it as test.**
4. **Pass gate 2 before believing anything at N > 2.**
5. **Report SI-SDR improvement, per N** — never a single average across speaker counts.
