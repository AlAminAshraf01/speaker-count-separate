# Count the talkers, then separate them

**Detect how many people are speaking (1–5) in a noisy mixture, and recover one clean speech
track per speaker.** One Conv-TasNet-style network, one training run, built to survive
Kaggle's free tier.

EEE 402 (AI/ML Laboratory) · Group 10 · Interpretable Acoustic Isolation & Source Separation

```
              ┌──────────────────────────── CountSepNet ───────────────────────────┐
              │                                                                    │
 mix (T,)  ─► │ Conv1d encoder ─► TCN (8 blocks × 3 repeats) ─┬─► masks ─► decoder │─► 6 waveforms
 RMS-normed   │   N=512, L=16                     skip-sum     │                    │   5 speakers
 3 s @ 8 kHz  │                                                └─► count head ────► │   + 1 noise
              └────────────────────────────────────────────────────────────────────┘   P(N = 1..5)
```

5.3 M parameters. The mask head is the only part that depends on N, so five speaker outputs
cost about 4 % more compute than two.

---

## Quick start

If you just want the thing running on Kaggle, do this and read the rest later:

| # | Notebook | Accelerator | Time | Runs |
|---|---|---|---|---|
| 0 | `kaggle_00_build_dataset.ipynb` | **None (CPU)** | ~10 min | once, ever |
| 1 | `kaggle_01_eda_and_leak.ipynb` | **None (CPU)** | 10–20 min | once |
| 2 | `kaggle_02_train.ipynb` | GPU T4 ×2 | 11 h × N sessions | resumable |
| 3 | `kaggle_03_hparam_search.ipynb` | GPU T4 ×2 | 1.5–2.5 h | resumable |
| 4 | `kaggle_04_evaluate.ipynb` | GPU T4 | 10–25 min | after training |
| 5 | `kaggle_05_interpretability.ipynb` | GPU T4 | 15–30 min | after training |
| 6 | `kaggle_06_demo_inference.ipynb` | any | seconds | the demo |

Notebooks 0 and 1 spend **no GPU quota**. Do them first.

Full step-by-step with every click: **[docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md)**.

---

## What is in here

```
speaker-count-separate/
├── README.md                    ← you are here
├── requirements.txt             everything is preinstalled on Kaggle
├── configs/                     base · paper · small · tiny · search
├── src/csnet/                   the library
│   ├── constants.py             8 kHz, 3 s, 5 slots, −30 dB silence floor
│   ├── audio.py                 levels, int16 packing, resampling, STFT
│   ├── pack.py                  Libri2Mix → one flat int16 file per split
│   ├── noise.py                 white/pink/brown/babble/real, byte-reproducible
│   ├── mixing.py                recipes: the frozen test set as a CSV
│   ├── datasets.py              dynamic mixing (train) · frozen recipes (dev/test)
│   ├── model.py                 Conv-TasNet + max-N mask head + noise slot + count head
│   ├── losses.py                rectangular PIT, brute-forced on the GPU
│   ├── metrics.py               SI-SDR(i), P-SI-SNR, confusion matrices
│   ├── baselines.py             IRM/IBM oracles, the naive count predictor
│   ├── checkpoint.py            atomic saves, the session clock, resume discovery
│   ├── engine.py                train/eval loops, AMP, throughput measurement
│   ├── interpret.py             filterbank, mask geometry, ablation
│   └── config.py                YAML + dotted CLI overrides
├── scripts/                     00…09, every one a standalone CLI
├── notebooks/                   ready-to-upload .ipynb (source in notebooks/src/)
├── tests/                       50+ checks, run in 35 s, no pytest required
├── tools/                       notebook builder · fake corpus · test runner
├── docs/                        RUNBOOK · DESIGN · REPORT_CHECKLIST · CONTRACT
└── legacy/                      the four original prototype scripts, preserved
```

### The scripts

| Script | Phase | What it produces |
|---|---|---|
| `00_pack_sources.py` | — | the packed store (~4 GB) |
| `01_make_frozen_sets.py` | — | `recipes_dev.csv`, `recipes_test.csv` — **commit these** |
| `02_eda.py` | **1** | correlation matrices, outliers, the before/after leak figure |
| `03_count_leak_probe.py` | **4** + 5 | the leakage audit *and* the naive-predictor floor |
| `04_train.py` | — | the model. Resumable, budget-aware |
| `05_hparam_search.py` | **3** | speaker-disjoint K-fold search, resumable |
| `06_evaluate.py` | **5a** | `eval_report.md` — paste it straight into the report |
| `07_interpret.py` | **2** + **5b** | filterbank, mask geometry vs N, filter ablation |
| `08_infer.py` | — | any audio in → speaker count + clean tracks out |
| `09_publish_dataset.py` | — | publish the packed store as a real Kaggle Dataset (optional) |

---

## Three ideas that make this work

### 1. Pack the sources; never render an N-speaker corpus

Libri2Mix ships the *isolated* sources, not just the mixtures. Packing every utterance into
one flat `int16` file per split, and mixing on the fly, gives four wins at once:

* **~4 GB instead of ~32 GB** — no Libri3/4/5Mix on disk.
* **One sequential memmap instead of 100k tiny file opens** — the biggest throughput win in
  the project, because Kaggle's input filesystem is slow on small files.
* **Any N**, decided at mix time, so nothing is committed to a speaker count.
* **Noise for free** — added at a random SNR during mixing, so "speech + noise" costs no disk.

Every utterance is stored as its **highest-energy 8-second window**, because a random 3 s crop
of a LibriSpeech utterance often lands in silence, and a silent target makes SI-SDR undefined.

### 2. The count label leaks — and the fix is structural

LibriMix normalises each source to an independent target drawn from U(−33, −25) LUFS and then
**sums** them, so mixture level climbs ~10·log₁₀(N) dB. `min` mode truncates to the shortest of
N sources, so duration falls with N. Measured with a depth-3 tree, 4 classes, chance 25 %:

| Features | Accuracy |
|---|---|
| Mixture level (LUFS) only | **53 %** |
| `min`-mode duration only | 33 % |
| Both, after a 3 s crop + RMS normalisation | **25 % (= chance)** |

So **every model input is a fixed 3 s crop divided by its RMS**. Both cues vanish *by
construction*, not by hoping the network ignores them. `render_recipe()` does the
normalisation as its final step and stores the factor, so even the frozen test set cannot leak.

What survives is the legitimate cue: summing N sparse speech signals makes the mixture less
sparse, so crest factor and kurtosis fall monotonically with N.

A second hazard, documented and avoided: **LibriCount** is built from LibriSpeech *test-clean* —
the same speakers as our test split.

### 3. Exact rectangular assignment, brute-forced on the GPU

Five output slots, `n_src` true sources, `5 − n_src` slots left over. The reference definition
is a rectangular linear assignment, but **scipy never runs in the training loop**: the number
of injective maps from n references into 5 slots is P(5,n) ≤ 120, so enumerating all of them
*is* the optimal assignment, vectorised, with no host synchronisation.

```
n     1    2    3    4    5
P   5   20   60  120  120
```

```python
loss = -matched_si_sdr.mean()                              # separation
     + relu(power_db(leftover_slots) - (-30 dB)).mean()    # surplus slots stay quiet
     + w_noise * noise_slot_term                           # the dedicated noise slot
     + w_count * cross_entropy(count_logits, n_src)        # the counter
```

Four invariants are asserted in `tests/test_losses.py`, and one of them checks the brute force
against `scipy.optimize.linear_sum_assignment` directly:

```bash
python -m csnet.losses
```

```
1. permutation invariance      delta = 0.00e+00   PASS
2. slot-shift invariance       delta = 0.00e+00   PASS
3. leaking mixture into spares rise  = 30.00 dB   PASS
4. perfect estimates           sep = -29.999, count_ce = 0.0000   PASS
```

Invariant 2 is the one that catches real bugs. A loss that is permutation invariant but not
*shift* invariant silently punishes the model for using a high-numbered slot, which looks like
slow convergence rather than a bug.

---

## Surviving a 12-hour session cap

Kaggle gives you 12 hours, then takes the machine away, and `/kaggle/working` is wiped between
sessions unless you **Save Version**.

| Failure | Response |
|---|---|
| Time budget expires | stops **cleanly and exits 0** at 11 h, so Save Version still captures the checkpoint |
| Hard kill (tab closed, OOM) | mid-epoch checkpoint every 400 steps — minutes lost, not hours |
| New session | `find_resume()` searches `/kaggle/working`, then every attached input dataset |
| Corrupt write | every save is `.tmp` + `os.replace`, so a kill mid-write cannot truncate |

**The resume loop:**

1. Run `kaggle_02_train.ipynb`. It stops on its budget and prints a `RESUME` banner.
2. **Save Version → Save & Run All (Commit).** Wait for it to finish.
3. **+ Add Input → Notebook Output →** *this* notebook's latest version.
4. Re-run. It picks up `last.pt` by itself. **No path edits, ever.**

Optimizer moments, scheduler position, AMP scaler and RNG streams all resume exactly.

Before training commits, it **times 20 real steps** and prints how many epochs the remaining
budget actually buys. That measurement replaces any FLOP table — depthwise separable 1-D
convolutions are memory-bound, so peak TFLOPS tells you very little.

---

## Running locally

Everything except full-size training runs fine on a CPU.

```bash
python -m pip install -r requirements.txt
```

Prove the pipeline works end to end without downloading 10 GB — this builds a synthetic
Libri2Mix-shaped corpus and runs all nine scripts against it:

```bash
python tools/make_fake_libri2mix.py --out /tmp/fake --speakers 20 --utts 6
python scripts/00_pack_sources.py --libri2mix_dir /tmp/fake/wav8k/min --out /tmp/store
python scripts/01_make_frozen_sets.py --store /tmp/store --out /tmp/data --n_per_class 20
python scripts/04_train.py --config configs/tiny.yaml --store /tmp/store \
    --recipes_dev /tmp/data/recipes_dev.csv --ckpt_dir /tmp/ckpt --resume none --dry_run
```

Run the test suite (35 s, no pytest needed):

```bash
python tools/run_all_tests.py
```

Rebuild the notebooks after editing `notebooks/src/*.py`:

```bash
python tools/build_notebooks.py
```

---

## Targets, and how to read a bad number

| System · corpus | 2 spk | 3 spk | 4 spk | 5 spk |
|---|---|---|---|---|
| Conv-TasNet · LibriMix 8k min (asteroid) | **14.76** | **11.98** | — | — |
| Conv-TasNet · WSJ0-mix (paper) | 15.3 | 12.7 | — | — |
| SepFormer + dynamic mixing · LibriMix 8k | 20.4 | 19.0 | — | — |
| OR-PIT recursive · WSJ0-mix (SDRi) | 15.0 | 12.9 | 10.6 | — |
| SepEDA transformer · WSJ0-mix, unknown N | 21.1 | 18.4 | 14.4 | 11.6 |

**Gate 2 is the one that matters**: reproduce fixed-N=2 within ~1 dB of **14.76 dB** SI-SDRi on
the *official, untouched* Libri2Mix test set (`06_evaluate.py --libri2mix_dir ...`). Until that
passes, every number at N > 2 is uninterpretable. If you are below ~9 dB at N=3, check in this
order **before** touching the architecture:

1. **Is the assignment actually permuting?** → `python -m csnet.losses`
2. **Are the sources really distinct speakers?** → `scripts/03_count_leak_probe.py`
3. **Did training converge, or stop on the budget?** → `train_log.csv`
4. Only then: more epochs, or a larger config.

Counting 2–5 fully-overlapped talkers is close to solved in the literature (90–99.9 %). The
risk was never the counter — it was that our own data would make the counter look better than
it is. Hence §2 above.

---

## What to say in the report

Full mapping of deliverable → script → figure: **[docs/REPORT_CHECKLIST.md](docs/REPORT_CHECKLIST.md)**.

Report **four numbers, never one**: counting accuracy *with the full confusion matrix* (the
classes are ordinal — 3→4 is not 3→5), P-SI-SNR over the whole test set, SI-SDRi per N on the
*count-correct subset only*, and the naive-predictor floor. Always report SI-SDR **improvement**;
raw input SI-SDR itself falls with N.

State these limitations plainly:

- Only the N=2 row on the official test set is literature-comparable; our N>2 mixtures are ours.
- `min`-mode mixtures are **fully overlapped**, so counting here is one global judgement about
  spectral density. A 97 % result **does not** mean speaker counting is solved — real
  conversation is sparse, and counting there is a diarisation problem.
- The epoch count was a **budget decision** (60–100, not 200) fixed before training, so our
  numbers sit a decibel or two below published baselines.
- Noise is synthetic unless a real noise corpus was attached. **No reverberation anywhere.**
  8 kHz, single channel, anechoic.
- The test set was frozen before training and never regenerated.

The **interpretability work is the contribution**, not a bonus. As N grows the network must
partition the *same* encoder basis among more sources; `07_interpret.py` measures whether mask
overlap rises and sparsity falls, which turns the degradation curve from an assertion into a
mechanism. Framing: **the count head is an interpretability probe** — it reads the same shared
features whose geometry we measure. If the schedule slips, cut training epochs, not this.

---

## Documentation

| Document | For |
|---|---|
| **[docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md)** | every click, every command, every failure mode |
| **[docs/DESIGN.md](docs/DESIGN.md)** | why the pipeline is shaped this way |
| **[docs/REPORT_CHECKLIST.md](docs/REPORT_CHECKLIST.md)** | the five mandated phases → evidence |
| **[docs/CONTRACT.md](docs/CONTRACT.md)** | the internal API spec, if you extend the code |

## References

- Luo & Mesgarani, *Conv-TasNet*, IEEE/ACM TASLP 2019 — https://arxiv.org/abs/1809.07454
- Cosentino, Pariente et al., *LibriMix*, 2020 — https://arxiv.org/abs/2005.11262
- JorisCos/LibriMix — https://github.com/JorisCos/LibriMix
- Zhu, Yeh & Hasegawa-Johnson, *Multi-Decoder DPRNN* (P-SI-SNR) — https://arxiv.org/abs/2011.12022
- Takahashi et al., *Recursive Speech Separation for Unknown Number of Speakers*, Interspeech 2019
- Nachmani, Adi & Wolf, *Voice Separation with an Unknown Number of Multiple Speakers*, ICML 2020
- Chetupalli & Habets, *Speech Separation for an Unknown Number of Speakers Using Transformers*, Interspeech 2022
- Stöter et al., *CountNet*, IEEE/ACM TASLP 2019 — https://github.com/faroit/CountNet

Dataset: Kaggle `unconscious/libri2mix-8khz-min` (CC0). Built from LibriSpeech (CC BY 4.0).

## Dependencies

`numpy · scipy · torch · soundfile · pandas · scikit-learn · matplotlib · tqdm · PyYAML · pyloudnorm`

All preinstalled on Kaggle. **No asteroid, no librosa, no torchaudio, no speechbrain** — Conv-TasNet,
PIT, SI-SDR and the STFT baselines are implemented directly in `src/csnet`, so nothing pins a
torch version and nothing breaks when Kaggle updates its image.
