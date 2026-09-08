# DESIGN — why this pipeline is shaped the way it is

Read this before changing anything. Every decision below was forced by one of three
constraints: **Kaggle's free tier**, **a measured data leak**, or **the evaluation protocol
we committed to before training**.

---

## 1. The task

> Given an audio file containing an unknown number of simultaneous talkers (1–5) plus
> background noise, output (a) how many people are talking and (b) one clean speech
> waveform per talker.

That is three problems welded together: counting, separation, and denoising. We solve them
with **one network, one training run**.

```
              ┌──────────────────────────────── CountSepNet ───────────────────────────────┐
              │                                                                            │
 mix (T,) ──► │ Conv1d encoder ─► TCN (8 blocks × 3 repeats) ─┬─► mask head ─► decoder ──► │──► est (6, T)
 RMS-normed   │  N=512, L=16                     skip-sum      │    6 slots                 │    5 speaker slots
 3 s @ 8 kHz  │                                                └─► count head ────────────► │──► count logits (5,)
              │                                                     stats-pool + MLP        │    P(N = 1..5)
              └────────────────────────────────────────────────────────────────────────────┘
```

*Slot 6 is a dedicated **noise slot**.* It gives the network somewhere to put the noise
instead of smearing it across the speech slots, and it costs 66 k parameters (+1.3 %).

---

## 2. Why one max-N model and not five specialists

| Design | Params | TFLOP/pass | Training runs | Verdict |
|---|---|---|---|---|
| Four specialists + separate counter | 20.35 M | 3,279 | 5 | Rejected |
| **One max-N=5 head + count head** | **5.20 M** | 3,368 | **1** | **Chosen** |
| OR-PIT recursive | 5.07 M | ~3,300 | 1 | Strong second |

There is **no FLOP saving** from sharing — the TCN is 92–96 % of the cost and is
N-independent, so five mask outputs cost +4.4 % over two. The savings are elsewhere:
a quarter of the parameters, one checkpoint stream instead of five, and — the real one —
**sample efficiency**: every mixture at every N trains the shared TCN. A dedicated N=5
specialist would see only ~5,700 mixtures.

Given a free-tier GPU quota, "one training run instead of five" is the decisive argument.

---

## 3. The leak, and why every input is a 3-second RMS-normalised crop

Two structural consequences of the LibriMix recipe carry the count label with **no speech
modelling at all**:

* **Level.** Each source is normalised to an independent target drawn from U(−33, −25) LUFS
  and the sources are then *summed*. Incoherent sums add in power, so mixture level climbs
  ~10·log₁₀(N) dB. Class means land ~1.4 dB apart against a within-class spread of 1.0–1.7 dB.
* **Duration.** `min` mode truncates to the shortest of N sources, and the minimum of N
  draws shrinks as N grows.

Measured with a depth-3 decision tree, 5-fold CV, 4 classes (chance = 25 %):

| Features | Accuracy |
|---|---|
| Mixture level (LUFS) only | **53 %** |
| `min`-mode duration only | 33 % |
| Both, after 3 s crop + RMS normalisation | **25 % (= chance)** |

**Consequence, applied everywhere in this codebase:** every model input is a fixed 3 s crop
divided by its RMS. Both cues vanish *by construction*, not by hoping the network ignores
them. `render_recipe()` performs the RMS normalisation as step 5 and stores the scale, so
even the frozen test set cannot leak.

What survives is the legitimate cue: summing N sparse speech signals makes the mixture less
sparse, so crest factor and kurtosis fall monotonically with N. That is what we want the
network to learn.

`scripts/03_count_leak_probe.py` re-measures this on *our* data and its output is
simultaneously the mandated **leakage audit** and the mandated **naive-predictor benchmark**.

---

## 4. Why we pack sources instead of writing N-speaker WAV corpora

The obvious route — render Libri3Mix/4Mix/5Mix to disk — costs ~8 GB per N, needs the
LibriMix generator patched, and pins the training set to one fixed mixture list.

Instead, `00_pack_sources.py` reads the isolated `s1`/`s2` sources that Libri2Mix already
ships and writes **one flat int16 file per split** plus an index CSV:

```
train-100/audio.i16   ~3 GB   27,800 utterances, ≤8 s each (highest-energy window)
dev/audio.i16         ~0.5 GB
test/audio.i16        ~0.5 GB
```

This buys four things at once:

1. **Disk**: ~4 GB total instead of ~32 GB for N = 2,3,4,5.
2. **I/O**: one sequential memmap instead of 100k tiny WAV opens. Kaggle's input filesystem
   is slow on small files; this is the single biggest throughput win in the project.
3. **Dynamic mixing** (Route C): a fresh mixture every `__getitem__`, worth +0.3–0.6 dB in
   the literature, for free.
4. **Noise for free**: noise is synthesised or drawn from the same store at mix time.

The cost is that our N>2 mixtures are *ours*, not the official LibriMix list — so they are
not directly literature-comparable. That is why `06_evaluate.py` keeps a `--libri2mix_dir`
path that evaluates the **official, untouched Libri2Mix test set** for the N=2 gate against
the published 14.76 dB.

### Why the highest-energy 8-second window

A random 3 s crop of a LibriSpeech utterance often lands in silence. A silent target makes
SI-SDR undefined and teaches the network nothing. Storing the loudest 8 s window, then
rejecting crops whose RMS is below 30 % of the utterance RMS, keeps every training target
alive.

---

## 5. Speaker-disjointness — the three audits

1. **Within a mixture.** A train-100 speaker contributes ~111 utterances, so uniform random
   sampling puts the same speaker in a mixture *often*. `sample_recipe()` draws N distinct
   **speaker ids** first, then one utterance each.
2. **Across splits.** Inherited from LibriMix (train-clean-100 / dev-clean / test-clean use
   disjoint LibriSpeech speakers) and re-asserted by `03_count_leak_probe.py`.
3. **Babble noise vs targets.** 20 % of each split's speakers are reserved with
   `role = babble` and are *never* used as separation targets. Without this, "noise" would
   contain a speaker the network is also asked to separate — an invisible leak that would
   inflate both counting and separation scores.

A fourth hazard, documented but not used: **LibriCount**, the obvious off-the-shelf speaker-
counting dataset, is built from LibriSpeech *test-clean* — the same 40 speakers as our test
split. Pre-training a counter on it would put the same speakers on both sides of the line.

---

## 6. The loss: exact rectangular assignment, brute-forced on the GPU

With a fixed 5-slot output and a variable true count N, each of the N references must be
matched to a *distinct* slot, leaving 5−N slots free.

```python
C = -pairwise_si_sdr(est, refs)        # (N_true, 5)
rows, cols = linear_sum_assignment(C)  # each source -> a distinct slot
loss = -matched.mean() \
     + relu(power_db(est[leftover]) - SILENCE_DB).mean() \
     + w_count * cross_entropy(count_logits, N_true)
```

That is the reference definition. **We do not run scipy in the training loop.** The number
of injective maps from N references into 5 slots is P(5,N) ≤ 120, so we enumerate all of
them exactly and take the max — vectorised, on the GPU, no host sync:

| N | P(5,N) |
|---|---|
| 1 | 5 |
| 2 | 20 |
| 3 | 60 |
| 4 | 120 |
| 5 | 120 |

Brute force here **is** the optimal assignment; Hungarian would return the same answer more
slowly, via the CPU. Batch items are grouped by their N (at most 5 groups per batch).

### Why PIT was never the bottleneck

An earlier version of the project record claimed uPIT was the problem because permutations
go 2 → 6 → 24 → 120. That is naive enumeration over *square* permutations and it is not
what costs anything. Measured at batch 24 / 3 s / 8 kHz: permutation *search* costs
0.035–0.087 ms at every N because it runs on a precomputed matrix and never touches
waveforms. The pairwise matrix itself grows 8.4 → 32.5 ms from N=2 to N=5 — **as N², not N!**

The real cost of variable-N is **data volume**: a pooled N=1..5 training set is ~36,000
mixtures per epoch instead of 9,300.

### The silence term

Surplus slots must stay quiet, or the network learns to copy the mixture into them and the
count head becomes meaningless. `SILENCE_DB = −30 dB` relative to the mixture (the
Multi-Decoder DPRNN floor). Verified: the loss is invariant to permuting *and* shifting the
correct outputs among slots, and rises by >20 dB when surplus slots leak the mixture.

---

## 7. Metrics: report four numbers, never one

SI-SDR is undefined when the predicted count is wrong, and that cannot be retrofitted after
training. Fixed before the first run:

1. **Counting accuracy + the full confusion matrix.** The classes are ordinal — 3→4 is not
   the same failure as 3→5, and an average hides that.
2. **P-SI-SNR** over the whole test set — the honest end-to-end number.
   ```
   P-SI-SNR = (L_match + L_pad) / max(N_true, N_pred)
   L_pad    = P_ref × |N_true − N_pred|,   P_ref = −30 dB
   ```
   The N_pred slots used are the N_pred highest-energy speaker slots.
3. **SI-SDRi per N on the count-correct subset only** — isolates separation from counting,
   and is the number that compares to the fixed-N literature.
4. **The naive-predictor row**, so the counter's accuracy has a floor.

Always report SI-SDR **improvement**, not raw SI-SDR: input SI-SDR itself falls with N, so
raw numbers across N are not comparable.

---

## 8. Kaggle: how a 12-hour cap is survived

| Constraint | Response |
|---|---|
| 12 h session cap | `TimeBudget` stops cleanly at `time_budget_h` (default 11.0) and prints a resume banner |
| `/kaggle/working` is wiped between sessions | Checkpoints are saved there, then **Save Version** turns them into a dataset; `find_resume()` searches `/kaggle/input/*/**/last.pt` automatically |
| Hard kill (browser closed, OOM) | Mid-epoch checkpoint every 400 steps — at most a few minutes lost |
| 30 GPU-h/week | Dataset build runs on a **CPU** notebook (no GPU quota spent); throughput is measured in the first 20 steps and the achievable epoch count is printed before training commits |
| 20 GB output limit | Packed store ~4 GB, checkpoints ~60 MB each, `keep_last_k` prunes |
| 4 vCPU | Mixing is crop + scale + sum on pre-packed int16, so 2–3 workers saturate the GPU |
| T4 has no bf16 | fp16 autocast + `GradScaler`; the loss body is forced to fp32 |

The split of work across notebooks matters as much as the code:

```
CPU notebook  ──►  packed store  ──►  Save Version  ──►  private dataset
                                                            │  (read-only, attached)
GPU notebook  ◄─────────────────────────────────────────────┘
   trains, checkpoints to /kaggle/working, Save Version
       │
       └──►  attached as input to the NEXT GPU session  ──►  resume
```

Never regenerate data inside the training notebook. It burns GPU quota on `soundfile`.

---

## 9. Budget arithmetic — decide the epoch count before training, not after

One pass over ~36,000 pooled mixtures ≈ 3,370 TFLOP (fwd+bwd, 1 MAC = 2 FLOP).

| Achieved throughput | Hours for 200 epochs | Epochs in a 30-hour week |
|---|---|---|
| 2 TFLOP/s | 94 h | 64 |
| 4 TFLOP/s | 47 h | 128 |
| 8 TFLOP/s | 23 h | 257 |
| 16 TFLOP/s | 12 h | 513 |

Depthwise separable 1-D convolutions are memory-bound, so expect a small fraction of peak
(T4 ≈ 65 TFLOPS fp16 peak). **Measure one epoch on day one and replace this table.**
`04_train.py` prints the measurement automatically.

Plan **60–100 epochs, not 200**, and state it in the report as a budget decision. Our
numbers will sit a decibel or two below published baselines; that is expected and defensible.

Levers, in order of preference when short on quota:
1. `kernel: 32` instead of 16 — halves the frame count, ~2× faster, costs ~0.5–1 dB.
2. `n_filters: 256` instead of 512.
3. `n_repeats: 2` instead of 3.
4. Fewer N classes (drop N=1 or N=5).

---

## 10. Order of work, with kill conditions

| # | Step | Kill / decision condition |
|---|---|---|
| 1 | `03_count_leak_probe.py` (CPU) | Output = mitigation mandate + naive baseline |
| 2 | **Reproduce fixed-N=2 within ~1 dB of 14.76 dB SI-SDRi** | **Kill: if this fails, stop.** Every downstream number is uninterpretable |
| 3 | Tiny N=3 run (500 mixtures, 5 epochs) — plumbing | Does the loader stack 3 sources, does PIT permute, does loss fall? |
| 4 | Full N=3 → 11–12 dB | The headline result |
| 5 | Pooled N=1..5, **measure one epoch** | **The real decision point.** Fix the epoch count here |
| 6 | Count head alone, frozen separator (~2 h) | Kill: below ~85 % with the leak mitigated → fall back to fixed-N=3 + interpretability |
| 7 | Joint training, rectangular loss | Sanity-check on ten batches: loss invariant to permuting *and* shifting slots |
| 8 | Evaluate: confusion, P-SI-SNR, per-N SI-SDRi, naive baseline | — |
| 9 | Interpretability sweep | Forward passes only. **Protect this time.** |

Where this most plausibly fails is step 5, on arithmetic — not on the counting, not on the
loss. The pooled set is ~3.9× a single-N set. Decide there whether the quota buys one joint
model *or* a clean fixed-N=3 result plus the interpretability work. Both are defensible;
half of each is not.

---

## 11. The interpretability contribution

Framing: **not** "we also built a speaker counter" but "**the count head is an
interpretability probe**". It produces an explicit, readable estimate of N from the *same*
shared features whose mask geometry we were already going to measure.

As N grows, the network must partition the **same** 512-filter encoder basis among more
sources. `07_interpret.py` measures, as functions of N and using forward passes only:

* mask **sparsity** (L1/L2 ratio, Gini),
* mean pairwise mask **overlap** (cosine, min-sum),
* mask distribution **entropy**.

If overlap rises and sparsity falls with N, the degradation curve has a mechanistic
explanation computed from the network's own internals instead of asserted.

Three questions this buys, all answerable on checkpoints we already have:

1. **Does the count head read the mask geometry?** Correlate its per-utterance confidence
   against measured pairwise overlap. If confident counts coincide with well-separated
   masks, we can say *why* the model knows how many speakers there are.
2. **Do miscounts have a signature?** Compare mask overlap and encoder-basis usage on
   utterances where k ≠ N. Two sources sharing one output slot is a mechanistic failure
   explanation, not a shrug.
3. **Which basis functions carry the count?** Ablate encoder filters and watch counting
   accuracy fall — the proposal's "extract the filterbank and explain performance"
   deliverable, with a scalar to attribute against.

This needs no extra training. **It is the part to protect when time runs short.**

---

## 12. Targets

| System · corpus | 1 spk | 2 spk | 3 spk | 4 spk | 5 spk |
|---|---|---|---|---|---|
| Conv-TasNet · LibriMix 8k min (asteroid) | — | 14.76 | 11.98 | — | — |
| Conv-TasNet · WSJ0-mix (paper) | — | 15.3 | 12.7 | — | — |
| SepFormer + dynamic mixing · LibriMix 8k | — | 20.4 | 19.0 | — | — |
| OR-PIT recursive · WSJ0-mix (SDRi) | — | 15.0 | 12.9 | 10.6 (zero-shot) | — |
| SepEDA transformer · WSJ0-mix, unknown N | — | 21.1 | 18.4 | 14.4 | 11.6 |

**Our target: 11–12 dB SI-SDRi on N=3, clean.** Below ~9 dB, suspect the loss (is the
assignment actually permuting?) or the data (are the three sources really three distinct
speakers?) *before* touching the architecture.

Counting targets, published on fully-overlapped mixtures: 90–99.9 %. Note the honest
caveat for the report — in `min` mode every source is active for the whole mixture, so
counting is one global judgement about spectral density, not a moment-to-moment decision.
Real conversation is sparse and counting there is a diarisation problem. **A 97 % result on
this data does not mean we solved speaker counting.**

A reportable crossover worth keeping: on LibriMix, Conv-TasNet *beats* IRM/IBM at 2 speakers
but *loses* to them at 3 — which is exactly why the ideal-mask baselines stay in the report.

---

## 13. Known limitations, stated up front

* Our N>2 mixtures are ours, not official LibriMix — comparable within this project, not to
  the literature. N=2 on the official test set is the anchor.
* Fully-overlapped `min`-mode mixtures are the easy case for counting (see §12).
* Noise is synthetic (white/pink/brown/babble) unless a real noise corpus is attached.
  Babble is drawn from held-out speakers of the same split, so it is realistic in spectrum
  but not in room acoustics. There is **no reverberation** anywhere in this pipeline.
* 8 kHz, single channel, anechoic. Nothing here transfers to far-field or multi-channel
  without retraining.
* The count head is trained and evaluated on 3 s crops. Long-file counting in
  `08_infer.py` aggregates per-window predictions and is *not* separately validated.
