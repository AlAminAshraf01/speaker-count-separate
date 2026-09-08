# REPORT CHECKLIST — the five mandated phases → what produces the evidence

EEE 402 (AI/ML Laboratory) · Group 10. Each row names the artefact you paste into the
report and the exact script that generates it. Nothing here needs a re-run of training.

---

## Phase 1 — EDA: correlation matrices, variance, outliers

**Produced by:** `scripts/02_eda.py` → `eda/` figures + `eda_report.json`

| Deliverable | Figure / table |
|---|---|
| Cross-source correlation matrices, N×N per mixture | `eda/corr_matrix_N{n}.png` |
| Distribution of max pairwise \|corr\| → the hard mixtures | `eda/corr_hist.png` |
| Input SI-SDR distribution per N | `eda/input_sisdr_per_n.png` |
| Segment-length + level distributions **before vs after** mitigation | `eda/leak_before_after.png` |
| Crest factor / kurtosis / spectral flatness vs N (the legitimate cue) | `eda/density_cues_vs_n.png` |
| Outlier class: same-sex / similar-pitch groups | `eda/outlier_pairs.png` |
| Per-split speaker and utterance counts | `eda_report.json → splits` |

**The sentence to write:** input SI-SDR falls with N, mean pairwise correlation rises, and
the same-sex/similar-F0 subset sits in the low tail of every separation metric — those are
the outliers, and they are outliers for an acoustic reason, not a bookkeeping one.

---

## Phase 2 — Custom feature extraction / domain transform

**Produced by:** `src/csnet/model.py` (the learned encoder) + `scripts/07_interpret.py`

The deliverable is that we **replace the STFT with a learned 1-D convolutional encoder** —
Conv-TasNet's core idea and this project's "custom spectral transformation".

| Deliverable | Figure |
|---|---|
| The learned filterbank, time domain, sorted by centre frequency | `interpret/filterbank_time.png` |
| Filter magnitude responses (FFT) | `interpret/filterbank_fft.png` |
| Centre-frequency distribution vs a mel scale | `interpret/centre_freq_vs_mel.png` |
| Optional: does optimal `n_filters`/`kernel` shift with N? | `search_results.json` |

**The sentence to write:** the encoder is not an STFT — it learns a non-uniform, roughly
log-spaced basis with a low-frequency emphasis, and that basis is *shared* across all
speaker slots, which is what makes §5 (interpretability) a real question.

---

## Phase 3 — Structured hyperparameter search + K-fold

**Produced by:** `scripts/05_hparam_search.py` → `search_results.json` + `best.yaml`

* Search space: TCN depth `X`, repeats `R`, bottleneck `B`, `n_filters`, kernel `L`, `lr`
  (defined in `configs/search.yaml`).
* **K-fold is speaker-disjoint** over the dev speaker pool (K = 3 by default) — a random
  k-fold over mixtures would put the same speaker on both sides and is not a valid fold here.
* Resumable: results append after every run, restarts skip finished `(config, fold)` pairs.

| Deliverable | Where |
|---|---|
| Ranked table of configs × folds with mean ± std | printed + `search_results.json` |
| The chosen configuration | `best.yaml` |
| Budget statement | see below |

**The sentence to write:** state explicitly that the search is *coarse* (N configs × K folds
× S steps) because of the free-tier quota, and that it was budgeted before running rather
than truncated after. A stated budget decision is defensible; an unexplained small search
is not.

---

## Phase 4 — Data-leakage audit

**Produced by:** `scripts/03_count_leak_probe.py` → `leak_report.json`

Four audits, three structural and one measured:

| # | Hazard | Result to report |
|---|---|---|
| 1 | Same speaker twice **inside** a mixture | assertion table, must be 0 |
| 2 | A speaker crossing **train/dev/test** | assertion table, must be 0 |
| 3 | Babble noise reusing a **target** speaker | assertion table, must be 0 (20 % of speakers reserved, `role=babble`) |
| 4 | The **count label leaking out of the mixing recipe** | the table below |

| Features given to a depth-3 tree (5-fold CV) | Accuracy |
|---|---|
| Mixture level (LUFS) only | ~53 % |
| `min`-mode duration only | ~33 % |
| Both, after 3 s crop + RMS normalisation | ~25 % (= chance at 4 classes) |

**The sentence to write:** the count label was recoverable from a single scalar with no
acoustics in it at roughly twice chance; the mitigation (fixed 3 s crop + RMS normalisation)
removes both cues *by construction*, and it is applied to training, validation and test
alike. Report the *mitigated* number as the floor the counting head must beat.

Also report the hazard you avoided: **LibriCount** is built from LibriSpeech test-clean —
the same 40 speakers as our test split — so it was not used.

---

## Phase 5 — Benchmarking vs naive baselines + interpretability

**Produced by:** `scripts/06_evaluate.py` → `eval_report.md` / `.json`,
and `scripts/07_interpret.py` → `interpret_report.json` + figures

### 5a. Benchmarking

Report **four numbers, never one**:

| # | Metric | Where |
|---|---|---|
| 1 | Counting accuracy **+ full confusion matrix** (classes are ordinal) | `eval/confusion.png` |
| 2 | P-SI-SNR over the whole test set | `eval_report.md` |
| 3 | SI-SDRi **per N, on the count-correct subset only** | `eval_report.md` |
| 4 | Naive-predictor row (decision tree) | from Phase 4 |

Baseline rows that must appear in the same table:

| Baseline | Meaning |
|---|---|
| Mixture as estimate | 0 dB by definition — the floor |
| IBM oracle | ideal binary mask upper bound |
| IRM oracle | ideal ratio mask upper bound |
| Naive count predictor | the counting floor |
| Published Conv-TasNet (14.76 dB @ N=2, 11.98 dB @ N=3) | the literature anchor |

Plus a clean-vs-noisy split and an SNR-binned breakdown, since the input is noisy speech.

**The crossover to keep:** on LibriMix, Conv-TasNet beats IRM/IBM at 2 speakers but loses to
them at 3. That is a finding, not an embarrassment — it is why the ideal-mask baselines stay.

### 5b. Interpretability — the actual contribution

| Question | Evidence | Figure |
|---|---|---|
| How does mask geometry change with N? | sparsity ↓, pairwise overlap ↑, entropy ↑ | `interpret/mask_stats_vs_n.png` |
| Does the count head read that geometry? | corr(count confidence, mask overlap) | `interpret/conf_vs_overlap.png` |
| Do miscounts have a signature? | overlap on k≠N vs k=N utterances | `interpret/miscount_geometry.png` |
| Which filters carry the count? | ablation curve: accuracy vs #filters zeroed | `interpret/filter_ablation.png` |

**The sentence to write:** as N grows the network must partition the *same* 512-filter basis
among more sources; overlap rises and sparsity falls, so the degradation curve has a
mechanistic explanation computed from the network's own internals rather than asserted.
Framing: the count head is an interpretability probe, not a second project.

---

## Honesty checklist — say these out loud in the report

- [ ] Our N>2 mixtures are **ours**, not official LibriMix. Only the N=2 result on the
      untouched official test set is literature-comparable.
- [ ] `min`-mode mixtures are **fully overlapped**, so counting is one global judgement about
      spectral density. A 97 % result here **does not** mean speaker counting is solved;
      real conversation is sparse and counting there is a diarisation problem.
- [ ] Epoch count was a **budget decision** (60–100, not 200), fixed before training, and our
      numbers therefore sit a decibel or two below published baselines.
- [ ] Noise is synthetic (white/pink/brown/babble from held-out speakers) unless a real noise
      corpus was attached. **No reverberation anywhere.** 8 kHz, single channel, anechoic.
- [ ] The counting head is trained and evaluated on 3 s crops; the long-file aggregation in
      `08_infer.py` is a demo, not a validated result.
- [ ] Report SI-SDR **improvement**, not raw SI-SDR — input SI-SDR itself falls with N.
- [ ] The test set was frozen before training and never regenerated (`recipes_test.csv` +
      seed are committed to the repo).

---

## Assembly order for the write-up

1. Run `02_eda.py` → Phase 1 figures.
2. Run `03_count_leak_probe.py` → Phase 4 table + the Phase 5 naive row. *(CPU only, no
   GPU quota.)*
3. Paste the filterbank figures from `07_interpret.py` → Phase 2.
4. Paste `search_results.json`'s ranked table → Phase 3.
5. Paste `eval_report.md` wholesale → Phase 5a.
6. Paste the four interpretability figures → Phase 5b.
7. Copy the honesty checklist above into the limitations section, converted to prose.
