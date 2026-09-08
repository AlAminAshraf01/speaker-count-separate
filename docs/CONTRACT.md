# CONTRACT — internal API spec (authoritative)

Every module in `src/csnet/` and every script in `scripts/` MUST conform to this file exactly.
Signatures, key names, dtypes, shapes and file formats are binding. If something is
under-specified, choose the simplest option and add a `# CONTRACT-EXT:` comment.

Project: EEE 402 Group 10 — "count the talkers, then separate them, under noise".
Target platform: **Kaggle free tier** (T4 x2 or P100, 12 h session cap, 30 GPU-h/week,
20 GB `/kaggle/working`, 4 vCPU, ~30 GB RAM). Everything must be resumable.

---

## 0. Global constants (in `csnet/constants.py`)

```python
SR              = 8000       # Hz, everything is 8 kHz
SEG_SECONDS     = 3.0        # training/eval segment
SEG_LEN         = 24000      # SR * SEG_SECONDS
CAP_SECONDS     = 8.0        # max stored excerpt per source utterance
CAP_LEN         = 64000
MAX_N_SRC       = 5          # speaker slots
N_CLASSES       = 5          # count classes for N in N_LIST
N_LIST          = (1, 2, 3, 4, 5)
SILENCE_DB      = -30.0      # dB rel. mixture; leftover-slot floor & P-SI-SNR pad
EPS             = 1e-8
INT16_SCALE     = 32767.0
```

`n_src -> class index` is `N_LIST.index(n_src)`. Provide `n_to_class(n)` and `class_to_n(c)`.

---

## 1. Audio conventions

* All waveforms in memory: `np.float32` (numpy) or `torch.float32`, mono, shape `(T,)` or `(..., T)`.
* On disk in the packed store: `int16`, `x_i16 = clip(round(x*32767), -32768, 32767)`.
* **RMS normalisation** (`csnet.audio.rms_normalize`) divides by `sqrt(mean(x**2)) + EPS`.
  This is mandatory on every model input — it is the anti-leak mitigation.
* Never resample above/below 8 kHz internally.

---

## 2. Packed source store  (`csnet/pack.py`)

Written once by `scripts/00_pack_sources.py`, then published as a Kaggle dataset.

```
<store_root>/
  manifest.json
  train-100/audio.i16      # flat raw int16, C-order, no header
  train-100/index.csv
  dev/audio.i16
  dev/index.csv
  test/audio.i16
  test/index.csv
  noise/audio.i16          # optional, only if --noise_dir given
  noise/index.csv
```

`index.csv` columns (header row required, comma separated):

```
utt_id,speaker_id,offset,length,role
```

* `utt_id`   — LibriSpeech utterance id, e.g. `4077-13754-0001` (unique within a split)
* `speaker_id` — string, e.g. `4077`
* `offset`   — sample offset into `audio.i16`
* `length`   — number of int16 samples (`<= CAP_LEN`)
* `role`     — `target` or `babble`  (babble speakers are reserved for noise, never separated)

`noise/index.csv` uses the same columns with `speaker_id = <category>` and `role = noise`.

`manifest.json` keys:
```json
{"sr":8000,"cap_len":64000,"created_utc":"...","seed":72,
 "source_dataset":"/kaggle/input/libri2mix-8khz-min/Libri2Mix/wav8k/min",
 "splits":{"train-100":{"n_utt":27800,"n_speakers":251,"n_babble_speakers":50,
                        "total_samples":123456789,"audio_bytes":246913578}},
 "babble_frac":0.2,"noise_dir":null,"csnet_version":"1.0.0"}
```

### `class SourceStore`

```python
class SourceStore:
    def __init__(self, root: str, split: str, mmap: bool = True): ...
    # attributes
    utt_ids: list[str]
    speaker_ids: np.ndarray          # dtype '<U16', len n_utt
    lengths: np.ndarray              # int64
    offsets: np.ndarray              # int64
    roles: np.ndarray                # '<U8'
    target_idx: np.ndarray           # int64 indices where role == 'target'
    babble_idx: np.ndarray           # int64 indices where role == 'babble'
    by_speaker: dict[str, np.ndarray]   # speaker -> int64 utterance indices (targets only)
    speakers: list[str]                 # sorted target speakers

    def get(self, i: int, start: int = 0, n: int | None = None) -> np.ndarray:
        """float32 in [-1,1], length min(n, length[i]-start); n=None -> to the end."""
    def __len__(self) -> int: ...
```

`SourceStore` must work with `mmap=True` (np.memmap, low RAM) and `mmap=False`
(whole array in RAM — used on Kaggle where the train store is ~3 GB and RAM is 30 GB).

### Packer

```python
def pack_split(libri2mix_min_dir: str, split: str, out_root: str, *,
               cap_len: int = CAP_LEN, babble_frac: float = 0.2,
               seed: int = 72, limit: int | None = None,
               progress: bool = True) -> dict
```
* Reads `<libri2mix_min_dir>/<split>/s1/*.wav` and `.../s2/*.wav`.
* Filename `<uttA>_<uttB>.wav`; the file in `sK` holds utterance `utt_ids[K-1]`.
* De-duplicates by `utt_id` (an utterance can appear in several pairs — keep the longest copy).
* For each utterance, keep the **highest-energy contiguous `cap_len` window**
  (moving RMS with hop 800 samples); if `length <= cap_len`, keep everything.
* Speaker split into `target` / `babble`: sort speakers, shuffle with `seed`, take the
  first `round(babble_frac * n_speakers)` as `babble`. Deterministic.
* Returns the per-split manifest dict.

`pack_noise_dir(noise_dir, out_root, *, cap_len, seed, limit)` — same, for a folder tree of
wav/flac/ogg/mp3; resample to 8 kHz mono with `scipy.signal.resample_poly`.

---

## 3. Noise bank (`csnet/noise.py`)

```python
class NoiseBank:
    def __init__(self, store: SourceStore | None,        # babble source (same split!)
                 real: SourceStore | None = None,        # optional real-noise store
                 kinds: tuple[str, ...] = ("white","pink","brown","babble","real"),
                 weights: tuple[float, ...] | None = None): ...

    def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, str, int]:
        """-> (noise float32 (n,), kind, noise_id)  — unit RMS, zero mean."""

    def render(self, kind: str, noise_id: int, n: int) -> np.ndarray:
        """Deterministic re-creation of exactly the noise `sample` produced.
           MUST be byte-reproducible given (kind, noise_id, n)."""
```

* `white`  — Gaussian, `np.random.default_rng(noise_id)`.
* `pink`, `brown` — 1/f and 1/f^2 shaped in the FFT domain, same seeding rule.
* `babble` — sum of `K = 4 + (noise_id % 5)` random excerpts from `store.babble_idx`,
  chosen with `np.random.default_rng(noise_id)`; each RMS-normalised then summed.
* `real`   — one excerpt from `real` store, index/offset chosen by `np.random.default_rng(noise_id)`.
* Every returned noise is RMS-normalised to 1.0 and mean-removed.
* `render` must not consume the caller's rng — it builds its own from `noise_id`.

---

## 4. Mixture recipes (`csnet/mixing.py`)

A recipe is a plain dict / CSV row. This is the *frozen test set* artefact.

```
mix_id,n_src,utt_idx,crop_start,gain_db,noise_kind,noise_id,snr_db,scale
```
* `utt_idx`     — `|`-joined int indices into the split's `SourceStore` (length `n_src`)
* `crop_start`  — `|`-joined int sample offsets, one per source
* `gain_db`     — `|`-joined float linear-gain-in-dB applied to each RMS-normalised source
* `noise_kind`  — one of the bank kinds, or `none`
* `noise_id`    — int (ignored when `noise_kind == none`)
* `snr_db`      — float, speech-mixture-to-noise ratio (ignored when `none`)
* `scale`       — float, final de-clip / RMS scale, see below

```python
def sample_recipe(store, bank, n_src, rng, *, seg_len=SEG_LEN,
                  gain_db_range=(-5.0, 5.0), snr_db_range=(0.0, 20.0),
                  p_clean=0.2, mix_id=None, min_crop_rms_ratio=0.3,
                  max_tries=8) -> dict
```
* Draw `n_src` **distinct speaker ids** from `store.speakers`, then one utterance each.
* Draw `crop_start` uniformly so the crop fits; retry (up to `max_tries`) until the crop RMS
  is `>= min_crop_rms_ratio * utterance RMS` (rejects silent crops). Pad with zeros only if
  the utterance itself is shorter than `seg_len`.
* `gain_db ~ U(gain_db_range)` per source.
* With prob `p_clean` -> `noise_kind='none'`, else `bank.sample`.

```python
def render_recipe(recipe: dict, store, bank, *, seg_len=SEG_LEN) -> dict
```
Returns
```python
{"mix":      np.float32 (seg_len,),    # RMS-normalised, this is the model input
 "sources":  np.float32 (n_src, seg_len),  # SAME scale as mix (mix = sum(sources)+noise)
 "noise":    np.float32 (seg_len,),    # zeros if noise_kind == 'none'
 "n_src":    int,
 "mix_id":   str}
```
**Invariant (must be asserted in tests):**
`max|mix - (sources.sum(0) + noise)| < 1e-5`.

Construction order (fixed, do not change — reproducibility depends on it):
1. crop each source, subtract mean, RMS-normalise, multiply by `10**(gain_db/20)`
2. `speech = sum(sources)`
3. if noisy: `noise = bank.render(kind, id, seg_len)`; scale it to
   `noise *= rms(speech) / (rms(noise)+EPS) * 10**(-snr_db/20)` ; else `noise = 0`
4. `mix = speech + noise`
5. `scale = 1 / (rms(mix) + EPS)`  — then multiply `mix`, every source, and `noise` by `scale`.
   (This is the RMS normalisation; it kills the level leak by construction.)
   When rendering a *stored* recipe, use the stored `scale` rather than recomputing it.

```python
def write_recipes(rows: list[dict], path: str) -> None
def read_recipes(path: str) -> list[dict]      # values already parsed to int/float/list
```

---

## 5. Datasets (`csnet/datasets.py`)

```python
class DynamicMixDataset(torch.utils.data.Dataset):
    """Infinite-ish training set: a fresh recipe per __getitem__."""
    def __init__(self, store, bank, *, n_list=N_LIST, steps=200_000,
                 seg_len=SEG_LEN, seed=72, max_n_src=MAX_N_SRC,
                 gain_db_range=(-5.,5.), snr_db_range=(0.,20.), p_clean=0.2,
                 n_weights=None): ...
    def __len__(self): return self.steps

class FrozenMixDataset(torch.utils.data.Dataset):
    """Reads a committed recipe CSV. Used for dev + test. Never regenerated."""
    def __init__(self, store, bank, recipes_path, *, seg_len=SEG_LEN,
                 max_n_src=MAX_N_SRC): ...
```

Both `__getitem__` return **exactly** this dict of torch tensors:
```python
{"mix":     torch.float32 (seg_len,),
 "refs":    torch.float32 (MAX_N_SRC, seg_len),   # zero-padded past n_src
 "noise":   torch.float32 (seg_len,),
 "n_src":   torch.int64 ()  scalar,
 "cls":     torch.int64 ()  scalar,               # n_to_class(n_src)
 "is_noisy":torch.int64 ()  scalar,               # 0/1
 "mix_id":  str}
```
`str` fields survive default collate as a list — that is fine and expected.
Seeding: `DynamicMixDataset` derives its rng per item as
`np.random.default_rng((seed, epoch_salt, idx))` where `epoch_salt` is set by
`set_epoch(e)`. Workers must not produce identical mixtures.

---

## 6. Model (`csnet/model.py`)

```python
@dataclass
class ModelConfig:
    n_filters: int = 512      # N
    kernel: int = 16          # L (encoder window, samples)
    bottleneck: int = 128     # B
    hidden: int = 512         # H
    skip: int = 128           # Sc
    conv_kernel: int = 3      # P
    n_blocks: int = 8         # X
    n_repeats: int = 3        # R
    max_n_src: int = 5
    predict_noise: bool = True
    n_classes: int = 5
    mask_act: str = "relu"    # 'relu' | 'sigmoid' | 'softmax'
    norm: str = "gLN"         # 'gLN' | 'cLN'
    count_hidden: int = 128
    count_dropout: float = 0.1
    causal: bool = False

class CountSepNet(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    @property
    def n_slots(self) -> int:          # max_n_src + int(predict_noise)
    def forward(self, mix: torch.Tensor, return_internals: bool = False) -> dict
```

`mix` is `(B, T)` **or** `(B, 1, T)`. Output dict:
```python
{"est":          (B, n_slots, T),     # slot MAX_N_SRC (last) is the noise slot iff predict_noise
 "count_logits": (B, n_classes),
 # only when return_internals=True:
 "masks":        (B, n_slots, n_filters, F),
 "enc":          (B, n_filters, F),
 "feat":         (B, skip, F)}        # the TCN skip-sum the count head reads
```
* Output length **must equal the input length** `T` (pad the encoder, crop the decoder).
* Count head: `Conv1d(skip, count_hidden, 1) -> PReLU -> stats-pool(mean‖std over F)
  -> Linear(2*count_hidden, count_hidden) -> ReLU -> Dropout -> Linear(count_hidden, n_classes)`.
* `torch.nn.DataParallel`-safe: no python-scalar branching on batch content, dict of
  tensors all with batch as dim 0.
* Provide `def count_params(self) -> int` and `def flops_per_second_of_audio(self) -> float`.

Presets in `csnet/model.py`:
```python
PRESETS = {"paper":  ModelConfig(),                                    # ~5.2 M
           "small":  ModelConfig(n_filters=256, kernel=32, hidden=256, n_blocks=8, n_repeats=2),
           "tiny":   ModelConfig(n_filters=128, kernel=32, hidden=128, n_blocks=6, n_repeats=2)}
```

---

## 7. Loss (`csnet/losses.py`)

```python
def pairwise_si_sdr(est: torch.Tensor, ref: torch.Tensor, eps=EPS) -> torch.Tensor:
    """est (B,S,T), ref (B,R,T) -> (B,R,S) SI-SDR in dB of est[s] against ref[r]."""

class RectangularPITLoss(nn.Module):
    def __init__(self, max_n_src=MAX_N_SRC, predict_noise=True, n_classes=N_CLASSES,
                 w_sep=1.0, w_sil=1.0, w_count=0.5, w_noise=0.2,
                 silence_db=SILENCE_DB, label_smoothing=0.05,
                 clamp_si_sdr: float | None = 30.0): ...
    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        """-> (scalar loss, {'loss','sep','sil','count','noise','sisdr','acc'} floats)"""
```

Rules:
* Only the first `max_n_src` slots are speaker slots. Assignment is over
  **all injective maps** from the `n_src` references to those `max_n_src` slots
  (`P(5,n) <= 120`), evaluated exactly and vectorised on GPU — no scipy in the training loop.
  Precompute per-`n` `arrangements` (LongTensor `(P,n)`) and `complement` (BoolTensor `(P,max_n_src)`)
  as registered buffers.
* Batch items are grouped by `n_src`; each group is scored with its own table.
* `sep = -mean(matched SI-SDR)`, optionally clamped with the standard
  `-10*log10(1 + 10**(-sisdr/10))` soft clamp when `clamp_si_sdr` is not None
  (prevents a single easy example dominating). Document whichever you implement.
* `sil = mean over leftover speaker slots of relu(10*log10(P_slot/P_mix + eps) - silence_db)`.
* `noise` term: if `predict_noise`, negative SI-SDR of slot `max_n_src` against
  `batch['noise']` for noisy items, and the silence penalty for clean items.
* `count = cross_entropy(count_logits, batch['cls'], label_smoothing=...)`.
* `total = w_sep*sep + w_sil*sil + w_count*count + w_noise*noise`.
* The returned dict values are python floats (`.item()`), for logging only.

**Test-mandated invariants** (`tests/test_losses.py`):
1. Loss is unchanged when the correct estimates are permuted among slots.
2. Loss is unchanged when they are *shifted* to different slots (rectangular case).
3. Loss rises by >20 dB when the leftover slots are filled with the mixture.
4. Perfect estimates + correct count give `sep` ≈ `-clamp_ceiling` and `count` ≈ 0.

---

## 8. Metrics (`csnet/metrics.py`)

```python
def si_sdr(est, ref, eps=EPS) -> np.ndarray | torch.Tensor   # (...,) over last dim T
def si_sdr_improvement(est, ref, mix) -> ...
def best_permutation(est, refs, n_src) -> tuple[np.ndarray, np.ndarray]
    """scipy Hungarian on the (n_src, S) matrix -> (rows, cols)."""
def p_si_snr(est_slots, refs, n_true, n_pred, p_ref=SILENCE_DB) -> float
    """Multi-Decoder-DPRNN P-SI-SNR. The n_pred slots used are the n_pred
       highest-energy speaker slots. (L_match + L_pad) / max(n_true, n_pred)."""
def count_report(y_true, y_pred, n_list=N_LIST) -> dict
    """-> {'accuracy','mae','per_class_acc':{n:acc},'confusion': (C,C) int array}"""
```

---

## 9. Baselines (`csnet/baselines.py`)

```python
def ideal_ratio_mask_sisdri(mix, sources, sr=SR, n_fft=256, hop=64, mode='irm') -> np.ndarray
    """mode in {'irm','ibm'}; returns per-source SI-SDRi of the oracle-masked mixture."""
def naive_count_features(x, sr=SR) -> dict     # duration,rms_db,crest,kurtosis,flatness,zcr,flux
def naive_count_baseline(X, y, max_depth=3, n_folds=5) -> dict
    """depth-limited decision tree, stratified CV -> {'accuracy','confusion','feature_importance'}"""
```

---

## 10. Checkpointing (`csnet/checkpoint.py`)

```python
class TimeBudget:
    def __init__(self, hours: float): ...
    def elapsed_h(self) -> float
    def remaining_h(self) -> float
    def expired(self, margin_min: float = 20.0) -> bool

def save_checkpoint(path, *, model, optimizer, scheduler, scaler, epoch, global_step,
                    best_metric, history, cfg_dict, extra=None) -> None
    """Atomic: write `path + '.tmp'` then os.replace. Unwrap DataParallel/compile.
       Stores python/numpy/torch RNG states."""
def load_checkpoint(path, *, model=None, optimizer=None, scheduler=None,
                    scaler=None, map_location='cpu', strict=True) -> dict
def find_resume(explicit: str | None = None, work_dir='/kaggle/working/ckpt',
                search_inputs=True) -> str | None
    """Resolution order:
       1. `explicit` if it exists
       2. `<work_dir>/last.pt`
       3. newest `/kaggle/input/*/**/last.pt`   (previous session's saved output)
       4. None"""
```
Checkpoint contents key names are binding: `model`, `optimizer`, `scheduler`, `scaler`,
`epoch`, `global_step`, `best_metric`, `history`, `cfg`, `rng`, `csnet_version`, `wall_h`.

---

## 11. Engine (`csnet/engine.py`)

```python
def build_optimizer(model, cfg) -> (optimizer, scheduler)
def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, *,
                    scheduler=None, grad_clip=5.0, log_every=50, budget=None,
                    global_step=0, amp=True, accum=1, on_step=None) -> dict
    """Returns {'loss','sep','count','acc','steps','stopped_early'}.
       If `budget.expired()` -> break cleanly and set stopped_early=True."""
@torch.no_grad()
def evaluate(model, loader, loss_fn, device, *, amp=True, max_batches=None,
             collect=False) -> dict
    """Returns {'loss','sisdri','count_acc','p_si_snr', 'per_n': {...}}
       and, when collect=True, 'y_true','y_pred' arrays for the confusion matrix."""
```

---

## 12. Config (`csnet/config.py`)

A single flat-ish dataclass tree loaded from YAML with CLI overrides
(`--set train.lr=1e-3 model.n_filters=256`).

```python
@dataclass class DataCfg:  store_root, splits, seg_seconds, n_list, n_weights,
                           gain_db_range, snr_db_range, p_clean, noise_kinds, noise_weights
@dataclass class TrainCfg: batch_size, lr, weight_decay, epochs, steps_per_epoch,
                           num_workers, amp, grad_clip, accum, warmup_steps, sched,
                           time_budget_h, seed, dataparallel, val_every, ckpt_dir,
                           freeze_separator, early_stop_patience
@dataclass class LossCfg:  w_sep, w_sil, w_count, w_noise, silence_db, label_smoothing, clamp_si_sdr
@dataclass class Cfg:      data: DataCfg; model: ModelConfig; train: TrainCfg; loss: LossCfg; name: str

def load_cfg(path: str | None = None, overrides: list[str] | None = None) -> Cfg
def cfg_to_dict(cfg) -> dict
def dict_to_cfg(d) -> Cfg
```

---

## 13. Scripts (`scripts/*.py`) — all argparse CLIs, all importable, all with `main()`

| script | purpose |
|---|---|
| `00_pack_sources.py` | Libri2Mix -> packed store (+ optional real-noise store) |
| `01_make_frozen_sets.py` | write `recipes_dev.csv`, `recipes_test.csv` (+ `--render_wav`) |
| `02_eda.py` | phase-1 EDA figures + `eda_report.json` |
| `03_count_leak_probe.py` | phase-4 leak audit + phase-5 naive baseline |
| `04_train.py` | the training run; resumable; time-budget aware |
| `05_hparam_search.py` | speaker-disjoint K-fold search, resumable via a results JSON |
| `06_evaluate.py` | confusion matrix, P-SI-SNR, per-N SI-SDRi, oracle masks |
| `07_interpret.py` | filterbank, mask statistics vs N, count-head probes, filter ablation |
| `08_infer.py` | any wav in -> `n_speakers` + separated wavs out |

Every long-running script must print, on clean exit or budget expiry, a block:
```
================ RESUME =================
<exact command / notebook variable to set next session>
=========================================
```

---

## 14. Style rules

* Python 3.10+, `from __future__ import annotations` at the top of every module.
* Only these third-party imports are allowed: `numpy, scipy, torch, soundfile,
  pandas, sklearn, matplotlib, tqdm, yaml, pyloudnorm`. **No asteroid, no librosa,
  no speechbrain, no torchaudio.** (Kaggle has all of the allowed ones preinstalled.)
* No global side effects at import time. No `print` in library modules — use `logging`.
* Every public function gets a one-line docstring and type hints.
* Deterministic given a seed. No `random` module in the data path — use
  `np.random.Generator` only.
* Windows-safe paths (`os.path.join`, never a hard-coded `/`).
* Guard every `matplotlib` use with `matplotlib.use("Agg")`.
