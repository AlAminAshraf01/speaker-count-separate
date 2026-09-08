# data/ — the frozen evaluation protocol

This directory holds `recipes_dev.csv` and `recipes_test.csv`. They are **committed to git on
purpose**, and they are the only generated artefacts in this repo that are.

## Why a CSV and not audio

A recipe names which packed utterances to crop, at which offsets, at what gain, with which
noise at which SNR, and the final normalisation factor. `csnet.mixing.render_recipe()` turns it
back into audio **bit for bit**. So the frozen test set is a few hundred kB of text rather than
several GB of WAV — small enough to version-control, which is what "freeze the test set"
actually requires.

The LibriMix authors' instruction is that test metadata "shouldn't be changed under any
circumstance". This is how we honour it.

## How they get here

`scripts/01_make_frozen_sets.py` writes them, once, from a packed store. On Kaggle that happens
in `kaggle_00_build_dataset.ipynb`. Download them from the notebook output and commit them.

```bash
python scripts/01_make_frozen_sets.py --store /kaggle/working/store --out data \
    --splits dev test --n_list 1 2 3 4 5 --n_per_class 300 --seed 72
```

## Rules

1. **Generate once.** The script refuses to overwrite an existing file without `--force`.
2. **Never regenerate after you have reported a number.** Different recipes mean a different
   test set, and every previously reported result becomes incomparable.
3. **Commit both files.** Without them, nobody — including you, next month — can reproduce your
   evaluation.
4. A recipe only means something together with the store it indexes: `utt_idx` is a row number
   in `<store>/<split>/index.csv`. Rebuilding the store with a different seed, `--limit`, or
   `--babble_frac` invalidates every recipe. That is the main reason to build the store once.

## Columns

| Column | Meaning |
|---|---|
| `mix_id` | stable identifier, e.g. `test_n3_00042` |
| `n_src` | number of speakers (the count label) |
| `utt_idx` | `\|`-joined row indices into the split's `index.csv` |
| `crop_start` | `\|`-joined sample offsets, one per source |
| `gain_db` | `\|`-joined per-source gains, applied after RMS normalisation |
| `noise_kind` | `white` / `pink` / `brown` / `babble` / `real` / `none` |
| `noise_id` | seed that deterministically regenerates that exact noise |
| `snr_db` | speech-mixture-to-noise ratio |
| `scale` | the final RMS normalisation factor (stored, not recomputed, so re-rendering is exact) |
