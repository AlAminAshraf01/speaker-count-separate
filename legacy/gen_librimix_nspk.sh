#!/usr/bin/env bash
# ============================================================================
# gen_librimix_nspk.sh - generate Libri{N}Mix, 8 kHz, min mode, sep_clean only
#                        WITHOUT downloading the 50 GB WHAM! noise corpus.
#
# Verified on a 5-mixture smoke test: produces s1..sN + mix_clean at 8 kHz with
# mix == sum(sources) to int16 precision, plus asteroid/SpeechBrain-compatible
# metadata CSVs.
#
# Why the patch is needed
# -----------------------
# `scripts/create_librimix_from_metadata.py` unconditionally (a) reads the WHAM
# noise file for every row and (b) calls write_noise() into a `noise/` folder
# that it does not create when `--types mix_clean` is used. So the stock script
# crashes on the very first utterance in clean-only mode *and* still demands the
# noise corpus. Two small edits fix both.
#
# Disk needed (8 kHz / min / sep_clean, train-100 + dev + test)
# -------------------------------------------------------------
#   LibriSpeech train-clean-100 + dev-clean + test-clean (input, read-only) ~7 GB
#   Libri3Mix output  ~8 GB      Libri4Mix output  ~8 GB      Libri5Mix  ~8 GB
# On Kaggle: attach LibriSpeech as an input dataset, write output to
# /kaggle/working (20 GB), then "Save Version" it as a reusable dataset.
#
# Usage
# -----
#   bash gen_librimix_nspk.sh <librispeech_root> <out_root> <n_src>
#     <librispeech_root>  dir that CONTAINS train-clean-100/, dev-clean/, test-clean/
#     <out_root>          where Libri{N}Mix/ will be created
#     <n_src>             2, 3 (official metadata shipped) or 4, 5, ... (generated)
# ============================================================================
set -euo pipefail

LIBRISPEECH_DIR="${1:?usage: gen_librimix_nspk.sh <librispeech_root> <out_root> <n_src>}"
OUT_ROOT="${2:?}"
N_SRC="${3:?}"

pip install -q soundfile pyloudnorm pandas numpy scipy tqdm

if [ ! -d LibriMix ]; then
  git clone --depth 1 https://github.com/JorisCos/LibriMix.git
fi
cd LibriMix

# ---------------------------------------------------------------- the patch --
python3 - <<'PY'
p = 'scripts/create_librimix_from_metadata.py'
s = open(p).read()

if 'PATCH: skip when no noise is needed' not in s:
    old = """    # Read the noise
    noise_path = os.path.join(wham_dir, row['noise_path'])
    noise, _ = sf.read(noise_path, dtype='float32', stop=max_length)"""
    new = """    # Read the noise  (PATCH: skip when no noise is needed)
    if wham_dir is None or str(wham_dir).lower() == 'none':
        return mixture_id, gain_list, sources_list
    noise_path = os.path.join(wham_dir, row['noise_path'])
    noise, _ = sf.read(noise_path, dtype='float32', stop=max_length)"""
    assert old in s, 'upstream script changed - re-check the patch'
    s = s.replace(old, new)

    old = """    # Write the noise and get its path
    abs_noise_path = write_noise(mix_id, transformed_sources, dir_path,
                                 freq)"""
    new = """    # Write the noise and get its path  (PATCH: only if a noise subdir exists)
    if 'noise' in subdirs:
        abs_noise_path = write_noise(mix_id, transformed_sources, dir_path,
                                     freq)
    else:
        abs_noise_path = None"""
    assert old in s, 'upstream script changed - re-check the patch'
    s = s.replace(old, new)

    s = s.replace("parser.add_argument('--wham_dir', type=str, required=True,",
                  "parser.add_argument('--wham_dir', type=str, default=None,")
    open(p, 'w').write(s)
    print('patched create_librimix_from_metadata.py')
else:
    print('already patched')
PY

# ------------------------------------------------- metadata for this n_src --
MD_DIR="metadata/Libri${N_SRC}Mix"
if [ ! -d "$MD_DIR" ]; then
  echo ">> no official metadata for n_src=${N_SRC}; generating it"
  echo ">> NOTE: create_librimix_metadata.py DOES need WHAM metadata (csv only,"
  echo ">>       not the audio) and it probes source durations from LibriSpeech."
  mkdir -p "$MD_DIR"
  python3 scripts/create_librimix_metadata.py \
      --librispeech_dir    "$LIBRISPEECH_DIR" \
      --librispeech_md_dir metadata/LibriSpeech \
      --wham_dir           metadata/Wham_noise \
      --wham_md_dir        metadata/Wham_noise \
      --metadata_outdir    "$MD_DIR" \
      --n_src "$N_SRC"
fi

# --------------------------------------------------------- generate the wavs --
# Keep only train-clean-100 / dev / test to stay inside a free-tier disk budget.
# Delete the train-360 line from the metadata dir if you do not want 36 GB more.
mkdir -p /tmp/md_used
cp "$MD_DIR"/libri${N_SRC}mix_train-clean-100.csv \
   "$MD_DIR"/libri${N_SRC}mix_dev-clean.csv \
   "$MD_DIR"/libri${N_SRC}mix_test-clean.csv /tmp/md_used/

python3 scripts/create_librimix_from_metadata.py \
    --librispeech_dir "$LIBRISPEECH_DIR" \
    --metadata_dir    /tmp/md_used \
    --librimix_outdir "$OUT_ROOT" \
    --n_src "$N_SRC" \
    --freqs 8k \
    --modes min \
    --types mix_clean

echo
echo "done -> ${OUT_ROOT}/Libri${N_SRC}Mix/wav8k/min/{train-100,dev,test}"
du -sh "${OUT_ROOT}/Libri${N_SRC}Mix" || true
