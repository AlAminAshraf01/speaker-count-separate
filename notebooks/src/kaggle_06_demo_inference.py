# %% [markdown]
# # 06 - Demo: any audio file in, speaker count + clean tracks out
#
# **Accelerator: GPU T4 or None.** **Runtime: seconds per file.**
#
# This is the product the project describes: give it a recording with an unknown number of
# talkers plus background noise, and it tells you **how many people are speaking** and hands
# back **one clean waveform per speaker** (plus the isolated noise).
#
# ## How long files are handled
#
# The model works on 3-second windows, so a long file is processed with 50 % overlap. The
# catch: a separator has no idea that slot 2 in window 7 is the same person as slot 4 in
# window 8. So each window is **permutation-aligned** against the previous window's overlap
# region (correlate, then solve the assignment) before being cross-faded in. Without that,
# speakers swap tracks every few seconds.
#
# Per-window counts are aggregated by averaging the softmax.
#
# ## Before you run
#
# **+ Add Input -> Notebook Output ->** `02_train` (for `best.pt`), and `00_build_dataset` if
# you want to demo on the frozen test mixtures.
#
# To use **your own recording**: **+ Add Input -> Upload -> New Dataset**, then point
# `INPUT_FILE` at it. Any format and sample rate; it is resampled to 8 kHz internally.

# %include _bootstrap.py

# %%
import glob
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_store

ckpts = sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True)
               + glob.glob("/kaggle/working/**/best.pt", recursive=True))
CKPT = ckpts[0] if ckpts else None
print("checkpoint:", CKPT)
assert CKPT, "attach the 02_train notebook output"

# A rendered test mixture makes a good first demo because you have the ground truth.
candidates = sorted(glob.glob("/kaggle/input/**/samples/**/*_mix.wav", recursive=True))
INPUT_FILE = candidates[len(candidates) // 2] if candidates else None
print("input     :", INPUT_FILE)

# %% [markdown]
# ## Or build a demo mixture right now
#
# Three known speakers plus pink noise at 8 dB SNR, so you can check the answer.

# %%
import numpy as np
from csnet.audio import write_wav
from csnet.mixing import sample_recipe, render_recipe
from csnet.noise import NoiseBank
from csnet.pack import SourceStore

STORE = autodetect_store()
if STORE:
    store = SourceStore(STORE, "test")
    bank = NoiseBank(store)
    rng = np.random.default_rng(2026)
    recipe = sample_recipe(store, bank, 3, rng, p_clean=0.0, snr_db_range=(8.0, 8.0))
    demo = render_recipe(recipe, store, bank)
    INPUT_FILE = "/kaggle/working/demo_mix.wav"
    write_wav(INPUT_FILE, demo["mix"] / (np.abs(demo["mix"]).max() + 1e-9) * 0.9)
    for k, source in enumerate(demo["sources"]):
        write_wav(f"/kaggle/working/demo_truth_s{k + 1}.wav",
                  source / (np.abs(demo["mix"]).max() + 1e-9) * 0.9)
    print(f"built a demo mixture: N = {demo['n_src']} speakers, "
          f"{demo['noise_kind']} noise at {demo['snr_db']:.1f} dB SNR")
    print("ground truth: 3 speakers")

# %%
from IPython.display import Audio, display

print("INPUT (the mixture the model is given):")
display(Audio(INPUT_FILE))

# %% [markdown]
# ## Run it

# %%
run(f"python scripts/08_infer.py"
    f" --ckpt {CKPT}"
    f" --input {INPUT_FILE}"
    f" --out /kaggle/working/separated"
    f" --win 3.0 --hop 1.5"
    f" --agg mean_logit")

# %% [markdown]
# ## Listen to the result

# %%
import json

summary = json.load(open("/kaggle/working/separated/summary.json"))
print(f"DETECTED {summary['n_speakers']} SPEAKERS  "
      f"(confidence {summary['confidence'] * 100:.1f} %)\n")

for item in summary["outputs"]:
    path = os.path.join("/kaggle/working/separated", item["file"])
    print(f"{item['file']}   ({item['energy_db']:+.1f} dB)")
    display(Audio(path))

# %% [markdown]
# ## Compare against the ground truth
#
# Only possible on a mixture you built. On a real recording you have nothing to compare to -
# which is exactly why the frozen test set in notebook `04` is the number that counts.

# %%
truths = sorted(glob.glob("/kaggle/working/demo_truth_s*.wav"))
if truths:
    print("GROUND TRUTH sources:")
    for path in truths:
        print(os.path.basename(path))
        display(Audio(path))

# %% [markdown]
# ## Try your own recording
#
# Set `INPUT_FILE` to an uploaded file and re-run the inference cell. Things worth trying:
#
# * a phone recording of two people talking over each other,
# * a podcast clip with music underneath,
# * a single speaker (does it correctly say **1**?),
# * silence (what does it do? note the answer honestly - the model was never trained on it).
#
# ## The caveat that must go in the report
#
# The count head was trained and validated on **3-second, fully-overlapped** crops. Aggregating
# window votes over a long, sparsely-overlapped recording is a **demonstration, not a validated
# result**. Real conversation is sparse, speakers take turns, and counting there is a
# diarisation problem rather than a single judgement about spectral density.
#
# Say that plainly. A demo that works is worth showing; a demo presented as an evaluation is
# not.
