# %% [markdown]
# # 06 - Demo: any audio file in, clean tracks out
#
# **Accelerator: GPU T4 or None.** **Runtime: seconds per file.**
#
# Give it a recording with several talkers plus background noise, and it hands back **one
# clean waveform per speaker slot** (plus the isolated noise), ranked loudest first.
#
# ## Why there is no "N speakers detected" line
#
# There was one, and it has been removed. In fp32 this checkpoint's counting head answers
# **1** for almost every input, so that line printed the same number whoever was talking -
# a headline that looks like a result and carries no information about the audio.
#
# The counting result is **not** being hidden: notebook `04` measures it properly, against
# the naive floor and with the full confusion matrix, which is where a negative result can
# be read as one. A demo is the wrong place to report it. Pass `--count` to `08_infer.py`
# if you ever want it back - after the head is fixed, that is the flag to flip.
#
# ## How long files are handled
#
# The model works on 3-second windows, so a long file is processed with 50 % overlap. The
# catch: a separator has no idea that slot 2 in window 7 is the same person as slot 4 in
# window 8. So each window is **permutation-aligned** against the previous window's overlap
# region (correlate, then solve the assignment) before being cross-faded in. Without that,
# speakers swap tracks every few seconds.
#
# ## Before you run
#
# **+ Add Input -> Notebook Output ->** both of these:
#
# * `02_train` - for `best.pt`. Attach the **same version you gave notebooks 04 and 05**, so
#   the checkpoint printed below is the pooled model and not the gate-2 control.
# * `00_build_dataset` - for the source store the demo mixture is built from.
#
# `00` is not optional in practice: with no store attached and no upload, there is nothing
# to feed the model and the audio preview below has no file to play.
#
# To use **your own recording**: **+ Add Input -> Upload -> New Dataset**, then point
# `INPUT_FILE` at it. Any format and sample rate; it is resampled to 8 kHz internally.

# %include _bootstrap.py

# %%
import glob
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_ckpt, autodetect_store

# Which run to demo. Empty picks the furthest-along checkpoint, which is the pooled model
# -- the right one here, because the pooled model is the system the report is about: it
# handles N=1..5 in one network. Gate 2 will sound better on a two-speaker clip (7.40 dB
# against 0.50 dB val SI-SDR), but it only ever saw N=2 and is a control, not the system.
# Demo it by name if you want to show what the separator can do when the task is fixed --
# and say which one you are playing.
#   ""            the main pooled N=1..5 model
#   "ckpt_count"  the same thing, pinned
#   "ckpt_gate2"  the fixed-N=2 control -- separates better, N=2 only
#   "ckpt_silow"  the pooled rerun at w_sil 0.1 -- a fair demo subject, though its spare
#                 slots were only weakly penalised, so expect more leakage into them
ONLY = ""

print("checkpoints visible:")
CKPT = autodetect_ckpt(contains=ONLY or None)
print("\ncheckpoint:", CKPT)
assert CKPT, "attach the 02_train notebook output"

# %%
run(f"python scripts/12_preflight.py --for demo"
    f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}")

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
# No --count: the speaker count is measured in notebook 04, not headlined here. See the
# note at the top. --agg only selects how per-window counts are pooled, so it is gone too.
run(f"python scripts/08_infer.py"
    f" --ckpt {CKPT}"
    f" --input {INPUT_FILE}"
    f" --out /kaggle/working/separated"
    f" --win 3.0 --hop 1.5")

# %% [markdown]
# ## Listen to the result
#
# One track per speaker slot, loudest first, then the isolated noise. Read the dB column
# alongside: a slot the model did not use sits well below the ones carrying a voice, so the
# number of tracks you can actually hear a person in **is** the model's answer to "how
# many" - just an answer you read off the separator instead of the counting head.

# %%
import json

summary = json.load(open("/kaggle/working/separated/summary.json"))
print(f"{summary['duration_seconds']:.1f} s input, {summary['n_windows']} windows, "
      f"{len(summary['outputs'])} tracks out\n")

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
# * a single speaker (does one track hold the voice and the rest go quiet?),
# * silence (what does it do? note the answer honestly - the model was never trained on it).
#
# ## The caveat that must go in the report
#
# This notebook is a **demonstration, not a validated result**. The model was trained and
# validated on 3-second, fully-overlapped, 8 kHz crops; a long, sparsely-overlapped recording
# is outside that. The number that counts is the frozen test set in notebook `04`.
#
# Say plainly that the demo reports no speaker count, and why: notebook `04` measures the
# counting head at **20.0 %** in fp32 against a **35.0 %** naive floor, and it answers "1"
# for 1,445 of 1,500 test mixtures. A constant predictor at the top of a demo would look
# like a result and be none, so it was removed from the demo and left in the evaluation.
# That sentence is worth more in a report than a working demo would be - it says you read
# your own numbers.
