# %% [markdown]
# # 04 - Evaluate: the benchmark section
#
# **Accelerator: GPU T4 x2** (P100 or even CPU works, just slower). **Runtime: 10-25 minutes.**
#
# The mandated **Phase 5a** deliverable. Produces `eval_report.md`, which you paste straight
# into the report.
#
# ## Report four numbers, never one
#
# SI-SDR is undefined when the predicted count is wrong, and that cannot be retrofitted after
# training. So, fixed before the first run:
#
# 1. **Counting accuracy and the full confusion matrix.** The classes are ordinal - 3→4 is not
#    the same failure as 3→5, and an average hides that.
# 2. **P-SI-SNR** over the whole test set - the honest end-to-end number, defined even when
#    the count is wrong.
# 3. **SI-SDRi per N on the count-correct subset only** - isolates separation from counting,
#    and is the number comparable to fixed-N literature.
# 4. **The naive-predictor floor**, so the counter's accuracy has a floor, plus the IRM/IBM
#    oracle ceiling and the 0 dB mixture floor.
#
# Always report SI-SDR **improvement**: raw input SI-SDR itself falls with N.
#
# ## Gate 2
#
# `--libri2mix_dir` evaluates the **untouched official Libri2Mix test set** at N=2. That is
# the only literature-comparable number in the project. Target: within about 1 dB of the
# published **14.76 dB**.
#
# ## Before you run
#
# 1. **+ Add Input -> Notebook Output ->** `00_build_dataset`, and the `02_train` notebook.
# 2. **+ Add Input -> Datasets ->** `libri2mix-8khz-min` (for gate 2 only).

# %include _bootstrap.py

# %%
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_store, autodetect_libri2mix
import glob

STORE = autodetect_store()
LIBRI2MIX = autodetect_libri2mix()
hits = (glob.glob("/kaggle/input/**/recipes_test.csv", recursive=True)
        + glob.glob(os.path.join(REPO, "data", "recipes_test.csv")))
RECIPES_TEST = hits[0] if hits else None

ckpts = sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True)
               + glob.glob("/kaggle/working/**/best.pt", recursive=True))
CKPT = ckpts[0] if ckpts else None

print("store       :", STORE)
print("test recipes:", RECIPES_TEST)
print("checkpoint  :", CKPT)
print("official    :", LIBRI2MIX)
assert CKPT, "attach the 02_train notebook output so best.pt is visible"
assert STORE and RECIPES_TEST, "attach the 00_build_dataset notebook output"

# %% [markdown]
# ## Run the benchmark
#
# The IRM/IBM oracle rows are the slow part (an STFT per source per mixture), so they run on a
# 200-mixture subsample by default. Raise `--oracle_limit` if you have time.

# %%
cmd = (f"python scripts/06_evaluate.py"
       f" --ckpt {CKPT}"
       f" --store {STORE}"
       f" --split test"
       f" --recipes {RECIPES_TEST}"
       f" --out /kaggle/working/eval"
       f" --batch_size 12"
       f" --oracle_limit 200")
if LIBRI2MIX:
    cmd += f" --libri2mix_dir {LIBRI2MIX} --libri2mix_limit 300"
run(cmd)

# %%
from IPython.display import Image, Markdown, display

display(Image("/kaggle/working/eval/confusion.png"))

# %% [markdown]
# ## The report section, ready to paste

# %%
display(Markdown(open("/kaggle/working/eval/eval_report.md").read()))

# %% [markdown]
# ## Gate 2 verdict

# %%
import json

report = json.load(open("/kaggle/working/eval/eval_report.json"))
official = report.get("official_libri2mix")
if official:
    gap = official["si_sdri"] - official["reference_asteroid"]
    print(f"official Libri2Mix test, N=2, {official['n_files']} files")
    print(f"  ours      {official['si_sdri']:6.2f} dB")
    print(f"  published {official['reference_asteroid']:6.2f} dB")
    print(f"  gap       {gap:+6.2f} dB")
    print()
    if abs(gap) <= 1.0:
        print("PASS - within 1 dB. Numbers at N > 2 are now interpretable.")
    elif gap > 0:
        print("Above the published baseline. Check the evaluation is not accidentally "
              "oracle-informed before celebrating.")
    else:
        print("BELOW TARGET. Before touching the architecture, check in this order:")
        print("  1. is the assignment actually permuting?  python -m csnet.losses")
        print("  2. are the sources really distinct speakers?  scripts/03_count_leak_probe.py")
        print("  3. did training converge, or did it stop on the time budget?  train_log.csv")
        print("  4. only then: more epochs, or a larger config")
else:
    print("gate 2 not run - attach the libri2mix-8khz-min dataset to enable it")

# %% [markdown]
# ## Honesty checklist for the write-up
#
# Copy these into the limitations section, as prose:
#
# - Only the N=2 row on the official test set is comparable to the literature; our N>2
#   mixtures are our own.
# - `min`-mode mixtures are **fully overlapped**, so counting here is one global judgement
#   about spectral density. A high accuracy on this data **does not** mean speaker counting is
#   solved - real conversation is sparse, and counting there is a diarisation problem.
# - The epoch count was a **budget decision** fixed before training, so our numbers sit a
#   decibel or two below published baselines.
# - Noise is synthetic (white / pink / brown / babble from held-out speakers) unless a real
#   noise corpus was attached. **There is no reverberation anywhere.** 8 kHz, single channel,
#   anechoic.
# - The test set was frozen before training and never regenerated.
#
# ### The crossover worth keeping
#
# On LibriMix, Conv-TasNet **beats** IRM/IBM at 2 speakers but **loses** to them at 3. If your
# table shows that, it is a finding, not an embarrassment - and it is exactly why the ideal-mask
# baselines are mandated.
