# %% [markdown]
# # 01 - EDA and the leakage audit
#
# **Accelerator: None (CPU).** No GPU quota. **Runtime: 10-20 minutes.**
#
# This one notebook produces **two mandated report sections**:
#
# * **Phase 1 - EDA**: correlation matrices, variance, outliers.
# * **Phase 4 - data-leakage audit**, whose output doubles as the **Phase 5 naive-predictor
#   benchmark**.
#
# ## The finding you are here to reproduce
#
# LibriMix normalises every source to an independent loudness target drawn from
# U(-33, -25) LUFS and then **sums** them, so mixture level climbs about 10·log₁₀(N) dB. In
# `min` mode the mixture is also truncated to the shortest of the N sources, so duration
# falls as N rises. Either scalar alone predicts the speaker count far above chance -
# with no speech modelling in it at all.
#
# | Features given to a depth-3 tree | Accuracy (4 classes, chance 25 %) |
# |---|---|
# | Mixture level (LUFS) only | **53 %** |
# | `min`-mode duration only | 33 % |
# | Both, after a 3 s crop + RMS normalisation | **25 % (= chance)** |
#
# If your counting head scores 95 % and this script scores 60 %, most of that accuracy is
# bookkeeping, not acoustics. Run this **before** you train a counter.
#
# ## Before you run
#
# 1. **Settings -> Accelerator -> None.**
# 2. **+ Add Input -> Notebook Output ->** the `00_build_dataset` notebook.

# %include _bootstrap.py

# %%
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_store

STORE = autodetect_store()
# Recursive globs, because Kaggle mounts inputs at /kaggle/input/<slug> on some
# accounts and /kaggle/input/datasets/<owner>/<slug> on others.
import glob
hits = (glob.glob("/kaggle/input/**/recipes_test.csv", recursive=True)
        + glob.glob("/kaggle/working/**/recipes_test.csv", recursive=True)
        + glob.glob(os.path.join(REPO, "data", "recipes_test.csv")))
DATA = os.path.dirname(hits[0]) if hits else None

print("store   :", STORE)
print("recipes :", DATA)
assert STORE, "attach the 00_build_dataset notebook output via '+ Add Input'"
assert os.path.exists(os.path.join(DATA, "recipes_test.csv")), "recipes_test.csv not found"

# %% [markdown]
# ## Phase 1 - EDA
#
# Six figures. The one that matters most is `leak_before_after.png`: mixture level and
# duration, before and after the mitigation, side by side. That figure is the justification
# for the entire data design.

# %%
run(f"python scripts/02_eda.py"
    f" --store {STORE}"
    f" --split test"
    f" --recipes {DATA}/recipes_test.csv"
    f" --out /kaggle/working/eda"
    f" --per_class 200")

# %%
from IPython.display import Image, display

for name in ["leak_before_after.png", "corr_matrix_per_n.png", "corr_hist.png",
             "input_sisdr_per_n.png", "density_cues_vs_n.png", "outlier_pairs.png"]:
    path = f"/kaggle/working/eda/{name}"
    if os.path.exists(path):
        print(name)
        display(Image(path))

# %% [markdown]
# ## Phase 4 - the leakage audit, and Phase 5's naive baseline
#
# Four audits run here. Three are structural and must come out at exactly zero; the fourth
# is the measured one.
#
# | # | Hazard | Expected |
# |---|---|---|
# | 1 | Same speaker twice **inside** a mixture | 0 |
# | 2 | A speaker crossing **train/dev/test** | 0 |
# | 3 | Babble noise reusing a **target** speaker | 0 |
# | 4 | The count label recoverable from the mixing recipe | measured below |

# %%
run(f"python scripts/03_count_leak_probe.py"
    f" --store {STORE}"
    f" --split test"
    f" --recipes {DATA}/recipes_test.csv"
    f" --per_class 400"
    f" --crop 3.0"
    f" --max_depth 3"
    f" --folds 5"
    f" --out /kaggle/working/leak")

# %% [markdown]
# ## The numbers to paste into the report

# %%
import json

leak = json.load(open("/kaggle/working/leak/leak_report.json"))
chance = leak["chance"] * 100

print(f"{'probe':<48s} {'accuracy':>9s}")
print("-" * 60)
for stage, label in (("raw", "RAW"), ("mitigated", "MITIGATED")):
    for name, res in leak[stage].items():
        flag = "  <-- LEAK" if res["accuracy"] > res["chance"] * 1.5 else ""
        print(f"{label + ' | ' + name:<48s} {res['accuracy'] * 100:8.1f} %{flag}")
print("-" * 60)
print(f"{'chance':<48s} {chance:8.1f} %")
print(f"\nNAIVE BASELINE the counting head must beat: "
      f"{leak['naive_baseline_accuracy'] * 100:.1f} %")

# %% [markdown]
# ### What to write
#
# > The count label was recoverable from a single scalar with no acoustics in it at roughly
# > twice chance. The mitigation - a fixed 3-second crop and per-utterance RMS normalisation -
# > removes both cues **by construction**, and it is applied identically to training,
# > validation and test. We report the mitigated tree's accuracy as the floor the counting
# > head must beat.
#
# Also record the hazard you **avoided**: LibriCount, the obvious off-the-shelf speaker-
# counting dataset, is built from LibriSpeech *test-clean* - the same 40 speakers as our test
# split. Pre-training or validating a counter on it would put the same speakers on both sides
# of the line, so it was not used.
#
# ---
#
# Nothing here touched the GPU. Your weekly quota is intact for `02_train`.
