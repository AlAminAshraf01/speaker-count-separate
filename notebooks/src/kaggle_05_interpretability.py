# %% [markdown]
# # 05 - Interpretability: open the box
#
# **Accelerator: GPU T4** (CPU works too). **Runtime: 15-30 minutes. No training.**
#
# This is the project's actual contribution, and it is **forward passes only** on a checkpoint
# you already have. Protect this time when the schedule slips.
#
# ## The argument, in one sentence
#
# As N grows, the network must partition the **same** 512-filter encoder basis among more
# sources - so if mask overlap rises and sparsity falls with N, the degradation curve has a
# *mechanistic explanation computed from the network's own internals*, instead of being
# asserted.
#
# ## The framing that matters
#
# Not "we also built a speaker counter", but **"the count head is an interpretability probe"**.
# It produces an explicit, readable estimate of N from the *same* shared features whose mask
# geometry we measure. One contribution with two halves beats two half-contributions, for the
# same work.
#
# That buys three questions, all answerable here:
#
# 1. Does the count head **read** the mask geometry? (correlate confidence against overlap)
# 2. Do miscounts have a **signature**? (two sources sharing one slot is a mechanistic failure
#    explanation, not a shrug)
# 3. Which basis functions **carry the count**? (ablate filters, watch accuracy fall)
#
# ## Before you run
#
# **+ Add Input -> Notebook Output ->** `00_build_dataset` and `02_train`.

# %include _bootstrap.py

# %%
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_store
import glob

STORE = autodetect_store()
hits = (glob.glob("/kaggle/input/**/recipes_test.csv", recursive=True)
        + glob.glob(os.path.join(REPO, "data", "recipes_test.csv")))
RECIPES_TEST = hits[0] if hits else None
ckpts = sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True)
               + glob.glob("/kaggle/working/**/best.pt", recursive=True))
CKPT = ckpts[0] if ckpts else None

print("store     :", STORE)
print("checkpoint:", CKPT)
assert CKPT and STORE and RECIPES_TEST, "attach the 00_build_dataset and 02_train outputs"

# %%
run(f"python scripts/07_interpret.py"
    f" --ckpt {CKPT}"
    f" --store {STORE}"
    f" --split test"
    f" --recipes {RECIPES_TEST}"
    f" --out /kaggle/working/interpret"
    f" --batch_size 8"
    f" --max_batches 60"
    f" --ablate_batches 20"
    f" --ablate_steps 0 8 16 32 64 128 256")

# %% [markdown]
# ## Phase 2 deliverable - the learned filterbank
#
# The encoder replaces the STFT with a **learned** 1-D convolutional basis. That is this
# project's "custom spectral transformation". Compare the learned centre-frequency curve
# against mel and linear spacing: if it is neither, say so and show the figure.

# %%
from IPython.display import Image, display

for name in ["filterbank_time.png", "filterbank_fft.png"]:
    display(Image(f"/kaggle/working/interpret/{name}"))

# %% [markdown]
# ## Mask geometry as a function of N
#
# Three statistics, all measured from the network's own masks:
#
# * **sparsity** (Hoyer, Gini) - is each source claiming a small part of the basis?
# * **pairwise overlap** (cosine, IoU) - are two sources claiming the *same* part?
# * **entropy** across slots - how contested is an average time-frequency cell?

# %%
display(Image("/kaggle/working/interpret/mask_stats_vs_n.png"))

# %%
import json
import pandas as pd

report = json.load(open("/kaggle/working/interpret/interpret_report.json"))
df = pd.DataFrame(report["mask_stats_by_n"]).T
df.index.name = "N"
display(df.round(4))
print("\nOverlap and entropy are undefined at N=1 (there is no pair), so NaN there is "
      "correct, not missing data.")

# %% [markdown]
# ## Does the count head read the geometry?

# %%
display(Image("/kaggle/working/interpret/conf_vs_overlap.png"))

# %%
corr = report["confidence_correlation"]
for name, res in corr.items():
    print(f"corr(count confidence, mask {name:<9s}) r = {res['pearson_r']:+.3f}  "
          f"p = {res['pearson_p']:.3g}  n = {res['n']}")

print("\nmiscount signature:")
display(pd.DataFrame(report["miscount_signature"],
                     columns=["statistic", "count correct", "count wrong"]))

# %% [markdown]
# ## Which filters carry the count?
#
# Zero the most-active encoder filters and re-measure. A counting accuracy that collapses
# faster than separation quality would say the count decision leans on a *specific* part of
# the basis - which is the proposal's "extract the filterbank and explain the performance"
# deliverable, but with a scalar to attribute against.

# %%
display(Image("/kaggle/working/interpret/filter_ablation.png"))
display(pd.DataFrame(report["ablation"],
                     columns=["filters zeroed", "% of basis", "count acc %",
                              "P-SI-SNR", "SI-SDRi(cc)"]))

# %% [markdown]
# ## Writing this up
#
# Read your own numbers before choosing the sentence. The honest version depends on what the
# figures actually show:
#
# * **If overlap rises and sparsity falls with N** - state that the degradation curve has a
#   mechanism: the same basis is being divided among more sources, and the masks show it.
# * **If confidence correlates with overlap** - the count head is reading the partition
#   geometry, so you can say *why* the model knows how many speakers there are.
# * **If miscounted utterances have measurably higher overlap** - two sources sharing one
#   output slot is a mechanistic failure explanation.
# * **If ablation hurts counting faster than separation** (or the reverse) - report which, and
#   note that it localises the count decision in the basis.
# * **If a correlation is weak or a trend is flat, say so.** A null result here is still a
#   measurement of the network's internals, which is more than an assertion. Do not
#   over-claim: with a few hundred utterances, r = 0.1 is noise.
#
# ---
#
# This notebook is the part of the project that is *yours* rather than a reproduction. If
# quota runs short, cut epochs from `02_train` - not this.
