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
from _common import autodetect_ckpt, autodetect_libri2mix, autodetect_store, find_recipes
import glob

STORE = autodetect_store()
LIBRI2MIX = autodetect_libri2mix()
# Not a raw glob: a training notebook's output carries a whole git clone, so
# `glob("/kaggle/input/**/recipes_test.csv")[0]` can return the copy committed to
# the repo instead of the one notebook 00 built -- decided by filesystem walk
# order, which changes with whatever inputs happen to be attached. Two notebooks
# then evaluate one checkpoint against two different frozen sets and disagree.
RECIPES_TEST = find_recipes("recipes_test.csv", STORE)

# Which run to evaluate. Candidates are ranked by recorded global step rather than by
# path, because a training notebook's output holds `_dryrun/best.pt` from the 30-second
# plumbing check next to the real one and `_` sorts before every letter -- picking by name
# evaluates a five-step model and says nothing about it.
#
# Step count is the wrong tie-break for a short control run, though: `ckpt_gate2` stops at
# step 8000 and will always lose to the 50,800-step pooled model. Name it to override.
#   ""            the main pooled N=1..5 model
#   "ckpt_gate2"  the fixed-N=2 control
#   "ckpt_silow"  the pooled rerun at w_sil 0.1 -- also shorter than the reference, so it
#                 also has to be named; "" would quietly evaluate the reference instead
#                 and the two runs would look identical
ONLY = ""

print("checkpoints visible:")
CKPT = autodetect_ckpt(contains=ONLY or None)

print("\nstore       :", STORE)
print("test recipes:", RECIPES_TEST)
print("checkpoint  :", CKPT)
print("official    :", LIBRI2MIX)
assert CKPT, "attach the 02_train notebook output so best.pt is visible"
assert STORE and RECIPES_TEST, "attach the 00_build_dataset notebook output"

# %% [markdown]
# ## Preflight (30 seconds, no quota)
#
# Checks the things that have actually gone wrong before: stale cells, a checkpoint that is
# really the dry run, a missing frozen test set. It **stops the notebook** instead of
# producing a plausible-looking table from the wrong model.

# %%
run(f"python scripts/12_preflight.py --for evaluate"
    f" --store {STORE} --recipes_test {RECIPES_TEST}"
    f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}")

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
       f" --oracle_limit 200"
       # Evaluation runs in fp32. This also runs the counting pass under fp16 autocast
       # and prints both, because notebook 05 and this notebook once reported completely
       # different counting behaviour for one checkpoint and autocast is the only thing
       # left that differed. One extra forward pass over the test set, about 50 seconds.
       f" --compare_precision")
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
        print("BELOW TARGET. Already ruled out, so do not spend quota re-checking them:")
        print("  - the loss       python -m csnet.losses passes all four invariants")
        print("  - the model      one fixed batch reaches 19.7 dB with this exact loss")
        print("                   in 250 steps, and 28.4 dB with the separation term alone")
        print("  - the data       all five speaker-disjointness audits are zero")
        print()
        print("  - the task       gate 2 ran: fixed N=2 reached 7.40 dB val SI-SDR in")
        print("                   8 epochs where pooled N=1..5 reached 0.50 dB in 38.")
        print("                   The pooled task and the auxiliary objectives are the")
        print("                   cost, not the pipeline. Report that as the finding.")
        print()
        print("  Still open, if you have quota to spend:")
        print("  1. the objectives cost 8.7 dB at matched steps in the single-batch")
        print("     ablation, and gate 2 moved the task and the objective together, so")
        print("     neither number is attributable. configs/silow.yaml moves one: pooled")
        print("     N=1..5 at w_sil 0.1 instead of 1.0, same 38 epochs. SILOW = True in")
        print("     notebook 02, two sessions, ~12 GPU-h, then come back here with")
        print("     ONLY = \"ckpt_silow\" and compare this table against the reference.")
        print("  2. is part of the mask head dead, the way the counting head was?")
        print("     python scripts/13_inspect_separator.py --ckpt <this checkpoint>")
else:
    print("gate 2 not run - attach the libri2mix-8khz-min dataset to enable it")

# %% [markdown]
# ## Why the counting head does what it does
#
# One minute, no training. Run it whenever the counting number looks wrong -- and it does:
# in fp32 this head answers "1 speaker" for almost every mixture, and the 44.9 % it scores
# under fp16 autocast is a different function of the same weights. This prints where the
# two arithmetic paths part company: at the pooled statistics, or only at the logits.

# %%
RECIPES_DEV = find_recipes("recipes_dev.csv", STORE)
run(f"python scripts/11_inspect_count_head.py"
    f" --ckpt {CKPT} --store {STORE} --recipes_dev {RECIPES_DEV}"
    f" --batches 16 --batch_size 12 --compare_precision")

# %% [markdown]
# ## Honesty checklist for the write-up
#
# Copy these into the limitations section, as prose:
#
# - **SI-SDRi is not reported for a clean single-speaker mixture.** There `mix == s1`
#   exactly, so the unprocessed input already scores about 124 dB and "improvement" is a
#   measurement of the epsilon in the denominator, not of the model. Those sources are
#   excluded and counted; the line above the per-N table says how many. The first run of
#   this notebook averaged them in and reported **-8.68 dB** where the honest figure over
#   N=2..5 was **+1.2 dB**.
# - **The counting head does not transfer to the official test set at all.** It scores
#   52 % on our own 3 s N=2 mixtures and **0 % on 300 official Libri2Mix N=2 files** --
#   not "worse", never once right. Feeding it overlapping 3 s windows instead of one long
#   block (`csnet.inference.separate_long`) did not change that, so input length is not
#   the reason. What is left is how the mixtures are built: ours divide every source by
#   its own RMS and then jitter by at most +-5 dB, which *is* the count-leak mitigation,
#   while LibriMix uses its own loudness target with a much wider spread and leaves the
#   natural pauses our crops are screened against. Report this as the result it is: the
#   mitigation that makes our counting honest also makes it specific to our mixtures.
#   The `it predicted` line under gate 2 shows which count it gives instead.
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
