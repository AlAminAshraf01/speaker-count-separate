# %% [markdown]
# # 02 - Train the joint counter + separator
#
# **Accelerator: GPU T4 x2** (or P100). **Runtime: up to 11 h per session, resumable.**
#
# This is where the quota goes. Read the resume section before you start it.
#
# ## How a 12-hour cap is survived
#
# Kaggle gives you 12 hours and then takes the machine away, and `/kaggle/working` is wiped
# between sessions unless you **Save Version**. Three mechanisms handle that:
#
# | Failure | Response |
# |---|---|
# | Time budget expires | `TimeBudget` stops **cleanly and exits 0** at 11 h, so Save Version still captures the checkpoint |
# | Hard kill (tab closed, OOM) | mid-epoch checkpoint every 400 steps - you lose minutes |
# | New session | `find_resume` searches `/kaggle/working`, then every attached input dataset, for the newest `last.pt` |
#
# **The resume loop, in full:**
#
# 1. Run this notebook. It stops on its budget and prints a RESUME banner.
# 2. **Save Version -> Save & Run All (Commit).** Wait for it to finish.
# 3. **+ Add Input -> Notebook Output ->** *this* notebook's latest version.
# 4. Re-run. It picks up `last.pt` by itself. No path edits, ever.
#
# Optimizer moments, scheduler position, AMP scaler and RNG streams all resume exactly.
#
# ## Before you run
#
# 1. **Settings -> Accelerator -> GPU T4 x2.**
# 2. **+ Add Input -> Notebook Output ->** `00_build_dataset`.
# 3. From the second session onward, **also** add this notebook's own previous output.

# %include _bootstrap.py

# %%
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_store
import glob

STORE = autodetect_store()
hits = (glob.glob("/kaggle/input/**/recipes_dev.csv", recursive=True)
        + glob.glob(os.path.join(REPO, "data", "recipes_dev.csv")))
RECIPES_DEV = hits[0] if hits else None

print("store      :", STORE)
print("dev recipes:", RECIPES_DEV)
assert STORE and RECIPES_DEV, "attach the 00_build_dataset notebook output via '+ Add Input'"

previous = sorted(glob.glob("/kaggle/input/**/last.pt", recursive=True))
print("\ncheckpoints visible from a previous session:", previous or "none (first session)")

# %% [markdown]
# ## Choose the run
#
# | Config | Params | Speed | Use for |
# |---|---|---|---|
# | `configs/paper.yaml` | 5.3 M | 1x | the headline result |
# | `configs/small.yaml` | 1.9 M | ~5x faster | ablations, and if quota is tight |
# | `configs/tiny.yaml` | 0.3 M | CPU-able | plumbing only, learns nothing |
#
# **Set `EPOCHS` from the measurement, not from ambition.** A timed run on GPU T4 x2 at
# batch 12 gave **1.72 s/step**, so 1000 steps plus validation is about **30 minutes an
# epoch** and an 11-hour session buys roughly **21 epochs**. The free weekly quota is 30
# GPU-hours, so:
#
# | EPOCHS | sessions | GPU-h | leaves for evaluation |
# |---|---|---|---|
# | 40 | 2 | ~22 | 8 h |
# | 60 | 3 | ~31 | nothing -- over the weekly quota |
#
# 40 is the default here for that reason. State it in the report as a budget decision. Our
# numbers will sit a decibel or two below published baselines; that is expected and
# defensible. Half a joint model plus half an interpretability study is not.
#
# ## Memory: it was DataParallel
#
# A session died with `exit -9` -- the Linux OOM killer, which takes the process with no
# traceback. Telemetry showed the container limit is 30 GiB and host memory then climbed
# about 8 MiB **every step**, dead linear, until it hit the ceiling four hours in. 8 MiB is
# one batch: mix 1.15 + refs 5.76 + noise 1.15 MB.
#
# `scripts/10_memory_bisect.py` settled it in one 13-minute run by measuring the memory
# slope of the real training loop with one component switched off at a time:
#
# | configuration | cgroup MiB/step | rss MiB/step | ms/step |
# |---|---|---|---|
# | baseline | 17.34 | 7.97 | 1750 |
# | workers 0 | 8.12 | 12.74 | 1792 |
# | **no dataparallel** | **0.13** | **0.00** | **1140** |
# | workers 0 + no dataparallel | 0.04 | 4.66 | 1177 |
#
# Both flat rows have DataParallel off; both leaking rows have it on. So
# `train.dataparallel` is now **false** in `configs/base.yaml`, and that is a win three
# times over:
#
# * the leak is gone -- 0.1 MiB/step instead of 17;
# * it is **35 % faster** -- 1140 ms/step against 1750. Replicating a 5.3 M-parameter
#   model across two GPUs every forward and gathering `(B, 6, 24000)` outputs back costs
#   more than the second T4 returns on a model this small;
# * the `no amp` configuration crashed twice with `CUDA error: misaligned address`, in a
#   fresh process each time, and only ever with DataParallel on.
#
# **The second T4 is now idle.** That is the right trade at this model size, and the
# measurement above is the justification to put in the report.
#
# ## Budget, re-measured
#
# At **1045 ms/step**, 1000 steps plus validation is about **19 minutes an epoch**, so an
# 11-hour session buys roughly **37 epochs**. 40 epochs fits in two sessions comfortably.
#
# ## The counting head collapsed, and what was done about it
#
# After 38 epochs, separation was working -- validation SI-SDR went from **-11.35 to
# +0.50 dB** -- and the counting head had not moved at all: cross-entropy pinned at 1.611
# where ln(5) is 1.6094, accuracy at 0.200 where chance is 0.200, unchanged since step 2900.
#
# `scripts/11_inspect_count_head.py` read the answer straight off the checkpoint:
#
# | measurement | value |
# |---|---|
# | fc1 units never active | **128 of 128** |
# | post-ReLU values that are zero | **100 %** |
# | logit variation across samples | **0.0000** |
# | pooled feature variation | 0.4144 |
#
# Every hidden unit was negative for every input, so ReLU zeroed the layer and `fc2` could
# only emit its bias -- which settles on the class marginal, uniform for a balanced set,
# which *is* ln(5). The features feeding it varied perfectly well, so nothing upstream was
# wrong. And a dead ReLU receives no gradient, so it could never have recovered: all 96
# test samples were predicted `N=3`, the largest element of the bias vector.
#
# The head now has **LayerNorm on the pooled statistics, LayerNorm after `fc1`, and GELU**
# instead of ReLU. The LayerNorm after `fc1` subtracts the mean across the hidden
# dimension, so a uniform negative shift -- which is what one large early step produces,
# and the early steps are large because the loss starts near 64 -- is removed rather than
# saturating anything. A test drives the new head into the exact state that killed the old
# one and asserts it still varies and still receives gradient.
#
# Two things also changed so this cannot cost thirteen hours again:
#
# * the trainer **warns** if counting accuracy sits at chance for four epochs, naming the
#   inspector to run;
# * `--reset_count_head` re-initialises the head on resume while keeping the separator,
#   because a collapsed head's weights are a local optimum with no gradient out of them.
#
# ## Recovering from here
#
# The 13.5 hours of separator training are worth keeping; only the head needs redoing.
# Set `RECOVER = True` in the cell below: it resumes from your checkpoint, re-initialises
# the head, freezes the separator, and trains the counting head alone -- Gate 6 of the plan.
# With the separator frozen a step takes **331 ms instead of 1045**, so 12 epochs is about
# **70 minutes**, not eleven hours.
#
# Note `--extra_epochs 12` rather than `--set train.epochs=12`. `train.epochs` is an
# absolute target: resuming at epoch 38 and asking for 12 gives `range(38, 12)`, which is
# empty, so the run trains nothing and exits 0. The trainer now refuses that instead of
# pretending it worked.

# ## Gate 2 - the control that was never run
#
# The full evaluation came back with the counter working and the separator not:
#
# | | ours | floor | published |
# |---|---|---|---|
# | counting accuracy | **44.9 %** | 35.0 % naive, 20 % chance | - |
# | SI-SDRi, N=2..5 | **+1.2 dB** | 0 dB (the mixture) | 14.76 dB at N=2 |
# | IRM/IBM oracle | - | - | 10.6-12.8 dB |
#
# Before changing the architecture, it is worth knowing that it is not broken. A single
# fixed batch of four two-speaker mixtures, 250 steps, nothing else changed:
#
# | objective | SI-SDR reached |
# |---|---|
# | separation term alone | **28.36 dB** |
# | the full production loss | **19.66 dB** |
#
# So the model, the loss and the optimiser are all capable -- no bug. What the A/B also
# shows is that the auxiliary objectives cost about **8.7 dB of progress at matched step
# count**: the silence term is driven to 0.000 within a hundred steps, and separation is
# nine decibels behind for all of them. It is a real tax, and it is not the whole story,
# because 38,800 steps of the real run reached 0.19 dB on its own *training* data.
#
# What is left is the thing the design document said to check first and the project never
# did: **is +1.2 dB the pipeline's ceiling, or the pooled N=1..5 task's?** `GATE2 = True`
# answers it in about 2.5 GPU-hours -- two speakers in every mixture, no counting term, no
# silence term, which is the standard fixed-N setup the 14.76 dB was measured under.
#
# * **8-12 dB** - the pipeline is sound and the five-count task plus a forty-epoch budget
#   is the entire explanation. That is a finding and a defensible one for the report.
# * **still near 1 dB** - something deeper is wrong; run
#   `scripts/13_inspect_separator.py` on the checkpoint next.

# %%
CONFIG = "configs/paper.yaml"
EPOCHS = 40                # measured: ~19 min/epoch, so ~37 epochs per 11 h session
BATCH_SIZE = 12          # drop to 8 if you hit CUDA OOM
TIME_BUDGET_H = 11.0     # stop cleanly before Kaggle's 12 h cap
STEPS_PER_EPOCH = 1000

# The two switches that change what this notebook does. They live here, above everything
# that reads them, because CKPT_DIR depends on RECOVER: the recovery run writes to its own
# directory, and a progress plot pointed at the other one silently shows nothing.
RECOVER = False          # True: keep the separator, re-init the counting head, train it alone
GATE2 = False            # True: the fixed-N=2 control run, ~2.5 h -- see "Gate 2" below
BISECT = False           # True: spend ~15 min finding which component leaks, and train nothing
EXTRA_EPOCHS = 12        # RECOVER only: this many MORE epochs, counted from the checkpoint

CKPT_DIR = ("/kaggle/working/ckpt_gate2" if GATE2 else
            "/kaggle/working/ckpt_count" if RECOVER else
            "/kaggle/working/ckpt")
print("writing to :", CKPT_DIR)

# %% [markdown]
# ## Preflight (30 seconds, no quota)
#
# Every expensive failure this project has had was visible in the first minute: cells
# imported two pushes ago, a resume that picked the five-step dry run, an epoch range that
# was empty, a container already half full of RAM. This cell checks those and **stops the
# notebook** rather than letting an eleven-hour session discover them.
#
# If it says `PREFLIGHT: STOP`, read the `->` line under the failing row. Nothing has been
# spent yet.

# %%
run(f"python scripts/12_preflight.py --for {'recover' if RECOVER else 'train'}"
    f" --config {'configs/gate2.yaml' if GATE2 else CONFIG} --store {STORE}"
    f" --recipes_dev {RECIPES_DEV}"
    f" --ckpt_dir {CKPT_DIR} --time_budget_h {TIME_BUDGET_H}"
    + (f" --extra_epochs {EXTRA_EPOCHS}" if RECOVER else "" if GATE2 else f" --epochs {EPOCHS}")
    + f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}"
    f" --set train.batch_size={BATCH_SIZE} train.steps_per_epoch={STEPS_PER_EPOCH}")

# %% [markdown]
# ## Plumbing check first (30 seconds)
#
# Five steps and one eval batch. Never spend an hour of quota discovering that a path is wrong.

# %%
run(f"python scripts/04_train.py --config {CONFIG}"
    f" --store {STORE} --recipes_dev {RECIPES_DEV}"
    f" --ckpt_dir /kaggle/working/_dryrun --resume none --dry_run")

# %% [markdown]
# ## Train
#
# Before it commits, the script times 20 real steps and prints how many epochs the remaining
# budget actually buys. **That measurement replaces the FLOP table** - depthwise separable
# 1-D convolutions are memory-bound, so a peak-TFLOPS estimate is not worth much.
#
# `--resume auto` means: use `CKPT_DIR/last.pt` if it exists, otherwise the **furthest-along**
# `last.pt` in any attached dataset, otherwise start fresh. Furthest-along, not newest: a
# notebook output holds `_dryrun/last.pt` beside the real one and picking by timestamp is a
# coin flip that, lost, restarts eleven hours of training from step 5.

# %%
if BISECT:
    run(f"python scripts/10_memory_bisect.py --store {STORE}"
        f" --config {CONFIG} --set train.batch_size={BATCH_SIZE}")
elif GATE2:
    # The control the design document asks for and the project skipped. Two speakers in
    # every mixture, no counting term, no silence term: the standard fixed-N setup that
    # the published 14.76 dB was measured under. `neg_loss` is the right metric to track
    # here because with w_count and w_sil at zero the loss *is* negative SI-SDR.
    run(f"python scripts/04_train.py"
        f" --config configs/gate2.yaml"
        f" --store {STORE}"
        f" --recipes_dev {RECIPES_DEV}"
        f" --ckpt_dir {CKPT_DIR}"
        f" --dev_n 2"
        f" --resume auto"
        f" --time_budget_h {TIME_BUDGET_H}"
        f" --best_metric neg_loss"
        f" --set train.batch_size={BATCH_SIZE}"
        f" train.steps_per_epoch={STEPS_PER_EPOCH}")
elif RECOVER:
    # Gate 6: the separator is frozen, so only the counting head learns. Its loss is the
    # only one left, which is also the cleanest test of whether the features can support
    # counting at all.
    run(f"python scripts/04_train.py"
        f" --config {CONFIG}"
        f" --store {STORE}"
        f" --recipes_dev {RECIPES_DEV}"
        f" --ckpt_dir {CKPT_DIR}"
        f" --resume auto --reset_count_head"
        f" --time_budget_h {TIME_BUDGET_H}"
        f" --best_metric count_acc"
        f" --extra_epochs {EXTRA_EPOCHS}"   # more epochs from here, not an absolute target
        f" --set train.batch_size={BATCH_SIZE}"
        f" train.steps_per_epoch={STEPS_PER_EPOCH}"
        f" train.freeze_separator=True"
        f" loss.w_sep=0.0 loss.w_sil=0.0 loss.w_noise=0.0")
else:
    run(f"python scripts/04_train.py"
        f" --config {CONFIG}"
        f" --store {STORE}"
        f" --recipes_dev {RECIPES_DEV}"
        f" --ckpt_dir {CKPT_DIR}"
        f" --resume auto"
        f" --time_budget_h {TIME_BUDGET_H}"
        f" --best_metric p_si_snr"
        f" --set train.epochs={EPOCHS}"
        f" train.batch_size={BATCH_SIZE}"
        f" train.steps_per_epoch={STEPS_PER_EPOCH}")
if BISECT:
    print("BISECT is True -- skipping training. Set it back to False once the table")
    print("above names the component to switch off.")

# %% [markdown]
# ## Progress

# %%
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

log = os.path.join(CKPT_DIR, "train_log.csv")
df = pd.read_csv(log) if os.path.exists(log) else None
if df is not None and df.empty:
    # A session that stopped before finishing an epoch writes a header and no rows.
    # Plotting that raises, which turns a clean stop into a failed notebook.
    print("train_log.csv has no completed epochs yet -- this session stopped early.")
    print("The checkpoint is still saved; the curves appear once an epoch finishes.")
    df = None
if df is not None:
    display(df.tail(12))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_ylabel("loss"); axes[0].legend()
    axes[1].plot(df["epoch"], df["train_sisdr"], label="train")
    axes[1].plot(df["epoch"], df["val_sisdr"], label="val")
    axes[1].set_ylabel("matched SI-SDR (dB)"); axes[1].legend()
    axes[2].plot(df["epoch"], df["val_count_acc"] * 100, label="count accuracy %")
    ax2 = axes[2].twinx(); ax2.grid(False)
    ax2.plot(df["epoch"], df["val_p_si_snr"], color="tab:red", label="P-SI-SNR")
    axes[2].set_ylabel("count accuracy (%)"); ax2.set_ylabel("P-SI-SNR (dB)")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()

    print(f"total wall clock across all sessions: {df['wall_h'].max():.2f} h")
    for column, label, unit, scale in (("val_p_si_snr", "P-SI-SNR", "dB", 1.0),
                                       ("val_count_acc", "count accuracy", "%", 100.0)):
        best = df[column].dropna()
        if best.empty:
            print(f"no validated epoch yet, so there is no best {label} to report")
        else:
            print(f"best val {label}: {scale * best.max():.2f} {unit} "
                  f"at epoch {int(df.loc[best.idxmax(), 'epoch'])}")
    # The bar the counting head has to clear is not chance (20 %) but the naive predictor
    # measured in notebook 01: 41.1 %. Anything between the two is a model that has
    # learned the level cue and nothing else.
    print("naive-predictor floor to beat: 41.1 %   (chance is 20.0 %)")
elif not os.path.exists(log):
    print("no train_log.csv yet")

# %%
for name in ("last.pt", "best.pt"):
    path = os.path.join(CKPT_DIR, name)
    if os.path.exists(path):
        print(f"{name:<8} {os.path.getsize(path) / 1e6:6.1f} MB")

# %% [markdown]
# ## If it stopped on the budget
#
# It printed a `======== RESUME ========` block. That is the normal, healthy ending - not an
# error. Do exactly this:
#
# 1. **Save Version -> Save & Run All (Commit).**
# 2. When it finishes: **+ Add Input -> Notebook Output ->** this notebook, latest version.
# 3. Re-run. It continues from the exact step it stopped at.
#
# Repeat until `EPOCHS` is reached. Each session costs one Save Version, nothing else.
#
# ---
#
# ### Gate 2 - do this before trusting anything at N > 2
#
# Reproduce fixed-N=2 within about 1 dB of the published **14.76 dB** SI-SDRi on the official
# Libri2Mix test set. Notebook `04_evaluate` does it with `--libri2mix_dir`. **If that fails,
# stop and fix it** - every downstream number is uninterpretable until it passes.
#
# ### Gate 6 - the count head alone
#
# If joint training struggles, freeze the separator and train only the count head (~2 h):
#
# ```
# --set train.freeze_separator=True loss.w_sep=0.0 loss.w_sil=0.0 loss.w_noise=0.0
# ```
#
# If that cannot beat ~85 % with the leak mitigated, fall back to fixed-N=3 plus the
# interpretability work and say so in the report.
