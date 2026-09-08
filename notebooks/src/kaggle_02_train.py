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
CKPT_DIR = "/kaggle/working/ckpt"

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
# **Set `EPOCHS` deliberately.** The project record's budget arithmetic says a free weekly
# quota buys roughly 60-130 epochs of the pooled N=1..5 set. Plan **60-100, not 200**, and
# state it in the report as a budget decision. Our numbers will sit a decibel or two below
# published baselines; that is expected and defensible. Half a joint model plus half an
# interpretability study is not.

# %%
CONFIG = "configs/paper.yaml"
EPOCHS = 60
BATCH_SIZE = 12          # drop to 8 if you hit CUDA OOM
TIME_BUDGET_H = 11.0     # stop cleanly before Kaggle's 12 h cap
STEPS_PER_EPOCH = 1000

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
# `--resume auto` means: use `/kaggle/working/ckpt/last.pt` if it exists, otherwise the newest
# `last.pt` in any attached dataset, otherwise start fresh.

# %%
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

# %% [markdown]
# ## Progress

# %%
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

log = os.path.join(CKPT_DIR, "train_log.csv")
if os.path.exists(log):
    df = pd.read_csv(log)
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
    print(f"best val P-SI-SNR: {df['val_p_si_snr'].max():.2f} dB "
          f"at epoch {int(df.loc[df['val_p_si_snr'].idxmax(), 'epoch'])}")
else:
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
