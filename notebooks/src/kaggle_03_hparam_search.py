# %% [markdown]
# # 03 - Hyperparameter search with speaker-disjoint K-fold
#
# **Accelerator: GPU T4 x2.** **Runtime: 1.5-2.5 h with the defaults. Resumable.**
#
# The mandated **Phase 3** deliverable.
#
# ## Two things to understand before running
#
# **The folds are over speakers, not mixtures.** A random k-fold over mixtures would put the
# same speaker on both sides of the fold, which is not a valid fold for this task.
#
# **What K-fold buys here is a variance estimate.** Train and dev already use disjoint
# LibriSpeech speakers, so a validation speaker was never "seen" in the first place. Three
# disjoint speaker groups give you mean ± standard deviation - which is the difference
# between "config A beats config B" and "config A got a lucky fold". The script prints that
# verdict for you and will tell you when a ranking is **within fold noise**.
#
# **The budget is a decision, not an accident.** Every grid point costs (folds × steps) of GPU
# time. The defaults are sized for the free tier on purpose, and the script refuses a grid
# larger than `search.max_points` without `--allow_big`. State the budget in the report;
# a stated budget is defensible, a silently truncated search is not.
#
# ## Before you run
#
# 1. **Settings -> Accelerator -> GPU T4 x2.**
# 2. **+ Add Input -> Notebook Output ->** `00_build_dataset`.
# 3. From a second session, also add this notebook's own previous output (it resumes).

# %include _bootstrap.py

# %%
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_store
import glob, shutil

STORE = autodetect_store()
OUT = "/kaggle/working/search"
os.makedirs(OUT, exist_ok=True)

# carry results forward from a previous session so the search resumes
for path in glob.glob("/kaggle/input/**/search/search_results.json", recursive=True):
    if not os.path.exists(os.path.join(OUT, "search_results.json")):
        shutil.copy(path, os.path.join(OUT, "search_results.json"))
        for fold in glob.glob(os.path.join(os.path.dirname(path), "recipes_fold*.csv")):
            shutil.copy(fold, OUT)
        print("resuming from", path)

print("store :", STORE)
assert STORE, "attach the 00_build_dataset notebook output"

# %% [markdown]
# ## The search space
#
# Defined in `configs/search.yaml`. The interesting question is not only "which config wins"
# but **whether the optimum moves as the mixture densifies** - the Conv-TasNet paper never
# asks that, and it is a genuine finding either way.

# %%
print(open(os.path.join(REPO, "configs/search.yaml")).read())

# %% [markdown]
# ## Run it
#
# Results are rewritten to `search_results.json` after **every** (config, fold) run, so you can
# kill this notebook whenever you like and lose at most one run.

# %%
run(f"python scripts/05_hparam_search.py"
    f" --config configs/search.yaml"
    f" --store {STORE}"
    f" --out {OUT}"
    f" --val_per_class 60"
    f" --time_budget_h 10.5")

# %% [markdown]
# ## The ranked table for the report

# %%
import json
import pandas as pd
from IPython.display import display

summary_path = os.path.join(OUT, "search_summary.json")
if os.path.exists(summary_path):
    summary = json.load(open(summary_path))
    df = pd.DataFrame(summary["summary"])
    df["p_si_snr"] = df["p_si_snr_mean"].round(2).astype(str) + " ± " + \
        df["p_si_snr_std"].round(2).astype(str)
    df["count %"] = (df["count_acc_mean"] * 100).round(1)
    df["params (M)"] = (df["n_parameters"] / 1e6).round(2)
    display(df[["point", "folds", "p_si_snr", "count %", "params (M)"]])

    best, *rest = summary["summary"]
    print(f"\nbest: {best['point']}")
    if rest:
        gap = best["p_si_snr_mean"] - rest[0]["p_si_snr_mean"]
        sd = max(best["p_si_snr_std"], rest[0]["p_si_snr_std"], 1e-6)
        print(f"gap to runner-up {gap:.2f} dB against a fold sd of {sd:.2f} dB")
        print("-> " + ("a real difference" if gap > 2 * sd
                       else "WITHIN FOLD NOISE - do not claim a winner"))
    print(f"\nbudget: {len(summary['summary'])} configs x {summary['folds']} folds "
          f"x {summary['steps']} steps")

# %%
best_yaml = os.path.join(OUT, "best.yaml")
if os.path.exists(best_yaml):
    print(open(best_yaml).read())

# %% [markdown]
# ## Using the result
#
# Copy `best.yaml` into your repo as `configs/best.yaml`, then in notebook `02_train`:
#
# ```python
# CONFIG = "configs/best.yaml"
# ```
#
# **But read the noise verdict first.** If the gap to the runner-up is inside the fold
# standard deviation, the honest thing is to say the search found no significant difference
# and to keep the cheaper configuration. Write that down - a negative result reported clearly
# is worth more than a winner asserted on one fold.
#
# If this session stopped on its budget: **Save Version -> Save & Run All (Commit)**, then add
# this notebook's own output as an input and re-run. Finished (config, fold) pairs are skipped.
