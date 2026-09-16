# %% [markdown]
# # 01 - EDA and the leakage audit
#
# **Accelerator: None (CPU).** No GPU quota. **Runtime: 10-20 minutes.**
#
# This one notebook produces **two mandated report sections**, after a cheap integrity
# gate:
#
# * **Step 0 - frozen-set check**: do the recipes in the attached store match the ones
#   committed to git? Seconds when they agree; two minutes of re-derivation when they do
#   not, which tells you which copy is the reproducible one.
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

# %%
run(f"python scripts/12_preflight.py --for eda --store {STORE}"
    f" --recipes_dev {os.path.join(DATA, 'recipes_dev.csv')}"
    f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}")

# %% [markdown]
# ## Step 0 - is the frozen set actually frozen?
#
# The evaluation protocol lives in two places: the CSVs inside the `00_build_dataset`
# output you just attached, and the copies committed to git. They are supposed to be the
# same bytes - that is the whole claim behind "a few hundred kB of CSV re-renders the test
# set exactly".
#
# It is a claim, so check it rather than assume it. A silent divergence here means every
# SI-SDR number you report later was measured on a different test set than the one in your
# repo, and nothing downstream would tell you.

# %%
import csv
import hashlib


# Git on Windows with core.autocrlf=true rewrites CRLF to LF when a text file is
# committed. The csv module writes CRLF on every platform, so a committed recipe file
# can hash differently from the one Kaggle generates while being the same data line for
# line. That is a packaging artefact, not a broken protocol, and it is worth telling the
# two apart: .gitattributes fixes the first, nothing fixes the second.
_CRLF, _LF = bytes([13, 10]), bytes([10])


def _sha(path: str, eol_blind: bool = False) -> str:
    """First 16 hex digits of SHA-256; optionally ignoring line terminators."""
    data = open(path, "rb").read()
    return hashlib.sha256(data.replace(_CRLF, _LF) if eol_blind
                          else data).hexdigest()[:16]


RECIPE_NAMES = ("recipes_dev.csv", "recipes_test.csv")
REPO_DATA = os.path.join(REPO, "data")
RECHECK = "/kaggle/working/recheck"

FROZEN_OK = True
COMPARED = 0
EOL_ONLY = False
for name in RECIPE_NAMES:
    attached, repo = os.path.join(DATA, name), os.path.join(REPO_DATA, name)
    if not os.path.exists(repo):
        print(f"{name:<20} no copy committed to the repo yet -- skipping")
        continue
    a, r = _sha(attached), _sha(repo)
    COMPARED += 1
    if a == r:
        status = "MATCH"
    elif _sha(attached, True) == _sha(repo, True):
        status = "EOL ONLY"
        EOL_ONLY = True
    else:
        status = "DIFFER"
        FROZEN_OK = False
    print(f"{name:<20} attached {a}   repo {r}   {status}")

print()
if not COMPARED:
    print("Nothing to compare - no recipes committed to the repo yet.")
    print("Download them from the attached 00 output and commit them; see data/README.md.")
elif FROZEN_OK:
    print("Frozen set agrees with the repo.")
    if EOL_ONLY:
        print()
        print("Line terminators differ, every data line does not. Git rewrote CRLF to")
        print("LF on commit (core.autocrlf on Windows). The protocol is intact - the")
        print("csv reader is blind to this - but the byte hashes will keep disagreeing")
        print("until the repo carries a .gitattributes line:")
        print("    data/recipes_*.csv -text")
        print("then: git add --renormalize data/ && git commit && git push")
else:
    print("MISMATCH. Run the next two cells to find out which copy is reproducible.")
    print("The EDA below is unaffected - it describes the corpus, not one particular")
    print("draw - but resolve this before reporting any number from notebook 04.")

# %% [markdown]
# ### Only if they differed: re-derive the recipes, see which copy comes back
#
# This regenerates the frozen sets from the **attached** store with the current code, into
# a scratch directory. Nothing is overwritten. About two minutes on CPU, and it is skipped
# entirely when the hashes already agree, so `Save & Run All` stays cheap.

# %%
if FROZEN_OK or not COMPARED:
    print("nothing to re-derive.")
else:
    run(f"python scripts/01_make_frozen_sets.py --store {STORE}"
        f" --out {RECHECK} --splits dev test --n_list 1 2 3 4 5"
        f" --n_per_class 300 --seg_seconds 3.0 --p_clean 0.25 --snr_db 0 20"
        f" --seed 72 --force")

# %%
VERDICTS = {
    "recheck==attached": (
        "The pipeline IS deterministic: the same store and the same code reproduce the",
        "attached copy. The committed copy is stale - it predates the seeding fix.",
        "FIX: download data/recipes_*.csv from the attached 00 output, replace the repo",
        "copies, commit, push. Then this cell goes quiet.",
    ),
    "recheck==repo": (
        "The committed copy is what the current code reproduces; the attached one is not.",
        "That points at the 00 run rather than at the recipes.",
        "FIX: re-run notebook 00 from a clean container and commit a new version.",
    ),
    "all differ": (
        "All three differ - something in the chain is still nondeterministic.",
        "Do not train on this. The row counts above localise it: 0 of 1500 means the rng",
        "seed diverged at row one, a high number means the packed store moved.",
    ),
}

if not FROZEN_OK and COMPARED:
    verdict = None
    for name in RECIPE_NAMES:
        paths = {"attached": os.path.join(DATA, name),
                 "repo": os.path.join(REPO_DATA, name),
                 "recheck": os.path.join(RECHECK, name)}
        paths = {k: p for k, p in paths.items() if os.path.exists(p)}
        rows = {k: list(csv.reader(open(p, newline="", encoding="utf-8")))
                for k, p in paths.items()}
        print(f"[{name}]")
        for k, p in paths.items():
            print(f"  {k:<9} {_sha(p)}")
        for k in ("repo", "recheck"):
            if k in rows:
                same = sum(x == y for x, y in zip(rows["attached"], rows[k]))
                print(f"  rows identical to attached: {k:<9} "
                      f"{same} / {len(rows['attached'])}")
        if name == "recipes_test.csv" and "recheck" in paths:
            here = _sha(paths["recheck"])
            if here == _sha(paths["attached"]):
                verdict = "recheck==attached"
            elif "repo" in paths and here == _sha(paths["repo"]):
                verdict = "recheck==repo"
            else:
                verdict = "all differ"
    print("=" * 78)
    for row in VERDICTS.get(verdict, ("re-derivation did not run",)):
        print(row)
    print("=" * 78)


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
        flag = ("" if res["accuracy"] <= res["chance"] * 1.5 else
                "  <-- LEAK" if "ARTEFACT" in name.upper() else
                "  <-- signal (legitimate)" if "ACOUSTIC" in name.upper() else
                "  <-- above chance")
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
