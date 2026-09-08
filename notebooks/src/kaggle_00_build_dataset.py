# %% [markdown]
# # 00 - Build the dataset
#
# **Accelerator: None (CPU).** This notebook spends **no GPU quota**. Do not enable a GPU;
# you would burn 30 hours a week of quota on `soundfile`.
#
# **Runtime: 25-45 minutes** for the full corpus.
#
# ## What this does
#
# Libri2Mix ships the *isolated* sources (`s1/`, `s2/`), not just the mixtures. This notebook
# packs every one of them into a single flat `int16` file per split, then writes the frozen
# dev and test sets as **recipe CSVs**.
#
# That buys four things at once:
#
# | | |
# |---|---|
# | **Disk** | ~4 GB, instead of ~32 GB if you rendered Libri3/4/5Mix to WAV |
# | **Speed** | one sequential memmap instead of 100k tiny file opens - the biggest throughput win in the project |
# | **Any N** | mixtures for N = 1..5 are made on the fly, so no corpus is committed to a speaker count |
# | **Noise** | added at mix time at a random SNR, so "speech + noise" costs no extra disk |
#
# ## Before you run
#
# 1. **Settings -> Accelerator -> None**.
# 2. **Settings -> Internet -> On** (only needed if you clone from GitHub; skip if you
#    uploaded the repo as a dataset).
# 3. **+ Add Input -> Datasets -> search `libri2mix-8khz-min`** (by `unconscious`, 9.96 GB).
# 4. Optional, for realistic noise: also add a noise corpus such as
#    `mmoreaux/environmental-sound-classification-50` and set `NOISE_DIR` below.
#
# ## After it finishes
#
# **Save Version -> Save & Run All (Commit).** The output becomes a dataset you attach to
# every later notebook. You only ever run this notebook once.

# %include _bootstrap.py

# %% [markdown]
# ## Locate the input dataset
#
# `autodetect_libri2mix` searches the usual Kaggle paths, so normally you change nothing.

# %%
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_libri2mix

LIBRI2MIX_DIR = autodetect_libri2mix()          # or hard-code the path yourself
STORE = "/kaggle/working/store"
DATA = "/kaggle/working/data"
NOISE_DIR = None                                 # e.g. "/kaggle/input/environmental-sound-classification-50/audio"

print("Libri2Mix :", LIBRI2MIX_DIR)
if LIBRI2MIX_DIR is None:
    raise SystemExit(
        "Not found. Add the dataset 'libri2mix-8khz-min' via '+ Add Input' in the right-hand "
        "panel, then re-run this cell.")
for split in ("train-100", "dev", "test"):
    path = os.path.join(LIBRI2MIX_DIR, split)
    print(f"  {split:<10} {'ok' if os.path.isdir(path) else 'MISSING'}  {path}")

# %% [markdown]
# ## Step 1 - pack the sources  (~25-45 min)
#
# Every utterance is stored as its **highest-energy 8-second window**. A random 3 s crop of a
# LibriSpeech utterance often lands in silence, and a silent target makes SI-SDR undefined.
#
# 20 % of speakers in each split are reserved with `role = babble` and are **never** used as
# separation targets - babble noise built from a speaker the model is also asked to separate
# would be an invisible leak.
#
# This step is restartable: an already-packed split is skipped.

# %%
cmd = (f"python scripts/00_pack_sources.py"
       f" --libri2mix_dir {LIBRI2MIX_DIR}"
       f" --out {STORE}"
       f" --splits train-100 dev test"
       f" --babble_frac 0.2"
       f" --cap_seconds 8.0"
       f" --seed 72")
if NOISE_DIR:
    cmd += f" --noise_dir {NOISE_DIR} --noise_limit 4000"
run(cmd)

# %% [markdown]
# ## Step 2 - freeze the evaluation sets
#
# The LibriMix authors' rule is that a test set "shouldn't be changed under any circumstance".
# Ours is a CSV of recipes plus a seed, small enough to commit to git, and it re-renders
# **bit for bit**. Generate it once. Never regenerate it.
#
# `--render_wav` also dumps a few dozen listenable mixtures so you can check with your own
# ears that a 5-speaker mixture really does contain five people.

# %%
run(f"python scripts/01_make_frozen_sets.py"
    f" --store {STORE}"
    f" --out {DATA}"
    f" --splits dev test"
    f" --n_list 1 2 3 4 5"
    f" --n_per_class 300"
    f" --seg_seconds 3.0"
    f" --p_clean 0.25"
    f" --snr_db 0 20"
    f" --seed 72"
    f" --render_wav /kaggle/working/samples --render_limit 40")

# %% [markdown]
# ## Step 3 - verify before you trust it
#
# Four things must hold, and all four are checked here rather than assumed.

# %%
run("python tools/run_all_tests.py --quiet")

# %%
import numpy as np
from csnet.pack import SourceStore
from csnet.mixing import read_recipes, render_recipe
from csnet.noise import NoiseBank

for split in ("train-100", "dev", "test"):
    print(SourceStore(STORE, split).summary())

store = SourceStore(STORE, "test")
bank = NoiseBank(store)
recipes = read_recipes(os.path.join(DATA, "recipes_test.csv"))
worst = 0.0
for r in recipes[::37]:
    out = render_recipe(r, store, bank)
    worst = max(worst, float(np.abs(out["mix"] - (out["sources"].sum(0) + out["noise"])).max()))
print(f"\nmax |mix - (sum(sources) + noise)| over the test set: {worst:.2e}   "
      f"{'OK' if worst < 1e-5 else 'FAILED'}")

sizes = {s: sum(os.path.getsize(os.path.join(STORE, s, f))
                for f in os.listdir(os.path.join(STORE, s)))
         for s in os.listdir(STORE) if os.path.isdir(os.path.join(STORE, s))}
for name, size in sizes.items():
    print(f"  {name:<10} {size / 1e9:5.2f} GB")
print(f"  {'TOTAL':<10} {sum(sizes.values()) / 1e9:5.2f} GB   (limit is 20 GB)")

# %% [markdown]
# ## Listen to a few mixtures
#
# Worth thirty seconds of your time. If a "3-speaker" mixture sounds like one person, the
# problem is here, not in the model.

# %%
import glob
from IPython.display import Audio, display

for path in sorted(glob.glob("/kaggle/working/samples/test/**/*_mix.wav", recursive=True))[:4]:
    print(os.path.relpath(path, "/kaggle/working/samples"))
    display(Audio(path))

# %% [markdown]
# ## Now publish it - pick ONE of two routes
#
# The store has to become something other notebooks can attach. There are two ways, and
# nothing else in this repo cares which you pick, because `autodetect_store()` just looks
# for `manifest.json` under `/kaggle/input`.
#
# | | **Route 1 - Save Version** | **Route 2 - a real Dataset** |
# |---|---|---|
# | Setup | none | an API token, once |
# | Cost of publishing | **re-runs this whole notebook** (~45 min) | uploads the files (~10 min) |
# | Mounts at | `/kaggle/input/<notebook-slug>/store` | `/kaggle/input/csnet-store` |
# | Attach with | + Add Input -> Notebook Output | + Add Input -> Datasets |
#
# **Route 1 is the safe default.** Route 2 is worth the one-time token if you will attach
# this store to several notebooks over several weeks - it is created once and never
# recomputed, and the path is a name you chose rather than a notebook slug.
#
# ### Route 1 - Save Version
#
# 1. **Save Version -> Save & Run All (Commit)** and wait for the header to turn green.
# 2. The contents of `/kaggle/working` become this notebook's output.
# 3. In notebook `02_train`: **+ Add Input -> Notebook Output ->** this notebook.
#
# Note that the commit **re-executes every cell**, so the packing runs a second time. That
# is normal; it is the price of this route.

# %% [markdown]
# ### Route 2 - publish as a real Kaggle Dataset
#
# One-time credential setup, which **you** do - the notebook never asks for a token:
#
# 1. <https://www.kaggle.com/settings> -> **API** -> **Create New Token** (downloads `kaggle.json`).
# 2. In this notebook: **Add-ons -> Secrets -> Add a new secret**, twice:
#    * label `KAGGLE_USERNAME`, value = the `"username"` field inside `kaggle.json`
#    * label `KAGGLE_KEY`, value = the `"key"` field inside `kaggle.json`
# 3. Tick both so they attach to this notebook.
#
# Then set `PUBLISH = True` and run the cell. Leave it `False` to skip Route 2 entirely.

# %%
PUBLISH = False          # set True after adding the two secrets above
SLUG = "csnet-store"     # lowercase, digits and hyphens; this becomes /kaggle/input/<SLUG>
UPDATE = False           # True to push a NEW VERSION of a dataset that already exists

if PUBLISH:
    cmd = (f"python scripts/09_publish_dataset.py"
           f" --dir {STORE}"
           f" --slug {SLUG}"
           f" --include_recipes {DATA}"
           f" --title 'CSNet packed store (Libri2Mix sources, 8 kHz)'")
    if UPDATE:
        cmd += " --update --message 'rebuild'"
    run(cmd, check=False)
else:
    print("PUBLISH is False - skipping. Use Route 1 (Save Version) instead,")
    print("or set PUBLISH = True after adding the KAGGLE_USERNAME / KAGGLE_KEY secrets.")

# %% [markdown]
# ### If you used Route 2, verify it before relying on it
#
# Whether subdirectories survive an upload is Kaggle's behaviour, not this repo's, so check
# rather than assume. Attach the new dataset (**+ Add Input -> Datasets ->** search `SLUG`),
# then run:
#
# ```
# !python scripts/09_publish_dataset.py --verify /kaggle/input/csnet-store
# ```
#
# It must print `OK` for `manifest.json` and for all three splits. If it does not, re-upload
# with `--dir_mode tar`, or just fall back to Route 1.

# %% [markdown]
# ## Two rules from here on
#
# **Do not rebuild the store between training sessions.** The frozen dev and test sets would
# change, and every number you have already reported would become incomparable.
#
# **Commit `data/recipes_dev.csv` and `data/recipes_test.csv` to your GitHub repo.** They are
# a few hundred kB and they *are* the evaluation protocol - see `data/README.md`. Download
# them from this notebook's output panel on the right.
