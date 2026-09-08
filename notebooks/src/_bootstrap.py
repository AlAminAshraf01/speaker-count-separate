# %% [markdown]
# ## Bootstrap (this cell is identical in every notebook)
#
# Three ways to get the code onto the Kaggle machine, tried in order:
#
# 1. **GitHub clone** — set `REPO_URL` below and turn *Internet* ON in the notebook
#    settings panel (Settings → Internet → On). This is the recommended route.
# 2. **Repo-as-dataset** — upload this folder as a Kaggle Dataset called
#    `speaker-count-separate` and attach it. No internet needed. Use this if your
#    account cannot enable internet (phone-verification is required for that).
# 3. **Already there** — if `/kaggle/working/speaker-count-separate` exists it is used
#    as-is, so re-running the notebook is cheap.

# %%
REPO_URL = "https://github.com/AlAminAshraf01/speaker-count-separate.git"
REPO_DIR = "/kaggle/working/speaker-count-separate"
REPO_AS_DATASET = "/kaggle/input/speaker-count-separate"

import os
import shutil
import subprocess
import sys


def bootstrap(repo_url: str = REPO_URL, repo_dir: str = REPO_DIR) -> str:
    """Put the repo at `repo_dir`, put its `src/` on sys.path, and chdir into it."""
    if not os.path.isdir(os.path.join(repo_dir, "src")):
        if os.path.isdir(os.path.join(REPO_AS_DATASET, "src")):
            shutil.copytree(REPO_AS_DATASET, repo_dir, dirs_exist_ok=True)
            print(f"copied repo from the attached dataset {REPO_AS_DATASET}")
        else:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir], check=True)
            print(f"cloned {repo_url}")
    src = os.path.join(repo_dir, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(repo_dir)
    return repo_dir


REPO = bootstrap()

import csnet  # noqa: E402

print("csnet", csnet.__version__, "at", REPO)
print("python", sys.version.split()[0])

import torch  # noqa: E402

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| devices", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory / 1e9:.1f} GB")


# %%
import shlex
import time


def run(cmd: str, check: bool = True) -> int:
    """Run a shell command, streaming its output into the notebook."""
    print("$", cmd, flush=True)
    t0 = time.time()
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    print(f"\n[exit {code} in {time.time() - t0:.1f}s]", flush=True)
    if check and code != 0:
        raise SystemExit(f"command failed with exit code {code}")
    return code
