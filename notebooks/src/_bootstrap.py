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
# 3. **Already there** — an existing clone is **fast-forwarded to the newest commit**,
#    not reused as-is. A Kaggle session outlives many pushes, and silently running code
#    from an hour ago is the most expensive kind of confusion: the log looks fine and the
#    fix you are testing is not in it. Any local edits inside the clone are discarded.
#
# Whichever route runs, the commit is printed. Every log can then be traced to the exact
# code that produced it.

# %%
REPO_URL = "https://github.com/AlAminAshraf01/speaker-count-separate.git"
REPO_DIR = "/kaggle/working/speaker-count-separate"
REPO_AS_DATASET = "/kaggle/input/speaker-count-separate"

import hashlib
import os
import shutil
import subprocess
import sys


def cells_fingerprint(src_dir: str, name: str) -> str:
    """Short hash of one notebook's percent source plus this shared bootstrap.

    ``tools/build_notebooks.py`` stamps this into every generated ``.ipynb``. The copy
    running on Kaggle recomputes it from the freshly-cloned repo, so a notebook whose
    cells were imported before the last push says so in the first ten seconds instead of
    eleven hours later.

    Line endings are normalised first. The same file is CRLF in a Windows working tree
    and LF in a Linux clone, and a fingerprint that disagrees with itself across
    platforms is worse than no fingerprint at all.
    """
    digest = hashlib.sha256()
    for part in (name, "_bootstrap.py"):
        with open(os.path.join(src_dir, part), "rb") as fh:
            digest.update(fh.read().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def cells_status(repo_dir: str, src_name: str | None, stamp: str | None) -> str:
    """Compare the stamp baked into these cells with the repo they are about to run.

    Never raises. A check that can take down every notebook is a worse bug than the one
    it detects, so anything unreadable degrades to "cannot verify".
    """
    if not src_name or not stamp:
        return "unstamped -- re-import this notebook to enable the staleness check"
    try:
        current = cells_fingerprint(os.path.join(repo_dir, "notebooks", "src"), src_name)
    except Exception as exc:
        return f"cannot verify ({exc})"
    if current == stamp:
        return f"current ({stamp})"
    return "\n".join([
        f"STALE  cells {stamp} but repo has {current}",
        "",
        "  These notebook cells were imported before the newest push, so the fix you",
        "  are about to test is not in them. scripts/ and src/ just updated themselves;",
        "  notebook cells cannot, because Kaggle owns them.",
        "",
        "  Fix: File -> Import Notebook -> upload notebooks/" + src_name[:-3] + ".ipynb",
        "       again, re-attach the inputs, and re-run.",
    ])


def _git(repo_dir: str, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo_dir, *argv],
                          capture_output=True, text=True)


def update_clone(repo_dir: str) -> str:
    """Fast-forward an existing clone to the remote's newest commit.

    Returns a short status for printing; never raises. Losing internet is a reason to
    carry on with the code that is already there, but it is not a reason to be quiet
    about it -- running stale code unknowingly is how a fix gets tested without being
    present.
    """
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        return "not a git clone, left as it is"
    branch = _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "main"
    before = _git(repo_dir, "rev-parse", "--short", "HEAD").stdout.strip()
    fetched = _git(repo_dir, "fetch", "--depth", "1", "origin", branch)
    if fetched.returncode != 0:
        tail = (fetched.stderr or "").strip().splitlines()
        return f"COULD NOT FETCH ({tail[-1] if tail else 'unknown'}) -- code may be stale"
    reset = _git(repo_dir, "reset", "--hard", f"origin/{branch}")
    if reset.returncode != 0:
        tail = (reset.stderr or "").strip().splitlines()
        return f"COULD NOT UPDATE ({tail[-1] if tail else 'unknown'}) -- code may be stale"
    after = _git(repo_dir, "rev-parse", "--short", "HEAD").stdout.strip()
    return "already newest" if before == after else f"updated {before} -> {after}"


def describe_commit(repo_dir: str) -> str:
    """``<short sha> <date> <subject>`` for the checked-out commit, or a plain note."""
    out = _git(repo_dir, "log", "-1", "--format=%h %cs %s").stdout.strip()
    return out or "no git metadata"


def bootstrap(repo_url: str = REPO_URL, repo_dir: str = REPO_DIR) -> str:
    """Put the repo at `repo_dir`, put its `src/` on sys.path, and chdir into it."""
    if not os.path.isdir(os.path.join(repo_dir, "src")):
        if os.path.isdir(os.path.join(REPO_AS_DATASET, "src")):
            shutil.copytree(REPO_AS_DATASET, repo_dir, dirs_exist_ok=True)
            print(f"copied repo from the attached dataset {REPO_AS_DATASET}")
        else:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir], check=True)
            print(f"cloned {repo_url}")
    else:
        print(f"existing clone: {update_clone(repo_dir)}")
    src = os.path.join(repo_dir, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(repo_dir)
    return repo_dir


REPO = bootstrap()

import csnet  # noqa: E402

print("csnet", csnet.__version__, "at", REPO)
print("code ", describe_commit(REPO))
# CELLS_SRC / CELLS_SHA are set by the stamp cell that tools/build_notebooks.py puts at
# the top of every generated notebook. globals().get keeps this working in a notebook
# assembled by hand, where that cell may not exist.
CELLS = cells_status(REPO, globals().get("CELLS_SRC"), globals().get("CELLS_SHA"))
print("cells", CELLS if "\n" not in CELLS else "")
if "\n" in CELLS:
    print(CELLS)
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
