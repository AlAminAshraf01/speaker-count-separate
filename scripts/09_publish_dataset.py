#!/usr/bin/env python3
"""Publish the packed store as a real Kaggle Dataset (instead of a notebook output).

    python scripts/09_publish_dataset.py --dir /kaggle/working/store --slug csnet-store
    python scripts/09_publish_dataset.py --dir /kaggle/working/store --slug csnet-store --update \
        --message "rebuilt with real noise"

Why bother, when "Save Version" already works
---------------------------------------------
Both routes give you something attachable, but they behave differently:

===========================  ==========================================================
Notebook output              A real Dataset
===========================  ==========================================================
zero setup                   needs an API token once
"Save & Run All (Commit)"    created once, never recomputed
re-runs the whole ~10-minute
packing job
mounts at                    mounts at ``/kaggle/input/<your-slug>``
``/kaggle/input/<slug>``     -- a name you chose and can rely on
versioned per notebook run   versioned explicitly, with a message
===========================  ==========================================================

If you are going to attach this store to five other notebooks over several weeks, the
Dataset route is worth the one-time token setup. Both are supported; nothing else in this
repo cares which you use, because ``autodetect_store()`` searches for ``manifest.json``.

Credentials (you do this yourself -- this script never asks for or stores a token)
---------------------------------------------------------------------------------
Get ``kaggle.json`` from https://www.kaggle.com/settings -> API -> "Create New Token".
Then pick ONE of:

1. **Kaggle Secrets** (best inside a Kaggle notebook).
   Add-ons -> Secrets -> add two secrets, ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``, using the
   two values from inside ``kaggle.json``. Attach both to the notebook. This script reads
   them automatically.
2. **Environment variables**: ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.
3. **The file itself** at ``~/.kaggle/kaggle.json`` (chmod 600).

A note on directory handling
----------------------------
The Kaggle CLI **skips subdirectories by default**, and the store has one per split. This
script therefore passes ``--dir-mode zip``, which archives each subdirectory; Kaggle
extracts archives when it builds the dataset. Because that behaviour is Kaggle's and not
ours, the script prints an explicit verification step at the end -- **do it**, rather than
discovering a flat dataset three notebooks later. ``--verify`` re-checks a mounted copy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from _common import banner, resolve

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{4,48}[a-z0-9]$")


# --------------------------------------------------------------------------- credentials

def load_credentials() -> tuple[str | None, str | None, str]:
    """Find Kaggle API credentials. Returns ``(username, key, where_from)``."""
    username, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key, "environment variables"

    try:  # Kaggle Secrets, only importable inside a Kaggle notebook
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        username = secrets.get_secret("KAGGLE_USERNAME")
        key = secrets.get_secret("KAGGLE_KEY")
        if username and key:
            return username, key, "Kaggle Secrets"
    except Exception:
        pass

    path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            return blob.get("username"), blob.get("key"), path
        except Exception:
            pass
    return None, None, "nowhere"


CREDENTIAL_HELP = """
No Kaggle API credentials found. Set them up once:

  1. https://www.kaggle.com/settings -> API -> "Create New Token"  (downloads kaggle.json)
  2. Inside a Kaggle notebook, the tidiest route is Secrets:
       Add-ons -> Secrets -> "Add a new secret"
         label  KAGGLE_USERNAME   value  <the "username" field from kaggle.json>
         label  KAGGLE_KEY        value  <the "key" field from kaggle.json>
       then tick both so they attach to this notebook.
  3. Or, equivalently, set the KAGGLE_USERNAME and KAGGLE_KEY environment variables,
     or place kaggle.json at ~/.kaggle/kaggle.json

This script does not read, print or store the token beyond handing it to the Kaggle CLI.

You do NOT need any of this if you would rather use the Save Version route --
see docs/KAGGLE_RUNBOOK.md section 1.
"""


# --------------------------------------------------------------------------- helpers

def check_store(directory: str) -> dict:
    """Fail early if the directory is not a packed store."""
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"{directory!r} has no manifest.json -- that is not a packed store.\n"
                         "  Run scripts/00_pack_sources.py first.")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def directory_size(directory: str) -> tuple[int, int]:
    """Total bytes and file count under a directory."""
    total, count = 0, 0
    for root, _dirs, names in os.walk(directory):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
                count += 1
            except OSError:
                pass
    return total, count


def write_metadata(directory: str, slug: str, title: str, username: str,
                   licence: str = "CC0-1.0") -> str:
    """Write the dataset-metadata.json the Kaggle CLI expects."""
    path = os.path.join(directory, "dataset-metadata.json")
    payload = {"title": title, "id": f"{username}/{slug}",
               "licenses": [{"name": licence}]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def run_kaggle(args: list[str], env: dict) -> int:
    """Run the Kaggle CLI, streaming its output."""
    for candidate in (["kaggle"], [sys.executable, "-m", "kaggle"]):
        try:
            proc = subprocess.Popen(candidate + args, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
        except FileNotFoundError:
            continue
        for line in proc.stdout:
            print(line, end="", flush=True)
        return proc.wait()
    raise SystemExit("the `kaggle` CLI is not installed.\n"
                     "  On Kaggle it is preinstalled. Locally: pip install kaggle")


def verify_mounted(path: str) -> int:
    """Check that a mounted dataset has the structure the rest of the repo expects."""
    banner("verify a mounted dataset")
    print(f"path: {path}\n")
    ok = True
    manifest = os.path.join(path, "manifest.json")
    if os.path.exists(manifest):
        print("  OK    manifest.json")
    else:
        print("  FAIL  manifest.json is missing -- did Kaggle flatten the upload?")
        ok = False
    for split in ("train-100", "dev", "test"):
        audio = os.path.join(path, split, "audio.i16")
        index = os.path.join(path, split, "index.csv")
        if os.path.exists(audio) and os.path.exists(index):
            print(f"  OK    {split}/  ({os.path.getsize(audio) / 1e9:.2f} GB)")
        else:
            print(f"  FAIL  {split}/audio.i16 or index.csv is missing")
            ok = False

    if ok:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "src"))
            from csnet.pack import SourceStore

            print()
            for split in ("train-100", "dev", "test"):
                print("  ", SourceStore(path, split).summary())
        except Exception as exc:
            print(f"  FAIL  the store did not open: {exc}")
            ok = False

    print("\n" + ("the dataset is usable -- point --store at this path"
                  if ok else
                  "NOT usable. Re-upload with --dir_mode tar, or fall back to the\n"
                  "Save Version route in docs/KAGGLE_RUNBOOK.md section 1."))
    return 0 if ok else 1


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/kaggle/working/store",
                    help="directory to publish (the packed store)")
    ap.add_argument("--slug", default="csnet-store",
                    help="dataset slug: lowercase, digits and hyphens, 6-50 chars")
    ap.add_argument("--title", default=None, help="human-readable title")
    ap.add_argument("--update", action="store_true",
                    help="push a new version of an existing dataset instead of creating one")
    ap.add_argument("--message", default="update", help="version message, with --update")
    ap.add_argument("--dir_mode", default="zip", choices=["zip", "tar", "skip"],
                    help="how the CLI handles subdirectories (default zip; the store has them)")
    ap.add_argument("--public", action="store_true",
                    help="make the dataset public (default: private)")
    ap.add_argument("--licence", default="CC0-1.0")
    ap.add_argument("--include_recipes", default=None,
                    help="also copy this data/ directory (the frozen recipe CSVs) into the upload")
    ap.add_argument("--verify", default=None, metavar="MOUNTED_PATH",
                    help="skip publishing; just check an already-mounted dataset")
    ap.add_argument("--dry_run", action="store_true",
                    help="write dataset-metadata.json and print the command, upload nothing")
    args = ap.parse_args()

    if args.verify:
        return verify_mounted(resolve(args.verify) or args.verify)

    directory = resolve(args.dir) or args.dir
    manifest = check_store(directory)
    slug = args.slug.strip().lower()
    if not SLUG_RE.match(slug):
        raise SystemExit(f"invalid slug {slug!r}: use lowercase letters, digits and hyphens, "
                         "6-50 characters, not starting or ending with a hyphen")

    username, key, source = load_credentials()
    if not username or not key:
        print(CREDENTIAL_HELP)
        return 2

    if args.include_recipes:
        import shutil

        recipes = resolve(args.include_recipes) or args.include_recipes
        target = os.path.join(directory, "data")
        os.makedirs(target, exist_ok=True)
        copied = 0
        for name in os.listdir(recipes):
            if name.startswith("recipes_") and name.endswith(".csv"):
                shutil.copy(os.path.join(recipes, name), os.path.join(target, name))
                copied += 1
        print(f"copied {copied} recipe file(s) into {target}")

    size, count = directory_size(directory)
    title = args.title or f"CSNet packed store ({slug})"

    banner("09 - publish the packed store as a Kaggle Dataset")
    print(f"directory   : {directory}")
    print(f"contents    : {count} files, {size / 1e9:.2f} GB")
    print(f"splits      : {', '.join(manifest.get('splits', {}))}")
    print(f"dataset id  : {username}/{slug}")
    print(f"visibility  : {'PUBLIC' if args.public else 'private'}")
    print(f"credentials : {source}")
    print(f"mode        : {'new version' if args.update else 'create'}")

    if size > 100e9:
        print("\nWARNING: Kaggle's per-dataset limit is around 100 GB. This will fail.")

    metadata_path = write_metadata(directory, slug, title, username, args.licence)
    print(f"\nwrote {metadata_path}")

    if args.update:
        command = ["datasets", "version", "-p", directory, "-m", args.message,
                   "--dir-mode", args.dir_mode]
    else:
        command = ["datasets", "create", "-p", directory, "--dir-mode", args.dir_mode]
        if args.public:
            command.append("--public")

    if args.dry_run:
        print("\ndry run -- would execute:\n  kaggle " + " ".join(command))
        return 0

    env = dict(os.environ, KAGGLE_USERNAME=username, KAGGLE_KEY=key)
    print("\nuploading (this takes several minutes for a few GB) ...\n")
    code = run_kaggle(command, env)
    if code != 0:
        print(f"\nkaggle CLI exited {code}.")
        print("Common causes:")
        print("  * the slug already exists  -> re-run with --update")
        print("  * --update on a dataset that does not exist yet -> drop --update")
        print("  * an invalid token -> regenerate it at kaggle.com/settings -> API")
        print("\nThe Save Version route in docs/KAGGLE_RUNBOOK.md section 1 always works "
              "and needs no token.")
        return code

    banner("now verify it, before you rely on it")
    print(f"1. Open  https://www.kaggle.com/datasets/{username}/{slug}")
    print("2. Wait until the status stops saying it is processing.")
    print("3. In any notebook: + Add Input -> Datasets -> search for "
          f"'{slug}'.")
    print("4. Confirm the structure survived, which Kaggle -- not this script -- decides.")
    print("   Find where it mounted (the layout has changed before), then verify:")
    print(f"     python -c \"from _common import autodetect_store; print(autodetect_store())\"")
    print(f"     python scripts/09_publish_dataset.py --verify <that path>")
    print("\nIf step 4 fails, the subdirectories were flattened. Re-upload with")
    print("  --dir_mode tar   (or just use the Save Version route instead).")
    print(f"\nOnce verified, every later script takes  --store /kaggle/input/{slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
