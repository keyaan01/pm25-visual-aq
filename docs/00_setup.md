# Phase 0 — Setup (plain-English walkthrough)

This is the same story as `notebooks/00_setup.ipynb`, written so you can read it
without running anything. It explains **what** we set up, **why**, and **how**.

## The problem this phase solves

Training a neural network needs a **GPU** (a chip built for the massive parallel math
that deep learning does). Your laptop doesn't have a usable one, and the PyTorch
installed on it is the CPU-only build. Doing the training there would take days.

The fix is **Google Colab**: a free website that runs your Python on Google's
computers, *with* a free GPU. We just have to get three things onto that Colab
computer: our **code**, the **libraries**, and the **dataset**.

## The three tools and how they connect

| Tool | Role | Analogy |
|------|------|---------|
| **GitHub** | Stores our code, with history. | A shared folder of our program that never loses a version. |
| **Google Colab** | Runs the code on a GPU. | A powerful rented computer you drive from your browser. |
| **Google Drive** | Permanent storage for the dataset. | Your own hard drive in the cloud. |

Colab is *temporary*: when you close it, its local disk is wiped. So the flow is:
**code lives in GitHub → cloned into Colab each session; dataset lives in Drive →
mounted into Colab each session.** Nothing important lives only inside Colab.

## Why GitHub (and not just uploading files)?

As we build the project, the code changes often. With GitHub, updating Colab is one
command (`git pull`) instead of re-uploading files by hand, and every version is
saved so nothing is ever lost. You'll make a free account once (steps are in the chat
handoff). The repository is public and code-only — no dataset images are committed
(they're CC-BY licensed and large), which is why `.gitignore` excludes `data/`.

## What each step in the notebook does, and why

1. **Turn on the GPU** — Colab gives CPUs by default; we switch to a *T4 GPU*. The
   check cell prints the GPU name so you know it worked.
2. **Clone the code** — copies this repository into Colab at
   `/content/pm25-visual-aq`. Re-running does `git pull` to fetch the latest version.
3. **Install libraries** — `requirements.txt` lists them. PyTorch is already on Colab
   *with GPU support*, so pip skips it (that's deliberate — reinstalling could replace
   it with a slower CPU build).
4. **Mount Drive** — connects your Google Drive so we can save the dataset there.
5. **Load + save the dataset (once)** — downloads PM25Vision (~1 GB) on Google's fast
   network and saves it to Drive with `save_to_disk`. Every later session loads the
   saved copy instantly instead of re-downloading. The code is *idempotent*: it checks
   whether the saved copy exists and skips the download if so.
6. **Confirm** — prints the columns, the row counts (~8,298 train + 2,921 test), the
   label range (AQI 1–530), and shows one street photo with its label.

## If something goes wrong

- **"GPU not available"** → you skipped Runtime → Change runtime type → T4 GPU.
- **`git clone` asks for a password / fails** → check the `REPO_URL` is your repo's
  HTTPS URL ending in `.git`, and that the repo is public.
- **Drive mount pop-up doesn't appear** → re-run the cell; approve access for the same
  Google account you're using for Colab.
- **Download is slow** → it runs on Google's network, not your home internet; a couple
  of minutes is normal. It only happens once.

## What's next

Phase 1 (`01_data_audit`) loads that saved dataset and audits it before we do any
modelling.
