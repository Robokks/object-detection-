"""Opens the training notebook straight from GitHub in Google Colab — no
manual upload needed, Colab loads it directly from the repo.

Run this (PyCharm's ▶ button, or `python scripts/open_colab.py`) and your
default browser opens to the notebook, already pointed at this repo and
branch. You still do the rest by hand in the Colab page that opens: set the
runtime to a T4 GPU (Runtime -> Change runtime type), then Runtime -> Run
all, and upload your photos/dataset zip when the upload cell asks — this
script only saves the "go find and re-upload the .ipynb file every time"
step.
"""

import webbrowser

REPO = "Robokks/object-detection-"
BRANCH = "claude/vision-model-object-detection-jbwh8s"  # update this if the notebook moves to another branch
NOTEBOOK_PATH = "colab/train_on_colab.ipynb"

URL = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{NOTEBOOK_PATH}"

if __name__ == "__main__":
    print(f"Opening {URL}")
    webbrowser.open(URL)
