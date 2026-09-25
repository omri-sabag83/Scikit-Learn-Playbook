"""Re-download both raw datasets from the UCI Machine Learning Repository.

The CSVs in data/raw/ are already committed, so you only need this script to
rebuild them from source. It checks each download's SHA-256 checksum, extracts
the files, converts the credit dataset from .xls to .csv (with LibreOffice, so
no extra Python package is needed), and raises if row/column counts don't match.

Run from the project root:  python data/get_data.py
"""

import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "raw"

SOURCES = {
    "credit": {
        "url": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "sha256": "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602",
    },
    "bank": {
        "url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        "sha256": "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4",
    },
}

SOFFICE_CANDIDATES = ["soffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice"]


def download(name: str, tmp: Path) -> Path:
    src = SOURCES[name]
    path = tmp / f"{name}.zip"
    urllib.request.urlretrieve(src["url"], path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != src["sha256"]:
        raise ValueError(f"{name}: checksum mismatch ({digest}); the UCI file has changed")
    return path


def find_soffice() -> str:
    for cand in SOFFICE_CANDIDATES:
        if shutil.which(cand) or Path(cand).exists():
            return cand
    raise FileNotFoundError("LibreOffice (soffice) not found; needed to convert the credit .xls to .csv")


def build_credit(tmp: Path) -> None:
    with zipfile.ZipFile(download("credit", tmp)) as z:
        z.extractall(tmp / "credit")
    xls = tmp / "credit" / "default of credit card clients.xls"
    subprocess.run([find_soffice(), "--headless", "--convert-to", "csv", "--outdir", str(tmp / "credit"), str(xls)],
                   check=True, capture_output=True)
    shutil.copy(xls.with_suffix(".csv"), RAW / "credit_default.csv")


def build_bank(tmp: Path) -> None:
    with zipfile.ZipFile(download("bank", tmp)) as z:
        z.extractall(tmp / "bank")
    with zipfile.ZipFile(tmp / "bank" / "bank.zip") as z:
        z.extractall(tmp / "bank" / "b")
    with zipfile.ZipFile(tmp / "bank" / "bank-additional.zip") as z:
        z.extractall(tmp / "bank" / "ba")
    shutil.copy(tmp / "bank" / "b" / "bank-full.csv", RAW / "bank-full.csv")
    shutil.copy(tmp / "bank" / "b" / "bank-names.txt", RAW / "bank-names.txt")
    shutil.copy(tmp / "bank" / "ba" / "bank-additional" / "bank-additional-names.txt", RAW / "bank-additional-names.txt")


def reconcile() -> None:
    credit = pd.read_csv(RAW / "credit_default.csv", header=1)
    bank = pd.read_csv(RAW / "bank-full.csv", sep=";")
    if credit.shape != (30_000, 25):
        raise ValueError(f"credit shape {credit.shape}, expected (30000, 25)")
    if bank.shape != (45_211, 17):
        raise ValueError(f"bank shape {bank.shape}, expected (45211, 17)")
    print(f"credit: {credit.shape[0]:,} rows OK; bank: {bank.shape[0]:,} rows OK")


if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        build_credit(Path(t))
        build_bank(Path(t))
    reconcile()
