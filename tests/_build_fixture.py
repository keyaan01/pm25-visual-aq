"""Build a tiny local fixture that mimics the real PM25Vision on-disk layout.

We pull a few dozen real rows from the Hugging Face datasets-server (fast, no 1 GB
download), decode the base64 JPEGs to raw bytes, and save a DatasetDict with
train/test splits via `save_to_disk` -- exactly the structure that Colab produces
with `load_dataset(...).save_to_disk(...)`. Local tests then run the real code path.

Run once:  python tests/_build_fixture.py
"""
import base64
import io
from pathlib import Path

import requests
from datasets import Dataset, DatasetDict, Features, Value
from PIL import Image

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "DeadCardassian/PM25Vision"
OUT = Path(__file__).resolve().parent / "fixture_ds"

# Columns present in the real dataset (image stored as raw JPEG bytes).
FEATURES = Features(
    {
        "image_id": Value("int64"),
        "station_id": Value("int64"),
        "captured_at": Value("string"),
        "camera_angle": Value("string"),
        "longitude": Value("float64"),
        "latitude": Value("float64"),
        "quality_score": Value("string"),
        "downloaded_at": Value("string"),
        "pm25": Value("float64"),
        "filename": Value("string"),
        "quality": Value("string"),
        "pm25_bin": Value("string"),
        "image": Value("binary"),
    }
)


def fetch(split: str, offsets, length=15):
    """Fetch rows from several offsets so we get varied stations and AQI levels."""
    out = []
    for off in offsets:
        r = requests.get(
            ROWS_URL,
            params=dict(dataset=DATASET, config="default", split=split,
                        offset=off, length=length),
            timeout=60,
        )
        r.raise_for_status()
        for rr in r.json()["rows"]:
            row = rr["row"]
            img_bytes = base64.b64decode(row["image"])
            # verify it decodes as an image; skip if not
            try:
                Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                continue
            out.append(
                {
                    "image_id": int(row["image_id"]),
                    "station_id": int(row["station_id"]),
                    "captured_at": str(row["captured_at"]),
                    "camera_angle": None,
                    "longitude": float(row["longitude"]),
                    "latitude": float(row["latitude"]),
                    "quality_score": None,
                    "downloaded_at": str(row["downloaded_at"]),
                    "pm25": float(row["pm25"]),
                    "filename": str(row["filename"]),
                    "quality": str(row["quality"]),
                    "pm25_bin": str(row["pm25_bin"]),
                    "image": img_bytes,
                }
            )
    return out


def main():
    train_rows = fetch("train", offsets=[0, 1500, 3000, 5000, 7000], length=14)
    test_rows = fetch("test", offsets=[0, 1000, 2000], length=10)
    dd = DatasetDict(
        {
            "train": Dataset.from_list(train_rows, features=FEATURES),
            "test": Dataset.from_list(test_rows, features=FEATURES),
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(OUT))
    print(f"Saved fixture: train={len(train_rows)} test={len(test_rows)} -> {OUT}")


if __name__ == "__main__":
    main()
