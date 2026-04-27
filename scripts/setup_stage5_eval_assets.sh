#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

echo "Installing local automatic-evaluation dependencies."
python -m pip install -q nltk spacy jsonlines pandas pyarrow

python - <<'PY'
import subprocess
import sys

try:
    import spacy
    spacy.load("en_core_web_lg")
except Exception:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
PY

python -m nltk.downloader -q punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet omw-1.4

mkdir -p playground/data/eval/amber/.hf/data playground/data/eval/amber/images
mkdir -p playground/data/eval/object-halbench/images playground/data/eval/object-halbench/coco2014

if [ ! -d /tmp/AMBER/.git ]; then
  rm -rf /tmp/AMBER
  git clone --depth 1 https://github.com/junyangwang0410/AMBER.git /tmp/AMBER
fi

amber_parquet="playground/data/eval/amber/.hf/data/query_generative-00000-of-00001.parquet"
if [ ! -s "${amber_parquet}" ]; then
  wget -nv -O "${amber_parquet}" \
    https://huggingface.co/datasets/visual-preference/AMBER/resolve/main/data/query_generative-00000-of-00001.parquet
fi

echo "Preparing AMBER generative assets."
python - <<'PY'
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

repo = Path(".").resolve()
amber_root = repo / "playground/data/eval/amber"
source_root = Path("/tmp/AMBER/data")
image_root = amber_root / "images"
image_root.mkdir(parents=True, exist_ok=True)

query_rows = json.loads((source_root / "query/query_generative.json").read_text(encoding="utf-8"))
with (amber_root / "query_generative.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for row in query_rows:
        handle.write(
            json.dumps(
                {
                    "id": int(row["id"]),
                    "image": row["image"],
                    "question": row.get("query") or row.get("question") or "Describe this image.",
                },
                ensure_ascii=False,
            )
        )
        handle.write("\n")

annotations = json.loads((source_root / "annotations.json").read_text(encoding="utf-8"))
with (amber_root / "annotations.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for row in annotations:
        if row.get("type") != "generative":
            continue
        handle.write(
            json.dumps(
                {
                    "id": int(row["id"]),
                    "type": row["type"],
                    "truth": row.get("truth") or [],
                    "hallu": row.get("hallu") or [],
                },
                ensure_ascii=False,
            )
        )
        handle.write("\n")

for name in ("relation.json", "safe_words.txt"):
    shutil.copy2(source_root / name, amber_root / name)

df = pd.read_parquet(amber_root / ".hf/data/query_generative-00000-of-00001.parquet")
for _, row in df.iterrows():
    image = row["image"]
    if isinstance(image, dict):
        image_name = Path(str(image.get("path") or f"AMBER_{int(row['id'])}.jpg")).name
        image_bytes = image.get("bytes")
    else:
        image_name = f"AMBER_{int(row['id'])}.jpg"
        image_bytes = image["bytes"]
    if image_bytes is None:
        raise RuntimeError(f"Missing AMBER image bytes for id={row['id']}")
    (image_root / image_name).write_bytes(image_bytes)

print(f"AMBER questions={len(query_rows)} images={len(list(image_root.glob('*')))}")
PY

if [ ! -d /tmp/RLAIF-V/.git ]; then
  rm -rf /tmp/RLAIF-V
  git clone --depth 1 https://github.com/RLHF-V/RLAIF-V.git /tmp/RLAIF-V
fi

echo "Preparing Object HalBench assets."
cp /tmp/RLAIF-V/eval/data/synonyms_refine.txt playground/data/eval/object-halbench/synonyms_refine.txt

coco_zip="playground/data/eval/object-halbench/coco2014/annotations_trainval2014.zip"
coco_ann_dir="playground/data/eval/object-halbench/coco2014/annotations"
if [ ! -s "${coco_ann_dir}/instances_val2014.json" ] || [ ! -s "${coco_ann_dir}/captions_val2014.json" ]; then
  wget -nv -O "${coco_zip}" http://images.cocodataset.org/annotations/annotations_trainval2014.zip
  unzip -q -o "${coco_zip}" -d playground/data/eval/object-halbench/coco2014
  rm -f "${coco_zip}"
fi

python - <<'PY'
from __future__ import annotations

import base64
import json
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer

repo = Path(".").resolve()
root = repo / "playground/data/eval/object-halbench"
image_root = root / "images"
image_root.mkdir(parents=True, exist_ok=True)
source_jsonl = Path("/tmp/RLAIF-V/eval/data/obj_halbench_300_with_image.jsonl")
synonyms_path = root / "synonyms_refine.txt"
ann_dir = root / "coco2014/annotations"

synonym_rows = [
    [part.strip() for part in line.strip().split(", ") if part.strip()]
    for line in synonyms_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
mscoco_objects: set[str] = set()
inverse: dict[str, str] = {}
for row in synonym_rows:
    canonical = row[0]
    mscoco_objects.update(row)
    for synonym in row:
        inverse[synonym] = canonical

double_words = [word for word in inverse if len(word.split()) >= 2]
double_words += ["home plate", "train track"]
double_word_dict = {word: word for word in double_words}
for animal_word in ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]:
    double_word_dict[f"baby {animal_word}"] = animal_word
    double_word_dict[f"adult {animal_word}"] = animal_word
for vehicle_word in ["jet", "train"]:
    double_word_dict[f"passenger {vehicle_word}"] = vehicle_word
double_word_dict["bow tie"] = "tie"
double_word_dict["toilet seat"] = "toilet"
double_word_dict["wine glas"] = "wine glass"

lemmatizer = WordNetLemmatizer()


def caption_to_objects(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(token) for token in tokens]
    merged: list[str] = []
    idx = 0
    while idx < len(words):
        double = " ".join(words[idx : idx + 2])
        if double in double_word_dict:
            merged.append(double_word_dict[double])
            idx += 2
        else:
            merged.append(words[idx])
            idx += 1
    if "toilet" in merged and "seat" in merged:
        merged = [word for word in merged if word != "seat"]
    return [inverse[word] for word in merged if word in mscoco_objects]


questions: list[dict] = []
with source_jsonl.open("r", encoding="utf-8") as handle:
    for idx, line in enumerate(handle):
        row = json.loads(line)
        image_id = int(row["image_id"])
        image_name = f"{image_id}.jpg"
        (image_root / image_name).write_bytes(base64.b64decode(row["image"]))
        questions.append(
            {
                "id": str(row.get("org_idx", idx)),
                "image_id": image_id,
                "image": image_name,
                "question": row["question"],
            }
        )

image_ids = {int(row["image_id"]) for row in questions}
gt_objects: dict[int, set[str]] = {image_id: set() for image_id in image_ids}

for split in ("train", "val"):
    instances = json.loads((ann_dir / f"instances_{split}2014.json").read_text(encoding="utf-8"))
    cat_to_name = {int(cat["id"]): cat["name"] for cat in instances["categories"]}
    for annotation in instances["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in gt_objects:
            continue
        category = cat_to_name[int(annotation["category_id"])]
        gt_objects[image_id].add(inverse.get(category, category))

for split in ("train", "val"):
    captions = json.loads((ann_dir / f"captions_{split}2014.json").read_text(encoding="utf-8"))
    for annotation in captions["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in gt_objects:
            continue
        gt_objects[image_id].update(caption_to_objects(annotation["caption"]))

with (root / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for row in questions:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")

with (root / "annotations.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for row in questions:
        payload = {
            "id": row["id"],
            "image_id": row["image_id"],
            "gt_objects": sorted(gt_objects[int(row["image_id"])]),
        }
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")

empty = [row["image_id"] for row in questions if not gt_objects[int(row["image_id"])]]
print(f"Object HalBench questions={len(questions)} images={len(list(image_root.glob('*')))} empty_gt={len(empty)}")
PY

echo "Stage 5 automatic evaluation assets are installed."
df -h /workspace || true
