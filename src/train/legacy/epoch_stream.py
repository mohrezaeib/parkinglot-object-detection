# src/train/legacy/epoch_stream.py
from __future__ import annotations
import re
from typing import Iterator, Optional

# Matches lines like:
# "Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size"
# and also tqdm lines that begin with "      1/9" etc are noisy.
RE_SUMMARY = re.compile(r"^\s*Class\s+Images\s+Instances\b.*mAP50", re.IGNORECASE)
# YOLOv5/v3 prints a line like "      0/9 ..." during training; we track the "x/y" range.
RE_EPOCH_HEADER = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\b")
# Some repos print "Saving model artifact on epoch 1" etc.
RE_SAVED_ARTIFACT = re.compile(r"epoch\s+(\d+)\b", re.IGNORECASE)

def iter_epoch_end_lines(stream) -> Iterator[int]:
    """
    Read training output line-by-line and yield the epoch index (0-based)
    once per epoch when the validation summary line appears.
    """
    current_seen: set[int] = set()
    last_epoch_seen: Optional[int] = None

    for raw in stream:
        line = raw.rstrip("\n")

        # Heuristic 1: "Class  Images  Instances ... mAP50 ..." summary line indicates val finished.
        if RE_SUMMARY.search(line):
            # If we have a last_epoch_seen from headers, use that. Else, increment.
            if last_epoch_seen is None:
                # Fallback: emit next epoch number based on how many we've emitted
                e = 0
                while e in current_seen:
                    e += 1
            else:
                e = last_epoch_seen
            if e not in current_seen:
                current_seen.add(e)
                yield e
            continue

        # Heuristic 2: capture "x/y" epoch header to know which epoch is in progress
        m = RE_EPOCH_HEADER.match(line)
        if m:
            try:
                e_now = int(m.group(1))
                last_epoch_seen = e_now
            except Exception:
                pass
            continue

        # Heuristic 3: explicit messages mentioning epoch index
        m2 = RE_SAVED_ARTIFACT.search(line)
        if m2:
            try:
                e_now = int(m2.group(1))
                last_epoch_seen = e_now
            except Exception:
                pass
            continue
