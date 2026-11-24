"""Chooses the 80 pairs (56 training, 24 held-out) -> $BLEND/probe/pairs.json (also copied to blend/probe/pairs.json)."""
import json, os, sys
import numpy as np
B = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
man = json.load(open(f"{B}/frames/manifest.json"))["samples"]
valids = set(json.load(open(f"{B}/train/cache/real_val_ids.json")))
def pick(lst, n):
    return [lst[int(round(i))] for i in np.linspace(0, len(lst) - 1, n)] if len(lst) >= n else list(lst)
def sel(clip, case, split, n):
    l = [s for s in man if s["clip"] == clip and s["case"] == case and s["t"] == 0.5 and s["target_kind"] == "real" and s["split"] == split]
    if split == "val":
        l = [s for s in l if s["id"] in valids]
    return pick(l, n)
out = []
def add(group, split, ss):
    for s in ss:
        out.append(dict(group=group, split=split, idx=man.index(s), id=s["id"], clip=s["clip"], case=s["case"], prev=s["prev"], curr=s["curr"], target=s["target"], t=s["t"]))
add("forza", "train", sel("racing_forza6_a", "30to60", "train", 16) + sel("racing_forza6_b", "30to60", "train", 16))
add("cs2", "train", sel("fps_cs2_a", "30to60", "train", 8))
add("minecraft120", "train", sel("minecraft120_02", "60to120", "train", 8))
add("battlefield", "train", sel("battlefield", "60to120", "train", 8))
add("forza", "val", sel("racing_forza6_c", "30to60", "val", 12))
add("cs2", "val", sel("fps_cs2_a", "30to60", "val", 2) + sel("fps_cs2_b", "30to60", "val", 2))
add("minecraft120", "val", sel("minecraft120_02", "60to120", "val", 2) + sel("minecraft120_04", "60to120", "val", 2))
bv = sel("battlefield", "60to120", "val", 4)
add("battlefield", "val", bv)
if len(bv) < 4:   # fewer than 4 validation samples: add a training-split sample >= 2 s (240 frames at 120 fps) from every training pair
    tr = [p["curr"] for p in out if p["clip"] == "battlefield" and p["split"] == "train"] + [p["prev"] for p in out if p["clip"] == "battlefield" and p["split"] == "train"]
    far = [s for s in man if s["clip"] == "battlefield" and s["case"] == "60to120" and s["t"] == 0.5 and s["target_kind"] == "real" and s["split"] == "train"
           and all(abs(s["prev"] - q) >= 240 for q in tr)]
    print("battlefield val found", len(bv), "far candidates", len(far))
    add("battlefield", "val", pick(far, 4 - len(bv)))
json.dump(out, open(f"{B}/probe/pairs.json", "w"), indent=1)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs.json"), "w"), indent=1)
import collections
print(collections.Counter((p["split"], p["group"]) for p in out))
