#!/usr/bin/env python3
"""Pair lists of real frames (blend/frames, raw RGBA8 1920x1080) for maps.py / measure.py.

  mkreal.py heldout --out held.json                  the 23 held-out pairs of blend/probe/pairs.json (true in-between frame known)
  mkreal.py add --out list.json NAME CLIP PREV CURR [--mid N] [--scale PX] [--group G] [--frames DIR]
                                                     appends one pair: frames FRAMES/CLIP/%06d.rgba (or full paths for PREV / CURR / MID);
                                                     --scale = the colour wheel's full-saturation magnitude in px (fixed per pair)
Entries have the schema of mkpairs.py's pairs.json (name, prev, curr, mid, W, H, scale, group, truth = false).
"""
import argparse
import json
import os

FRAMES = "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend/frames"
PROBE = "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend/probe/pairs.json"


def fpath(frames, clip, x):
    return x if os.path.exists(str(x)) else f"{frames}/{clip}/{int(x):06d}.rgba"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("heldout")
    h.add_argument("--out", required=True)
    h.add_argument("--frames", default=FRAMES)
    a = sub.add_parser("add")
    a.add_argument("--out", required=True)
    a.add_argument("--frames", default=FRAMES)
    a.add_argument("name")
    a.add_argument("clip")
    a.add_argument("prev")
    a.add_argument("curr")
    a.add_argument("--mid")
    a.add_argument("--scale", type=float, default=32.0)
    a.add_argument("--group", default="scene")
    o = ap.parse_args()
    if o.cmd == "heldout":
        out = []
        for p in json.load(open(PROBE)):
            if p["split"] != "val":
                continue
            c = p["clip"]
            out.append(dict(name=p["id"], prev=fpath(o.frames, c, p["prev"]), curr=fpath(o.frames, c, p["curr"]), mid=fpath(o.frames, c, p["target"]),
                            W=1920, H=1080, scale=32.0, truth=False, group=p["group"], case=p["case"]))
        json.dump(out, open(o.out, "w"), indent=1)
        print(len(out), "pairs ->", o.out)
    else:
        lst = json.load(open(o.out)) if os.path.exists(o.out) else []
        lst = [e for e in lst if e["name"] != o.name]
        lst.append(dict(name=o.name, prev=fpath(o.frames, o.clip, o.prev), curr=fpath(o.frames, o.clip, o.curr),
                        mid=fpath(o.frames, o.clip, o.mid) if o.mid else None, W=1920, H=1080, scale=o.scale, truth=False, group=o.group))
        json.dump(lst, open(o.out, "w"), indent=1)
        print(len(lst), "pairs in", o.out)


if __name__ == "__main__":
    main()
