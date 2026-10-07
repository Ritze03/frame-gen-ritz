#!/usr/bin/env python3
"""Blind-test session builder: renders contestant clips and stills with gpu/fgtest and writes a static page.

    python3 bench/blind/build_session.py \
        --contestant 'v1: algo=v1' --contestant 'v2: algo=v2' \
        --source footage/battlefield.mp4 --source 'footage/cuts/fps_cs2_1.mp4,start=3' --trials 24

Source spec: PATH[,start=SECONDS][,genre=NAME]. Without start, a seeded pick among the highest-motion
windows. Genre defaults to the <genre>_ prefix of the file name (fps|racing|minecraft|indie|realworld).
>= 90 fps sources: the even frames are the 60 fps base and the original frames join as the hidden "real"
contestant. 50-89 fps: base as is, no real contestant. < 50 fps: skipped.
Output: <out>/<session-id>/ with index.html, session.js (the manifest as a global, for file://),
manifest.json, media/ (clips .mp4, stills .png, neutral random names).
"""
import argparse, base64, datetime, json, os, random, shutil, subprocess, sys
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "8"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GPU = REPO / "gpu"
FGTEST = GPU / "build" / "fgtest"
# Applied before every contestant's own set string, as the user plays: whole-frame fallback off, CrosshairV2 UI protection.
BASE_SET = "globalFallback=1,ui=crosshairv2"
WARM = 2                               # unscored warm-up pairs before the clip start (bench/score's default) ...
# ... or, with UI protection on, uiStillFrames - 1 (7 by default): a fresh Interpolator masks a static pixel only after
# uiStillFrames still frames (see ui_warm). Each contestant's clip runs in ONE fgtest call (warm-up + clip, in time order), so
# the Interpolator's state (v2's field history, the UI counters / hysteresis) is continuous over the whole clip.
FPS_OUT = 120
GENRES = {"fps", "racing", "minecraft", "indie", "realworld"}
# One encode for every clip, the real one included.
ENC = ["-c:v", "libx264", "-preset", "slow", "-crf", "10", "-g", "60", "-keyint_min", "60",
       "-sc_threshold", "0", "-bf", "2", "-pix_fmt", "yuv420p", "-r", str(FPS_OUT),
       "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
       "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
       "-threads", "8", "-movflags", "+faststart", "-an"]


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def probe(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
              capture_output=True, text=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(n) / float(d), float(j["format"]["duration"])


def git_hash():
    h = run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = run(["git", "-C", str(REPO), "status", "--porcelain", "--", "gpu"], capture_output=True, text=True).stdout.strip()
    return h + ("-dirty" if dirty else "")


def build_fgtest():
    if not (GPU / "build" / "build.ninja").exists():
        run(["meson", "setup", "build"], cwd=GPU)
    run(["ninja", "-C", "build", "-j8", "fgtest"], cwd=GPU, env={**os.environ, "CCACHE_DISABLE": "1"})


def decoder(path, start, nframes, w, h, scale=None):
    vf = ["-vf", f"scale={scale}:flags=area,format=gray"] if scale else []
    pix = "gray" if scale else "rgba"
    cmd = ["ffmpeg", "-v", "error", "-threads", "8", "-ss", f"{start:.6f}", "-i", str(path)]
    cmd += ["-frames:v", str(nframes)] if nframes else []
    cmd += vf + ["-f", "rawvideo", "-pix_fmt", pix, "-threads", "8", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def encoder(out, w, h):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                             "-framerate", str(FPS_OUT), "-i", "-", *ENC, str(out)], stdin=subprocess.PIPE)


def png(raw, w, h, out):
    run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-i", "-",
         "-frames:v", "1", "-pix_fmt", "rgb24", str(out)], input=raw)


def motion_start(path, fps, dur, clip_len, rng):
    """Seeded pick among the top high-motion windows (mean abs luma diff at 1/8 res, cut-heavy windows skipped)."""
    sw, sh = 240, 136
    p = decoder(path, 0, 0, 0, 0, scale=f"{sw}:{sh}")
    prev, diffs = None, []
    while True:
        b = p.stdout.read(sw * sh)
        if len(b) < sw * sh: break
        f = np.frombuffer(b, np.uint8).astype(np.int16)
        diffs.append(0.0 if prev is None else float(np.abs(f - prev).mean()))
        prev = f
    p.wait()
    d = np.array(diffs)
    win = int(clip_len * fps)
    if len(d) <= win + 2: return 0.0
    cs = np.concatenate([[0], np.cumsum(d)])
    score = cs[win:] - cs[:-win]
    cut = np.convolve(d > 40, np.ones(win), "valid") > 0      # a scene cut in the window
    score[: len(cut)][cut] = -1
    score[-int(fps):] = -1                                     # keep off the very end
    picks = []
    for i in np.argsort(-score):
        if score[i] < 0 or len(picks) >= 3: break
        if all(abs(i - j) >= win for j in picks): picks.append(int(i))
    return round(rng.choice(picks) / fps, 3) if picks else 0.0


def parse_source(spec):
    parts = spec.split(",")
    src = {"path": Path(parts[0]).expanduser(), "start": None, "genre": None}
    for kv in parts[1:]:
        k, v = kv.split("=", 1)
        src[k.strip()] = float(v) if k.strip() == "start" else v.strip()
    if not src["genre"]:
        g = src["path"].stem.split("_")[0].lower()
        src["genre"] = g if g in GENRES else "unknown"
    return src


def ui_warm(setstrs):
    """Warm-up pairs a fresh Interpolator needs before UI protection acts (0 if no set string turns it on): fgtest observes
    every real frame once, a pixel enters the mask after uiStillFrames (default 8) still frames, and k pairs observe k + 1
    frames, so pair uiStillFrames - 1 (0-based) is the first protected one. Same rule as bench/score/score.py."""
    need = 0
    for st in setstrs:
        kv = dict(x.split("=", 1) for x in st.replace(" ", "").split(",") if "=" in x)   # later keys win, as in fgtest
        if kv.get("ui", "off") != "off":
            need = max(need, int(float(kv.get("uiStillFrames", 8))) - 1)
    return need


def token(rng):
    return "%08x" % rng.getrandbits(32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contestant", action="append", required=True, help="'name: set-string', e.g. 'v1: algo=v1'")
    ap.add_argument("--source", action="append", required=True, help="PATH[,start=S][,genre=G]")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--clip-len", type=float, default=4.0, help="seconds, 3-5")
    ap.add_argument("--stills", type=int, default=3, help="still instants per clip")
    ap.add_argument("--still-frac", type=float, default=0.5, help="share of fresh trials that are stills")
    ap.add_argument("--repeat-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--base-set", default=BASE_SET, help=f"set-string applied before every contestant's own (default {BASE_SET})")
    ap.add_argument("--warm", type=int, default=None,
                    help=f"unscored warm-up pairs before each clip (default {WARM}, or uiStillFrames - 1 = 7 with UI protection on)")
    ap.add_argument("--out", default=str(HERE / "sessions"))
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    if not 3 <= a.clip_len <= 5: sys.exit("--clip-len must be 3-5 s")
    seed = a.seed if a.seed is not None else random.randrange(10**6)
    rng = random.Random(seed)

    contestants = []
    for c in a.contestant:
        name, _, sets = c.partition(":")
        name, sets = name.strip(), sets.strip()
        if not name or name == "real": sys.exit(f"bad contestant name in {c!r}")
        contestants.append({"name": name, "set": sets})
    for c in contestants:
        c["eff"] = ",".join(x for x in (a.base_set, c["set"]) if x)    # the fgtest --set string
    uw = ui_warm(c["eff"] for c in contestants)
    warm_want = max(WARM, uw) if a.warm is None else a.warm
    if warm_want < uw:
        print(f"warning: --warm {warm_want} < {uw}: UI protection is not warm at the clip start")

    build_fgtest()
    fghash = git_hash()
    sid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-s{seed}"
    sess = Path(a.out) / sid
    media = sess / "media"
    media.mkdir(parents=True)
    work = HERE / "work" / sid
    work.mkdir(parents=True)
    print(f"session {sid}, fgtest {fghash}", flush=True)

    # contestant table: index 0.. = fgtest contestants, last = real (only on >= 90 fps sources)
    names = [c["name"] for c in contestants] + ["real"]
    REAL = len(contestants)
    clips = []
    fgcalls = []   # (clip, contestant, warm-up pairs, clip pairs): one fgtest call each
    try:
        for si, spec in enumerate(a.source):
            src = parse_source(spec)
            w, h, fps, dur = probe(src["path"])
            if fps < 50:
                print(f"skip {src['path']} ({fps:.1f} fps < 50)"); continue
            has_real = fps >= 90
            step = 2 if has_real else 1
            if src["start"] is None:
                src["start"] = motion_start(src["path"], fps, dur, a.clip_len, rng)
            nb = int(round(a.clip_len * 60)) + 1                   # base (60 fps) frames
            warm = min(warm_want, int(src["start"] * fps + 1e-6) // step)   # warm-up base frames before the start
            if warm < warm_want:
                print(f"warning: {src['path'].name}: only {warm} warm-up pairs fit before start {src['start']}s (wanted {warm_want})")
            t0 = src["start"] - warm * step / fps
            nsrc = (warm + nb - 1) * step + 1                       # source frames decoded, warm-up included
            cid = f"c{len(clips) + 1}"
            cw = work / cid
            cw.mkdir()
            gen_frames = nb - 1
            # still instants: generated frame indices, spread over the clip, seeded, off the edges
            lo, hi = 6, gen_frames - 6
            seg = (hi - lo) / a.stills            # one per equal segment, so they spread out
            stills = [lo + int(seg * i) + rng.randrange(max(1, int(seg))) for i in range(a.stills)]
            entry = {"id": cid, "source": src["path"].name, "genre": src["genre"], "src_fps": round(fps, 3),
                     "start": src["start"], "len": a.clip_len, "warm": warm, "w": w, "h": h, "has_real": has_real,
                     "contestants": list(range(len(contestants))) + ([REAL] if has_real else []),
                     "clip": {}, "stills": [{"k": k, "t": round(src["start"] + (k + 0.5) * step / fps, 4), "img": {}} for k in stills]}
            print(f"{cid}: {src['path'].name} {fps:.0f} fps, start {src['start']}s, {nb} base frames + {warm} warm-up, "
                  f"real={has_real}", flush=True)

            # one decode pass from the warm-up start: base frames to disk (b0000 = first warm-up frame, b<warm> = clip start),
            # the clip's full-rate stream straight into the real encode
            dec = decoder(src["path"], t0, nsrc, w, h)
            fb = w * h * 4
            real_enc = None
            if has_real:
                rname = token(rng) + ".mp4"
                real_enc = encoder(media / rname, w, h)
                entry["clip"][REAL] = "media/" + rname
            for i in range(nsrc):
                raw = dec.stdout.read(fb)
                if len(raw) < fb: sys.exit(f"{src['path']}: short read at frame {i} (clip runs past the end?)")
                j = i - warm * step                                 # index in the clip, < 0 = warm-up
                if real_enc and j >= 0: real_enc.stdin.write(raw)
                if i % step == 0:
                    (cw / f"b{i // step:04d}.rgba").write_bytes(raw)
                elif j >= 0 and (j // 2) in stills:                 # held-out real odd frame
                    n = token(rng) + ".png"
                    png(raw, w, h, media / n)
                    entry["stills"][stills.index(j // 2)]["img"][REAL] = "media/" + n
            dec.wait()
            if real_enc:
                real_enc.stdin.close(); real_enc.wait()

            for ci, c in enumerate(contestants):
                sets = c["eff"]
                cname = token(rng) + ".mp4"
                enc = encoder(media / cname, w, h)
                entry["clip"][ci] = "media/" + cname
                # ONE fgtest call for the whole clip, warm-up pairs first: one continuous Interpolator, as in game
                args = [str(FGTEST), "--set", sets, str(w), str(h)]
                for k in range(warm + gen_frames):
                    args += [str(cw / f"b{k:04d}.rgba"), str(cw / f"b{k + 1:04d}.rgba"), str(cw / f"g{k:04d}.rgba")]
                run(args, stderr=subprocess.DEVNULL)
                fgcalls.append((cid, c["name"], warm, gen_frames))
                for k in range(warm):
                    (cw / f"g{k:04d}.rgba").unlink()               # unscored warm-up outputs
                for k in range(gen_frames):
                    g = cw / f"g{k + warm:04d}.rgba"
                    enc.stdin.write((cw / f"b{k + warm:04d}.rgba").read_bytes())
                    graw = g.read_bytes()
                    enc.stdin.write(graw)
                    if k in stills:
                        n = token(rng) + ".png"
                        png(graw, w, h, media / n)
                        entry["stills"][stills.index(k)]["img"][ci] = "media/" + n
                    g.unlink()
                enc.stdin.write((cw / f"b{gen_frames + warm:04d}.rgba").read_bytes())
                enc.stdin.close(); enc.wait()
                print(f"  {cid}: contestant {ci + 1}/{len(contestants)} done", flush=True)
            shutil.rmtree(cw)
            clips.append(entry)
    finally:
        if not a.keep_work: shutil.rmtree(work, ignore_errors=True)
        try: (HERE / "work").rmdir()
        except OSError: pass
    if not clips: sys.exit("no usable sources")

    for cid, cn, wu, gp in fgcalls:
        print(f"fgtest call: {cid} {cn}: {wu} warm-up + {gp} clip pairs in one call")
    trials = schedule(clips, a, rng)
    for c in clips:
        tt = [t for t in trials if t["clip"] == c["id"]]
        print(f"{c['id']} ({c['source']}): {sum(t['type'] == 'clip' for t in tt)} clip / {sum(t['type'] == 'still' for t in tt)} still trials"
              f" ({sum('repeat_of' in t for t in tt)} repeats)")
    session = {
        "id": sid, "seed": seed, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "fgtest": fghash, "base_set": a.base_set, "warm": warm_want, "clip_len": a.clip_len, "fps": FPS_OUT,
        "encode": "ffmpeg " + " ".join(ENC),
        "contestant_count": len(names),
        "clips": [{k: v for k, v in c.items()} for c in clips],
        "trials": trials,
        # names stay out of the page until the end screen (base64 only so a glance at session.js does not spoil it)
        "reveal": base64.b64encode(json.dumps(
            [{"name": c["name"], "set": c["set"]} for c in contestants] +
            [{"name": "real", "set": "original 120 fps frame (ground truth)"}]).encode()).decode(),
    }
    (sess / "session.js").write_text("window.SESSION = " + json.dumps(session, separators=(",", ":")) + ";\n")
    shutil.copy(HERE / "index.html", sess / "index.html")
    manifest = dict(session, reveal=[{"name": c["name"], "set": c["set"]} for c in contestants] + [{"name": "real"}])
    (sess / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(trials)} trials -> {sess / 'index.html'}")


def schedule(clips, a, rng):
    """Fresh trials cycle through every (clip, contestant pair) so each contestant meets every other on every
    clip; about repeat-frac of the trials re-show an earlier one with sides swapped. Every clip gets at least one clip
    (video) trial and one still trial when trials >= 2 x clips (repeats are reduced to make room); below that a warning
    is printed and the clip trials are placed first."""
    nc = len(clips)
    nrep = max(1, round(a.trials * a.repeat_frac)) if a.trials >= 6 else 0
    if a.trials >= 2 * nc:
        nrep = min(nrep, a.trials - 2 * nc)
    else:
        print(f"warning: --trials {a.trials} < 2 x {nc} clips: not every clip gets both a clip and a still trial")
    nfresh = a.trials - nrep
    combos = [(c["id"], x, y) for c in clips for i, x in enumerate(c["contestants"]) for y in c["contestants"][i + 1:]]
    if nfresh < len(combos):
        print(f"warning: {nfresh} fresh trials < {len(combos)} clip/pair combinations; not every pairing is shown")
    byid = {c["id"]: c for c in clips}
    still_ctr = {(cid, x, y): rng.randrange(a.stills) for cid, x, y in combos}   # per clip+pair: the same pair sees a new instant
    pool = []

    def draw(cid=None):   # the next combo of the shuffled cycle (of clip cid, if given)
        nonlocal pool
        for _ in range(2):
            for i, cmb in enumerate(pool):
                if cid is None or cmb[0] == cid:
                    return pool.pop(i)
            pool = combos[:]; rng.shuffle(pool)
        raise RuntimeError(f"no contestant pair on clip {cid}")

    # reserved: one clip trial per clip, then one still trial per clip, as far as the fresh trials reach
    reserved = [(c["id"], "clip") for c in clips] + [(c["id"], "still") for c in clips]
    reserved = reserved[:nfresh]
    picks = [(draw(cid), typ) for cid, typ in reserved]
    rest = nfresh - len(picks)
    nstill = max(0, round(nfresh * a.still_frac) - sum(t == "still" for _, t in picks))
    nstill = min(nstill, rest)
    types = ["still"] * nstill + ["clip"] * (rest - nstill)
    rng.shuffle(types)
    picks += [(draw(), typ) for typ in types]
    fresh = []
    for (cid, x, y), typ in picks:
        l, r = (x, y) if rng.random() < 0.5 else (y, x)
        t = {"type": typ, "clip": cid, "L": l, "R": r}
        c = byid[cid]
        if typ == "still":
            s = still_ctr[cid, x, y] % len(c["stills"]); still_ctr[cid, x, y] += 1
            t["still"] = s
        fresh.append(t)
    rng.shuffle(fresh)
    trials = fresh[:]
    # repeats: pick originals from the first two thirds, insert later with >= 2 trials in between, sides swapped
    cand = list(range(max(1, (2 * len(fresh)) // 3)))
    for oi in sorted(rng.sample(cand, min(nrep, len(cand)))):
        o = fresh[oi]
        rep = dict(o, L=o["R"], R=o["L"], repeat_of=oi)
        pos_o = next(i for i, u in enumerate(trials) if u is o)
        at = rng.randint(min(pos_o + 3, len(trials)), len(trials))
        trials.insert(at, rep)
    for n, t in enumerate(trials):
        t["n"] = n + 1
    for t in trials:
        if "repeat_of" in t:
            t["repeat_of"] = next(u["n"] for u in trials if u is fresh[t["repeat_of"]])
    return trials


if __name__ == "__main__":
    main()
