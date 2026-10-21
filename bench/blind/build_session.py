#!/usr/bin/env python3
"""Blind-test session builder: renders contestant clips and stills with gpu/fgtest and writes a static page.

    python3 bench/blind/build_session.py \
        --contestant 'v1: algo=v1' --contestant 'v2: algo=v2' \
        --source footage/battlefield.mp4 --source 'footage/cuts/fps_cs2_a.mp4,windows=2' --trials 24

Source spec: PATH[,windows=N][,start=SECONDS][,genre=NAME]. The "arena": every source gives N clip windows
(--windows-per-source, default 4), non-overlapping and spread over the file (one per equal segment, seeded), each one
passing the motion check (see pick_windows). start=S instead pins ONE window at S. Genre defaults to the <genre>_
prefix of the file name (fps|racing|minecraft|indie|realworld).
>= 90 fps sources: the even frames are the 60 fps base and the original frames join as the hidden "real"
contestant. 50-89 fps: base as is, no real contestant. < 50 fps: skipped.
Output: <out>/<session-id>/ with index.html, session.js (the manifest as a global, for file://),
manifest.json, media/ (clips .mp4, stills .png, neutral random names).
"""
import argparse, base64, datetime, json, math, os, random, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GPU = REPO / "gpu"
FGTEST = GPU / "build" / "fgtest"
# Applied before every contestant's own set string, as the user plays: whole-frame fallback off, CrosshairV2 UI protection.
BASE_SET = "globalFallback=1,ui=crosshairv2"
WARM = 2                               # unscored warm-up pairs before the clip start (bench/score's default) ...
# ... or, with UI protection on, uiStillFrames - 1 (7 by default): a fresh Interpolator masks a static pixel only after
# uiStillFrames still frames (see ui_warm). Each contestant's window runs in ONE fgtest call (warm-up + clip, in time
# order), so the Interpolator's state (v2's field history, the UI counters / hysteresis) is continuous over the clip.
FPS_OUT = 120
GENRES = {"fps", "racing", "minecraft", "indie", "realworld"}
# Motion check (pick_windows), on 240x136 grey frames. Calibrated on footage/ (2026-10): the median 4 s window scores
# about 9.5 on battlefield, 5.2 on obs1, 8 on obs2, 6-8.5 on the 60 fps cs2 / forza / minecraft cuts.
SCAN_W, SCAN_H = 240, 136
MIN_MOTION = 4.0       # window mean abs diff between consecutive BASE (60 fps) frames, i.e. source frames `step` apart
CUT_DIFF = 40.0        # a scene cut: one consecutive SOURCE frame pair with mean abs diff above this
STATIC_DIFF, STATIC_SEC = 1.0, 0.5   # a static stretch (menu, pause, frozen capture): any 0.5 s mean below this
# One encode for every clip, the real one included.
ENC = ["-c:v", "libx264", "-preset", "slow", "-crf", "10", "-g", "60", "-keyint_min", "60",
       "-sc_threshold", "0", "-bf", "2", "-pix_fmt", "yuv420p", "-r", str(FPS_OUT),
       "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
       "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
       "-movflags", "+faststart", "-an"]
# Size / time estimate per encoded clip second at 1920x1080 (scaled by pixel count), measured on this machine
# with no thread caps (2026-10, battlefield + cs2 / forza cuts): x264 output, PNG still, wall time incl. fgtest and decode
# (an idle machine; the end of the build prints the measured rate).
EST_MB_PER_CLIP_S = 10.0
EST_MB_PER_STILL = 3.5
EST_S_PER_CLIP_S = 10.0


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
    run(["ninja", "-C", "build", "fgtest"], cwd=GPU, env={**os.environ, "CCACHE_DISABLE": "1"})


def decoder(path, start, nframes, w, h, scale=None):
    vf = ["-vf", f"scale={scale}:flags=area,format=gray"] if scale else []
    pix = "gray" if scale else "rgba"
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-i", str(path)]
    cmd += ["-frames:v", str(nframes)] if nframes else []
    cmd += vf + ["-f", "rawvideo", "-pix_fmt", pix, "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def encoder(out, w, h):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                             "-framerate", str(FPS_OUT), "-i", "-", *ENC, str(out)], stdin=subprocess.PIPE)


def png(raw, w, h, out):
    run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-i", "-",
         "-frames:v", "1", "-pix_fmt", "rgb24", str(out)], input=raw)


def motion_scan(path, step):
    """One low-res pass over the whole source (240x136 grey, area scaling). Per source frame i:
    d1[i] = mean |f_i - f_(i-1)| (the cut check) and ds[i] = mean |f_i - f_(i-step)| (the motion score: the spacing of the
    60 fps base frames the contestants interpolate between)."""
    p = decoder(path, 0, 0, 0, 0, scale=f"{SCAN_W}:{SCAN_H}")
    hist, d1, ds = [], [], []
    while True:
        b = p.stdout.read(SCAN_W * SCAN_H)
        if len(b) < SCAN_W * SCAN_H: break
        f = np.frombuffer(b, np.uint8).astype(np.int16)
        d1.append(float(np.abs(f - hist[-1]).mean()) if hist else 0.0)
        ds.append(float(np.abs(f - hist[-step]).mean()) if len(hist) >= step else 0.0)
        hist = (hist + [f])[-step:]
    p.wait()
    return np.array(d1), np.array(ds)


def window_table(d1, ds, L, step, fps):
    """For every window start s (L source frames from s): motion = mean ds over the window's base pairs, cut = any
    d1 > CUT_DIFF inside, static = any STATIC_SEC stretch inside whose mean ds < STATIC_DIFF."""
    N = len(d1)
    S = N - L + 1
    if S <= 0: return None
    s = np.arange(S)
    csd = np.concatenate([[0], np.cumsum(ds)])
    motion = (csd[s + L] - csd[s + step]) / (L - step)
    cc = np.concatenate([[0], np.cumsum(d1 > CUT_DIFF)])
    cut = (cc[s + L] - cc[s + 1]) > 0
    h = max(2, int(fps * STATIC_SEC))
    mm = (csd[h:] - csd[:-h]) / h                    # mm[j] = mean ds[j:j+h]
    sc = np.concatenate([[0], np.cumsum(mm < STATIC_DIFF)])
    a, b = np.minimum(s + step, len(mm)), np.minimum(s + L - h + 1, len(mm))
    static = (sc[np.maximum(a, b)] - sc[a]) > 0
    return motion, cut, static


def pick_windows(tab, L, nwin, lo, hi, rng, min_motion):
    """N non-overlapping window starts in [lo, hi], spread out: the span is cut into N equal segments and each segment
    gets a seeded pick among the top quarter (by motion) of its windows that pass the check (motion >= min_motion, no
    cut, no static stretch). A segment without a passing window is filled from passing windows anywhere else, then
    from cut-free windows by motion (flagged motion_ok=False, warned)."""
    motion, cut, static = tab
    ok = (motion >= min_motion) & ~cut & ~static
    span = hi + L - lo
    if span < L * nwin:
        n2 = max(1, span // L)
        print(f"  warning: only {n2} non-overlapping {L}-frame windows fit (wanted {nwin})")
        nwin = n2
    seg = span / nwin
    picks = []

    def choose(cand):
        cand = sorted(cand, key=lambda s: -motion[s])
        return int(rng.choice(cand[:max(1, len(cand) // 4)]))

    def free(s):
        return all(abs(s - p) >= L for p in picks)

    for k in range(nwin):
        a, b = math.ceil(lo + k * seg), math.floor(lo + (k + 1) * seg) - L
        cand = [s for s in range(a, b + 1) if ok[s] and free(s)]
        if cand: picks.append(choose(cand))
    missing = nwin - len(picks)
    if missing:
        for relax in (False, True):
            while len(picks) < nwin:
                cand = [s for s in range(lo, hi + 1) if (ok[s] if not relax else not cut[s]) and free(s)]
                if not cand: break
                picks.append(choose(cand))
        print(f"  warning: {missing} segment(s) without a passing window; filled elsewhere "
              f"({sum(not ok[p] for p in picks)} below the motion check)")
    return [(p, float(motion[p]), bool(ok[p])) for p in sorted(picks)]


def parse_source(spec):
    parts = spec.split(",")
    src = {"path": Path(parts[0]).expanduser(), "start": None, "genre": None, "windows": None}
    for kv in parts[1:]:
        k, v = kv.split("=", 1)
        k = k.strip()
        if k not in src: sys.exit(f"unknown source key {k!r} in {spec!r}")
        src[k] = float(v) if k == "start" else int(v) if k == "windows" else v.strip()
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


def du(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contestant", action="append", required=True, help="'name: set-string', e.g. 'v1: algo=v1'")
    ap.add_argument("--source", action="append", required=True, help="PATH[,windows=N][,start=S][,genre=G]")
    ap.add_argument("--windows-per-source", type=int, default=4, help="clip windows per source (default 4; windows=N per source)")
    ap.add_argument("--min-motion", type=float, default=MIN_MOTION,
                    help=f"motion check: window mean abs base-frame diff at {SCAN_W}x{SCAN_H} grey (default {MIN_MOTION})")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--clip-len", type=float, default=4.0, help="seconds, 3-5")
    ap.add_argument("--stills", type=int, default=3, help="still instants per window")
    ap.add_argument("--still-frac", type=float, default=0.5, help="share of fresh trials that are stills")
    ap.add_argument("--repeat-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--base-set", default=BASE_SET, help=f"set-string applied before every contestant's own (default {BASE_SET})")
    ap.add_argument("--warm", type=int, default=None,
                    help=f"unscored warm-up pairs before each window (default {WARM}, or uiStillFrames - 1 = 7 with UI protection on)")
    ap.add_argument("--out", default=str(HERE / "sessions"))
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    if not 3 <= a.clip_len <= 5: sys.exit("--clip-len must be 3-5 s")
    if a.windows_per_source < 1: sys.exit("--windows-per-source must be >= 1")
    seed = a.seed if a.seed is not None else random.randrange(10**6)
    rng = random.Random(seed)
    t_start = time.time()

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
    print(f"session {sid}, fgtest {fghash}", flush=True)

    # contestant table: index 0.. = fgtest contestants, last = real (only on >= 90 fps sources)
    names = [c["name"] for c in contestants] + ["real"]
    REAL = len(contestants)
    nb = int(round(a.clip_len * 60)) + 1                               # base (60 fps) frames per window
    gen_frames = nb - 1

    # ---- phase 1: scan every source, choose its windows ----
    plan = []    # (source dict, [(start frame, motion, ok)])
    for spec in a.source:
        src = parse_source(spec)
        w, h, fps, dur = probe(src["path"])
        if fps < 50:
            print(f"skip {src['path']} ({fps:.1f} fps < 50)"); continue
        step = 2 if fps >= 90 else 1
        src.update(w=w, h=h, fps=fps, step=step, has_real=fps >= 90, sid=f"s{len(plan) + 1}")
        L = (nb - 1) * step + 1                                        # source frames per window
        t0 = time.time()
        d1, ds = motion_scan(src["path"], step)
        tab = window_table(d1, ds, L, step, fps)
        if tab is None:
            print(f"skip {src['path']} (shorter than one {a.clip_len} s window)"); continue
        if src["start"] is not None:                                   # start=S: one pinned window
            s0 = min(int(round(src["start"] * fps)), len(tab[0]) - 1)
            ok = tab[0][s0] >= a.min_motion and not tab[1][s0] and not tab[2][s0]
            wins = [(s0, float(tab[0][s0]), bool(ok))]
        else:
            lo = warm_want * step                                      # the full warm-up fits before the window
            hi = len(tab[0]) - 1 - int(fps * 0.5)                      # keep off the very end
            if hi < lo: lo = hi = max(0, len(tab[0]) - 1)
            wins = pick_windows(tab, L, src["windows"] or a.windows_per_source, lo, hi, rng, a.min_motion)
        print(f"{src['sid']}: {src['path'].name} {fps:.0f} fps, {len(d1) / fps:.1f} s scanned in {time.time() - t0:.0f} s; "
              f"windows " + ", ".join(f"{s / fps:.2f}s (motion {m:.2f}{'' if ok else ', BELOW CHECK'})" for s, m, ok in wins),
              flush=True)
        plan.append((src, wins))
    if not plan: sys.exit("no usable sources")

    nwin = sum(len(ws) for _, ws in plan)
    clip_s = sum(len(ws) * (len(contestants) + src["has_real"]) * a.clip_len * src["w"] * src["h"] / (1920 * 1080)
                 for src, ws in plan)
    nst = sum(len(ws) * a.stills * (len(contestants) + src["has_real"]) * src["w"] * src["h"] / (1920 * 1080)
              for src, ws in plan)
    print(f"estimate: {nwin} windows, {sum(len(ws) * (len(contestants) + s['has_real']) for s, ws in plan)} clip encodes, "
          f"~{(clip_s * EST_MB_PER_CLIP_S + nst * EST_MB_PER_STILL) / 1e3:.2f} GB on disk, "
          f"~{clip_s * EST_S_PER_CLIP_S / 60:.0f} min to render (plus ~8 MB per 1080p pair of work files per window, freed after it)",
          flush=True)

    # ---- phase 2: render every window ----
    media.mkdir(parents=True)
    work = HERE / "work" / sid
    work.mkdir(parents=True)
    clips = []     # one entry per window (the page's "clip"); "src" groups the windows of one source
    fgcalls = []   # (window, contestant, warm-up pairs, clip pairs): one fgtest call each
    t_render = time.time()
    try:
        for src, wins in plan:
            w, h, fps, step, has_real = src["w"], src["h"], src["fps"], src["step"], src["has_real"]
            for wi, (s0, mscore, mok) in enumerate(wins):
                start = round(s0 / fps, 4)
                warm = min(warm_want, s0 // step)                       # warm-up base frames before the start
                if warm < warm_want:
                    print(f"warning: {src['path'].name}: only {warm} warm-up pairs fit before start {start}s (wanted {warm_want})")
                t0 = (s0 - warm * step) / fps
                nsrc = (warm + nb - 1) * step + 1                       # source frames decoded, warm-up included
                cid = f"c{len(clips) + 1}"
                cw = work / cid
                cw.mkdir()
                # still instants: generated frame indices, spread over the clip, seeded, off the edges
                lo, hi = 6, gen_frames - 6
                seg = (hi - lo) / a.stills            # one per equal segment, so they spread out
                stills = [lo + int(seg * i) + rng.randrange(max(1, int(seg))) for i in range(a.stills)]
                entry = {"id": cid, "src": src["sid"], "window": wi + 1, "windows": len(wins), "source": src["path"].name,
                         "genre": src["genre"], "src_fps": round(fps, 3), "start": start, "len": a.clip_len,
                         "motion": round(mscore, 2), "motion_ok": mok, "warm": warm, "w": w, "h": h, "has_real": has_real,
                         "contestants": list(range(len(contestants))) + ([REAL] if has_real else []),
                         "clip": {}, "stills": [{"k": k, "t": round(start + (k + 0.5) * step / fps, 4), "img": {}} for k in stills]}
                print(f"{cid}: {src['path'].name} window {wi + 1}/{len(wins)}, start {start}s, motion {mscore:.2f}, "
                      f"{nb} base frames + {warm} warm-up, real={has_real}", flush=True)

                # one decode pass from the warm-up start: base frames to disk (b0000 = first warm-up frame, b<warm> = clip
                # start), the window's full-rate stream straight into the real encode
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
                    cname = token(rng) + ".mp4"
                    enc = encoder(media / cname, w, h)
                    entry["clip"][ci] = "media/" + cname
                    # ONE fgtest call for the whole window, warm-up pairs first: one continuous Interpolator, as in game
                    args = [str(FGTEST), "--set", c["eff"], str(w), str(h)]
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

    for cid, cn, wu, gp in fgcalls:
        print(f"fgtest call: {cid} {cn}: {wu} warm-up + {gp} clip pairs in one call")
    trials = schedule(clips, a, rng)
    for src, _ in plan:
        tt = [t for t in trials if t["src"] == src["sid"]]
        print(f"{src['sid']} ({src['path'].name}): {sum(t['type'] == 'clip' for t in tt)} clip / "
              f"{sum(t['type'] == 'still' for t in tt)} still trials ({sum('repeat_of' in t for t in tt)} repeats); "
              f"clip trials per window " + ", ".join(
                  f"{c['id']}={sum(t['type'] == 'clip' and t['clip'] == c['id'] for t in tt)}"
                  for c in clips if c["src"] == src["sid"]))
    session = {
        "id": sid, "seed": seed, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "fgtest": fghash, "base_set": a.base_set, "warm": warm_want, "clip_len": a.clip_len, "fps": FPS_OUT,
        "encode": "ffmpeg " + " ".join(ENC),
        "motion_check": f"window mean abs diff of consecutive 60 fps base frames at {SCAN_W}x{SCAN_H} grey >= {a.min_motion}, "
                        f"no source frame pair > {CUT_DIFF:g} (cut), no {STATIC_SEC:g} s stretch < {STATIC_DIFF:g} (static)",
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
    rs = sum(c["len"] * (len(c["contestants"])) * c["w"] * c["h"] / (1920 * 1080) for c in clips)
    print(f"\n{len(trials)} trials -> {sess / 'index.html'}")
    print(f"actual: {du(sess) / 1e9:.2f} GB, render {time.time() - t_render:.0f} s "
          f"({(time.time() - t_render) / max(rs, 1e-9):.1f} s per 1080p clip second), total {time.time() - t_start:.0f} s")


def schedule(clips, a, rng):
    """clips = the windows. Fresh trials cycle through every (source, contestant pair), so each contestant meets every
    other on every source; each one then takes a window of that source: a clip trial the next window no clip trial of
    that source has shown yet (shuffled; a window repeats only once all of the source's windows are used, warned), a still
    trial the next unused (window, instant) of the source, round-robin over the windows. About repeat-frac of the trials
    re-show an earlier one with sides swapped, same window and instant. Every source gets at least one clip (video) trial
    and one still trial when trials >= 2 x sources (repeats are reduced to make room); below that a warning is printed and
    the clip trials are placed first."""
    srcs = list(dict.fromkeys(c["src"] for c in clips))
    ns = len(srcs)
    nrep = max(1, round(a.trials * a.repeat_frac)) if a.trials >= 6 else 0
    if a.trials >= 2 * ns:
        nrep = min(nrep, a.trials - 2 * ns)
    else:
        print(f"warning: --trials {a.trials} < 2 x {ns} sources: not every source gets both a clip and a still trial")
    nfresh = a.trials - nrep
    wins = {s: [c for c in clips if c["src"] == s] for s in srcs}
    cons = {s: wins[s][0]["contestants"] for s in srcs}
    combos = [(s, x, y) for s in srcs for i, x in enumerate(cons[s]) for y in cons[s][i + 1:]]
    if nfresh < len(combos):
        print(f"warning: {nfresh} fresh trials < {len(combos)} source/pair combinations; not every pairing is shown")
    pool = []

    def draw(sid=None):   # the next combo of the shuffled cycle (of source sid, if given)
        nonlocal pool
        for _ in range(2):
            for i, cmb in enumerate(pool):
                if sid is None or cmb[0] == sid:
                    return pool.pop(i)
            pool = combos[:]; rng.shuffle(pool)
        raise RuntimeError(f"no contestant pair on source {sid}")

    clip_left = {s: [] for s in srcs}     # windows not yet shown as a clip trial (refilled, reshuffled, when used up)
    still_left = {s: [] for s in srcs}    # (window id, still index), round-robin over the windows
    reused = []

    def next_clip(s):
        if not clip_left[s]:
            if any(t.get("src") == s and t["type"] == "clip" for t in fresh): reused.append(s)
            clip_left[s] = [c["id"] for c in wins[s]]; rng.shuffle(clip_left[s])
        return clip_left[s].pop()

    def next_still(s):
        if not still_left[s]:
            per = []
            for c in wins[s]:
                ks = list(range(len(c["stills"]))); rng.shuffle(ks); per.append([(c["id"], k) for k in ks])
            rng.shuffle(per)
            still_left[s] = [x for rnd in zip(*per) for x in rnd][::-1]   # popped from the end: round 1 first
        return still_left[s].pop()

    # reserved: one clip trial per source, then one still trial per source, as far as the fresh trials reach
    reserved = [(s, "clip") for s in srcs] + [(s, "still") for s in srcs]
    reserved = reserved[:nfresh]
    picks = [(draw(s), typ) for s, typ in reserved]
    rest = nfresh - len(picks)
    nstill = max(0, round(nfresh * a.still_frac) - sum(t == "still" for _, t in picks))
    nstill = min(nstill, rest)
    nclip = rest - nstill
    # the remaining picks: a clip trial goes to a source that still has an unshown window while any does
    unshown = {s: len(wins[s]) - sum(1 for (cs, _, _), t in picks if cs == s and t == "clip") for s in srcs}
    extra = [draw() for _ in range(rest)]
    order = list(range(rest)); rng.shuffle(order)
    types = ["still"] * rest
    for i in order:
        if nclip and unshown[extra[i][0]] > 0:
            types[i] = "clip"; unshown[extra[i][0]] -= 1; nclip -= 1
    for i in order:
        if nclip and types[i] == "still":
            types[i] = "clip"; nclip -= 1
    picks += list(zip(extra, types))
    fresh = []
    for (s, x, y), typ in picks:
        l, r = (x, y) if rng.random() < 0.5 else (y, x)
        t = {"type": typ, "src": s, "L": l, "R": r}
        if typ == "clip":
            t["clip"] = next_clip(s)
        else:
            t["clip"], t["still"] = next_still(s)
        fresh.append(t)
    if reused:
        print(f"warning: more clip trials than windows on {', '.join(sorted(set(reused)))}: some windows repeat as clip trials")
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
