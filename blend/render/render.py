#!/usr/bin/env python3
"""Comparison clips and scores of the V3 blend versions: rule = v3Blend 0, then value = 1 + 3 * set + net, names A B C (set 0), A-A A-B A-C, AR-A .., AR2-A .., AR3-A ..
(whole picture, hard 10 % blocks, and edge-only: edge PSNR / MAE / share > 30 and the edge sharpness ratio).

Options anywhere on the command line: --vers rule,C,A-C (default: all 16), --out ROOT (default $BLEND/render; raw frames, json, videos, scores.md go below it),
--fgtest PATH (or env FGTEST), --no60 (gen/video: only the 30 -> 60 test), --nogrid (video: no quadrant grid).

    render.py extract CLIP [START]       decode 240 consecutive source frames to $BLEND/render/raw/CLIP/ (START default: first of 600 840 1080 360 120 whose
                                         stretch has no repeated frame and some motion); writes raw/CLIP/start.json
    render.py gen CLIP [SECONDS]         run fgtest for both tests and the selected versions, in chunks of 40 pairs (+2 warm-up pairs), every call under the GPU lock;
                                         resumable (a chunk's marker file), stops starting chunks after SECONDS (default 420)
    render.py score CLIP                 "30 -> 60" scores of the selected versions (merged into json/CLIP.json), differences to the rule (needs the rule's outputs), edge scores;
                                         with the rule selected also the rule error map and the 960x540 window with the largest mean rule error
    render.py video CLIP                 every mp4 (30to60/, 60to120/), the quadrant grid for the racing clips; 60 -> 120 differences to the rule
    render.py clean CLIP                 delete raw/CLIP; "clean CLIP outputs" deletes only the selected versions' output folders (source frames stay)
    render.py report                     scores.md and scores.json from render/json/*.json

Test "30 -> 60": inputs = the even frames of the stretch (pair k: prev = START + 2k, curr = START + 2k + 2), t = 0.5, compared with the real frame START + 2k + 1.
Test "60 -> 120": inputs = consecutive frames (pair k: START + k, START + k + 1), t = 0.5, nothing to score against.
Each version is one continuous stream (consecutive pairs share their path: the library's history behaves as in a game).
"""
import json, os, subprocess, sys, time
import numpy as np


def opt(name, default=None, flag=False):
    """pull --name [value] out of sys.argv"""
    if name in sys.argv:
        i = sys.argv.index(name)
        if flag:
            del sys.argv[i]; return True
        v = sys.argv[i + 1]; del sys.argv[i:i + 2]; return v
    return False if flag else default


NO60, NOGRID = opt("--no60", flag=True), opt("--nogrid", flag=True)
_vers, _out, _fg = opt("--vers"), opt("--out"), opt("--fgtest")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from common import BLEND, W, H, FRAME_BYTES, clip_path  # noqa: E402

FGTEST = _fg or os.environ.get("FGTEST", os.path.join(HERE, "..", "..", "gpu", "build-blend", "fgtest"))
LOCK = os.path.expanduser("~/tmp/fg-handoff/gpu.lock")
R = _out or os.path.join(BLEND, "render")
RAW = os.path.join(R, "raw")
ALL = [("rule", 0)] + [((c if st == 0 else f"{('A', 'AR', 'AR2', 'AR3')[st - 1]}-{c}"), 1 + 3 * st + k) for st in range(5) for k, c in enumerate("ABC")]
VERS = ALL if not _vers else [(n, dict(ALL)[n]) for n in _vers.split(",")]
CLIPS = ["racing_forza6_a", "racing_forza6_b", "racing_forza6_c", "fps_cs2_a", "minecraft_rtx_a", "indie_schedule1_a", "realworld_portland_walk_a"]
N = 240
CHUNK, WARM = 40, 2
MODES = (30,) if NO60 else (30, 60)
FONT = "/usr/share/fonts/noto/NotoSansMono-Regular.ttf"


def rd(p):
    return np.fromfile(p, np.uint8).reshape(H, W, 4)[:, :, :3]


def fr(clip, i):
    return f"{RAW}/{clip}/{i:06d}.rgba"


def start_of(clip):
    return json.load(open(f"{RAW}/{clip}/start.json"))["start"]


def out_path(clip, mode, ver, idx):
    return f"{RAW}/{clip}/out{mode}_{ver}/{idx:06d}.rgba"


def pairs(clip, mode):
    """-> [(prev, curr, out index)] ; out index = the real frame the output is compared with (30) / the earlier input frame (60)"""
    s = start_of(clip)
    if mode == 30:
        return [(s + 2 * k, s + 2 * k + 2, s + 2 * k + 1) for k in range(N // 2 - 1)]
    return [(s + k, s + k + 1, s + k) for k in range(N - 1)]


# ---------------------------------------------------------------- extract
def decode(clip, a, dest):
    cmd = ["ffmpeg", "-v", "error", "-i", clip_path(clip), "-vf", f"select='between(n\\,{a}\\,{a + N - 1})'", "-fps_mode", "passthrough",
           "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"]
    os.makedirs(dest, exist_ok=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME_BYTES * 2)
    prev, rep, mot = None, [], []
    for k in range(N):
        b = p.stdout.read(FRAME_BYTES)
        while len(b) < FRAME_BYTES:
            c = p.stdout.read(FRAME_BYTES - len(b))
            assert c, (clip, a, k)
            b += c
        cur = np.frombuffer(b, np.uint8)
        if prev is not None:
            d = float(np.abs(cur[::7].astype(np.int16) - prev[::7]).mean())
            mot.append(d)
            if d < 0.05:
                rep.append(a + k)
        prev = cur
        with open(f"{dest}/{a + k:06d}.rgba", "wb") as f:
            f.write(b)
    p.stdout.close(); p.wait()
    return rep, float(np.mean(mot))


def scan(clip):
    """-> (dup flags [n-1], motion [n-1]) of the whole clip from a 480x270 gray decode"""
    cmd = ["ffmpeg", "-v", "error", "-i", clip_path(clip), "-vf", "scale=480:270:flags=area,format=gray", "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
    a = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(-1, 270 * 480).astype(np.int16)
    d = np.abs(a[1:] - a[:-1]).mean(axis=1)
    return d < 0.05, d


def extract(clip, start=None):
    if start is None:
        dup, mot = scan(clip)
        best = None
        for a in range(120, len(dup) + 1 - N, 20):
            key = (int(dup[a:a + N - 1].sum()), abs(a - 600))
            if mot[a:a + N - 1].mean() > 1.0 and (best is None or key < best[0]):
                best = (key, a)
        start = best[1]
        print(clip, "scan: best start", start, "repeated frames", best[0][0], flush=True)
    d = f"{RAW}/{clip}"
    rep, mot = decode(clip, start, d)
    print(clip, "start", start, "repeated frames", len(rep), "mean consecutive abs diff %.2f" % mot, flush=True)
    json.dump({"start": start, "repeated": rep, "motion": mot}, open(f"{d}/start.json", "w"))


# ---------------------------------------------------------------- gen
def fgtest_cmd(ver, plist, clip, mode, warm):
    args = [FGTEST, "--set", f"algo=v3,v3Blend={ver}", str(W), str(H)]
    junk = f"{RAW}/{clip}/junk.rgba"
    for i, (p, c, o) in enumerate(plist):
        args += [fr(clip, p), fr(clip, c), junk if i < warm else out_path(clip, mode, ver, o)]
    return ["flock", LOCK, "ccorral", "--"] + args


def gen(clip, budget=420):
    t0 = time.time()
    for mode in MODES:
        P = pairs(clip, mode)
        for name, ver in VERS:
            os.makedirs(f"{RAW}/{clip}/out{mode}_{ver}", exist_ok=True)
            for ci, s in enumerate(range(0, len(P), CHUNK)):
                mark = f"{RAW}/{clip}/out{mode}_{ver}/.done{ci}"
                if os.path.exists(mark):
                    continue
                if time.time() - t0 > budget:
                    print("budget used up; run again", flush=True)
                    return False
                w = min(WARM, s)
                plist = P[s - w:s + CHUNK]
                cmd = fgtest_cmd(ver, plist, clip, mode, w)
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode:
                    sys.exit(f"fgtest failed {clip} {mode} {name} chunk {ci}: {r.stderr[-600:]}")
                open(mark, "w").close()
                print(clip, mode, name, "chunk", ci, "%.0fs" % (time.time() - t0), flush=True)
    return True


# ---------------------------------------------------------------- score
def blocks(e):  # e [H, W] -> [67, 120] block means (16x16, bottom 8 rows dropped)
    return e[:1072, :].reshape(67, 16, 120, 16).mean(axis=(1, 3))


def luma(a):  # a [H, W, 3] -> float32 L
    return a[:, :, 0] * np.float32(0.299) + a[:, :, 1] * np.float32(0.587) + a[:, :, 2] * np.float32(0.114)


def sobel(L):
    """gradient magnitude of L (3x3 Sobel, edge-replicated border)"""
    p = np.pad(L, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.sqrt(gx * gx + gy * gy)


def maxfilt5(g):
    """maximum over the 5x5 window around each pixel (edge-replicated border)"""
    p = np.pad(g, 2, mode="edge")
    m = p[:, :-4]
    for d in range(1, 5):
        m = np.maximum(m, p[:, d:d + g.shape[1]])
    r = m[:-4]
    for d in range(1, 5):
        r = np.maximum(r, m[d:d + g.shape[0]])
    return r


def edge_mask(real):
    """edge mask of the REAL frame [H, W, 3] -> (bool mask, its Sobel magnitude G): pixels whose 5x5-window maximum of G is >= the 90th percentile of G in the frame"""
    G = sobel(luma(real.astype(np.float32)))
    return maxfilt5(G) >= np.percentile(G, 90), G


def score(clip):
    s = start_of(clip)
    P = pairs(clip, 30)
    names = [n for n, _ in VERS]
    if "rule" not in names:
        names = ["rule"] + names
    vers = dict(ALL)
    acc = {n: dict(sq=0.0, ab=0.0, hi=0, px=0, sqh=0.0, abh=0.0, hih=0, pxh=0, d2=0, d10=0, dab=0.0, sqe=0.0, abe=0.0, hie=0, pxe=0, gout=0.0, greal=0.0) for n in names}
    emap = np.zeros((H, W), np.float64)
    nb = int(np.ceil(67 * 120 * 0.1))
    for p, c, o in P:
        real = rd(fr(clip, o)).astype(np.int16)
        out = {n: rd(out_path(clip, 30, vers[n], o)).astype(np.int16) for n in names}
        err = {n: out[n] - real for n in out}
        rb = blocks((err["rule"].astype(np.float32) ** 2).mean(axis=2))
        sel = np.zeros(67 * 120, bool); sel[np.argsort(rb.ravel())[-nb:]] = True
        m = np.zeros((H, W), bool); m[:1072] = np.repeat(np.repeat(sel.reshape(67, 120), 16, 0), 16, 1)
        em_, Greal = edge_mask(real)
        greal_sum = float(Greal[em_].sum())
        emap += (err["rule"].astype(np.float32) ** 2).mean(axis=2)
        for n in out:
            e = err[n]; a = acc[n]
            a["sq"] += float((e.astype(np.float64) ** 2).sum()); a["ab"] += float(np.abs(e).sum()); a["hi"] += int((np.abs(e).max(axis=2) > 30).sum()); a["px"] += H * W
            em = e[m]
            a["sqh"] += float((em.astype(np.float64) ** 2).sum()); a["abh"] += float(np.abs(em).sum()); a["hih"] += int((np.abs(em).max(axis=1) > 30).sum()); a["pxh"] += em.shape[0]
            ee = e[em_]
            a["sqe"] += float((ee.astype(np.float64) ** 2).sum()); a["abe"] += float(np.abs(ee).sum()); a["hie"] += int((np.abs(ee).max(axis=1) > 30).sum()); a["pxe"] += ee.shape[0]
            a["gout"] += float(sobel(luma(out[n].astype(np.float32)))[em_].sum()); a["greal"] += greal_sum
            d = np.abs(out[n] - out["rule"])
            a["d2"] += int((d.max(axis=2) > 2).sum()); a["d10"] += int((d.max(axis=2) > 10).sum()); a["dab"] += float(d.sum())

    def fin(a):
        mse = a["sq"] / (a["px"] * 3); mseh = a["sqh"] / (a["pxh"] * 3); msee = a["sqe"] / (a["pxe"] * 3)
        return dict(psnr=10 * np.log10(255 ** 2 / mse), mae=a["ab"] / (a["px"] * 3), share30=100 * a["hi"] / a["px"],
                    psnr_hard=10 * np.log10(255 ** 2 / mseh), mae_hard=a["abh"] / (a["pxh"] * 3), share30_hard=100 * a["hih"] / a["pxh"],
                    psnr_edge=10 * np.log10(255 ** 2 / msee), mae_edge=a["abe"] / (a["pxe"] * 3), share30_edge=100 * a["hie"] / a["pxe"],
                    edge_share=100 * a["pxe"] / a["px"], edge_sharpness=a["gout"] / a["greal"],
                    diff_rule_mean=a["dab"] / (a["px"] * 3), diff_rule_gt2=100 * a["d2"] / a["px"], diff_rule_gt10=100 * a["d10"] / a["px"])
    jp = f"{R}/json/{clip}.json"
    res = json.load(open(jp)) if os.path.exists(jp) else {}
    res.update({"clip": clip, "start": s, "frames_scored": len(P)})
    res.setdefault("versions", {}).update({n: fin(a) for n, a in acc.items()})
    # 960x540 window with the largest mean rule error (stride 8)
    ii = np.pad(emap, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best = (-1, 0, 0)
    for y in range(0, H - 540 + 1, 8):
        for x in range(0, W - 960 + 1, 8):
            v = ii[y + 540, x + 960] - ii[y, x + 960] - ii[y + 540, x] + ii[y, x]
            if v > best[0]:
                best = (v, x, y)
    res["window"] = {"x": best[1], "y": best[2], "w": 960, "h": 540, "mean_rule_mse": best[0] / (960 * 540 * len(P)), "whole_picture_mean_rule_mse": float(emap.sum() / (H * W * len(P)))}
    os.makedirs(f"{R}/json", exist_ok=True)
    json.dump(res, open(jp, "w"), indent=1)
    for n in names:
        print(clip, n, {k: round(v, 3) for k, v in res["versions"][n].items() if k in ("psnr", "psnr_edge", "mae_edge", "share30_edge", "edge_share", "edge_sharpness")}, flush=True)


# ---------------------------------------------------------------- video
COLOR = "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p"


def enc(path, fps, w=1920, h=1080, vf=COLOR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-vf", vf,
                             "-threads", "3", "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                             "-color_trc", "bt709", "-movflags", "+faststart", path], stdin=subprocess.PIPE)


def rgba(a):
    b = np.empty((a.shape[0], a.shape[1], 4), np.uint8); b[:, :, :3] = a; b[:, :, 3] = 255
    return b.tobytes()


def video(clip):
    s = start_of(clip)
    j = json.load(open(f"{R}/json/{clip}.json"))
    # real, all frames
    if not os.path.exists(f"{R}/30to60/{clip}_real.mp4"):
        e = enc(f"{R}/30to60/{clip}_real.mp4", 60)
        for i in range(s, s + N):
            e.stdin.write(np.fromfile(fr(clip, i), np.uint8).tobytes())
        e.stdin.close(); e.wait()
    for mode in MODES:
        P = pairs(clip, mode)
        for name, v in VERS:
            if os.path.exists(f"{R}/{'30to60' if mode == 30 else '60to120'}/{clip}_{name}.mp4"):
                continue
            t0 = time.time()
            e = enc(f"{R}/{'30to60' if mode == 30 else '60to120'}/{clip}_{name}.mp4", 60 if mode == 30 else 120)
            d2 = d10 = 0; dab = 0.0
            for k, (p, c, o) in enumerate(P):
                e.stdin.write(np.fromfile(fr(clip, p), np.uint8).tobytes())
                g = out_path(clip, mode, v, o)
                e.stdin.write(np.fromfile(g, np.uint8).tobytes())
                if mode == 60 and v:
                    d = np.abs(rd(g).astype(np.int16) - rd(out_path(clip, 60, 0, o)))
                    d2 += int((d.max(axis=2) > 2).sum()); d10 += int((d.max(axis=2) > 10).sum()); dab += float(d.sum())
            e.stdin.write(np.fromfile(fr(clip, P[-1][1]), np.uint8).tobytes())
            e.stdin.close(); e.wait()
            if mode == 60:
                j.setdefault("diff60", {})[name] = dict(mean=dab / (len(P) * H * W * 3), gt2=100 * d2 / (len(P) * H * W), gt10=100 * d10 / (len(P) * H * W))
            print(clip, mode, name, "video done %.0fs" % (time.time() - t0), flush=True)
    if clip.startswith("racing_forza6") and not NOGRID and len(VERS) >= 4:
        GV = VERS[:4]
        wx, wy = j["window"]["x"], j["window"]["y"]
        P = pairs(clip, 30)
        vf = (f"{COLOR},"  # labels in the corners of the four quadrants
              + ",".join(f"drawtext=fontfile={FONT}:text='{t}':x={x}:y={y}:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6"
                         for t, x, y in zip([n for n, _ in GV], (10, 970, 10, 970), (10, 10, 550, 550))))
        e = enc(f"{R}/30to60/{clip}_grid.mp4", 60, vf=vf)

        def quad(rgb4):
            q = np.empty((H, W, 4), np.uint8); q[:, :, 3] = 255
            for i, a in enumerate(rgb4):
                q[(i // 2) * 540:(i // 2) * 540 + 540, (i % 2) * 960:(i % 2) * 960 + 960, :3] = a[wy:wy + 540, wx:wx + 960]
            return q.tobytes()
        for k, (p, c, o) in enumerate(P):
            r = rd(fr(clip, p)); e.stdin.write(quad([r] * 4))
            e.stdin.write(quad([rd(out_path(clip, 30, v, o)) for _, v in GV]))
        e.stdin.write(quad([rd(fr(clip, P[-1][1]))] * 4))
        e.stdin.close(); e.wait()
        print(clip, "grid done", flush=True)
    json.dump(j, open(f"{R}/json/{clip}.json", "w"), indent=1)


# ---------------------------------------------------------------- report
def report():
    J = {c: json.load(open(f"{R}/json/{c}.json")) for c in CLIPS if os.path.exists(f"{R}/json/{c}.json")}
    json.dump(J, open(f"{R}/scores.json", "w"), indent=1)
    order = [n for n, _ in ALL]
    L = ["# Learned blend: V3 blend versions, whole-picture, hard-block and edge scores", "",
         "rule = v3Blend 0, then value = 1 + 3 * set + net (A, B, C, A-A, A-B, A-C, AR-A .. AR3-C; `blend/weights/README.md`); otherwise default V3 settings (`--set algo=v3,v3Blend=N`). Test 30 -> 60: inputs the even frames of 240 consecutive source frames, t = 0.5, compared with the real odd frame;",
         "one continuous fgtest stream per version (chunks of 40 pairs, each started 2 pairs early, those outputs discarded). PSNR pooled over RGB and all frames (levels 0..255); MAE in levels;",
         "'>30' = share of pixels with any channel error above 30 levels. 'Hard 10 %' = the 10 % of 16x16 blocks (per frame) where the rule's error is largest, the same blocks for all versions.",
         "'Edge' = pixels of an edge mask of the REAL frame: luma L = 0.299 R + 0.587 G + 0.114 B, Sobel magnitude G of L, mask = pixels where the maximum of G over the 5x5 window around the pixel is >= the 90th percentile of G in that frame;",
         "'mask %' = the mask's share of the picture. 'Sharpness' = mean Sobel magnitude of the OUTPUT's luma over the mask / the real frame's mean Sobel magnitude over the mask (pooled over frames; 1.0 = as sharp as the real frame, below 1 = softer).", ""]
    for key, title, cols in (("", "Whole picture", ("psnr", "mae", "share30")), ("_hard", "Hard 10 % blocks (chosen by the rule's error)", ("psnr_hard", "mae_hard", "share30_hard"))):
        L += [f"## {title}", "", "| clip | start | version | PSNR dB | MAE | share > 30 % |", "|---|---|---|---|---|---|"]
        for c, j in J.items():
            for n in order:
                if n in j["versions"]:
                    v = j["versions"][n]
                    L.append(f"| {c} | {j['start']} | {n} | {v[cols[0]]:.3f} | {v[cols[1]]:.3f} | {v[cols[2]]:.3f} |")
        L.append("")
    L += ["## Edge pixels of the real frame", "", "| clip | version | whole PSNR dB | edge PSNR dB | edge MAE | edge share > 30 % | sharpness | mask % |", "|---|---|---|---|---|---|---|---|"]
    for c, j in J.items():
        for n in order:
            if n in j["versions"] and "psnr_edge" in j["versions"][n]:
                v = j["versions"][n]
                L.append(f"| {c} | {n} | {v['psnr']:.3f} | {v['psnr_edge']:.3f} | {v['mae_edge']:.3f} | {v['share30_edge']:.3f} | {v['edge_sharpness']:.4f} | {v['edge_share']:.2f} |")
    L.append("")
    L += ["## Difference of each version to the rule (generated frames, any channel)", "", "| clip | version | 30->60 mean abs diff | 30->60 > 2 levels % | 30->60 > 10 levels % | 60->120 mean | 60->120 > 2 % | 60->120 > 10 % |", "|---|---|---|---|---|---|---|---|"]
    for c, j in J.items():
        for n in order[1:]:
            if n in j["versions"]:
                v = j["versions"][n]; d = j.get("diff60", {}).get(n, {})
                L.append(f"| {c} | {n} | {v['diff_rule_mean']:.3f} | {v['diff_rule_gt2']:.2f} | {v['diff_rule_gt10']:.2f} | {d.get('mean', float('nan')):.3f} | {d.get('gt2', float('nan')):.2f} | {d.get('gt10', float('nan')):.2f} |")
    L += ["", "## Start frames and crop windows", "", "| clip | start frame | repeated frames | grid window (x, y, 960x540) | mean rule MSE in window / whole picture |", "|---|---|---|---|---|"]
    for c, j in J.items():
        w = j["window"]
        L.append(f"| {c} | {j['start']} | {len(j.get('repeated', []))} | {w['x']}, {w['y']} | {w['mean_rule_mse']:.2f} / {w['whole_picture_mean_rule_mse']:.2f} |")
    L += ["", "## Commands", "", "fgtest per chunk (see `blend/render/render.py`, `fgtest_cmd`): `flock ~/tmp/fg-handoff/gpu.lock ccorral -- fgtest --set algo=v3,v3Blend=N 1920 1080 PREV CURR OUT [PREV CURR OUT ...]`,",
          "pairs (PREV, CURR, OUT) = (S+2k, S+2k+2, ...) for 30 -> 60 and (S+k, S+k+1, ...) for 60 -> 120, one t (default 0.5), 40 pairs per call after 2 warm-up pairs.",
          "Videos: ffmpeg libx264 -crf 12 -preset medium -pix_fmt yuv420p (bt709), 30to60 at 60 fps, 60to120 at 120 fps. Grid: first four selected versions, 1:1 crops."]
    open(f"{R}/scores.md", "w").write("\n".join(L) + "\n")
    print("report written")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "extract":
        extract(a[1], int(a[2]) if len(a) > 2 else None)
    elif a[0] == "gen":
        sys.exit(0 if gen(a[1], float(a[2]) if len(a) > 2 else 420) else 3)
    elif a[0] == "score":
        score(a[1])
    elif a[0] == "video":
        video(a[1])
    elif a[0] == "clean":
        if len(a) > 2 and a[2] == "outputs":
            for m_ in MODES:
                for _, v in VERS:
                    if v:  # the rule's outputs stay (the hard mask and the differences need them)
                        subprocess.run(["rm", "-rf", f"{RAW}/{a[1]}/out{m_}_{v}"])
        else:
            subprocess.run(["rm", "-rf", f"{RAW}/{a[1]}"])
    elif a[0] == "report":
        report()
    else:
        sys.exit(__doc__)
