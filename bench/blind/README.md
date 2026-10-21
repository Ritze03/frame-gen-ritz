# bench/blind

Blind A/B test of frame-generation versions, judged by eye. The builder renders every contestant on the
same clips, the page shows them in pairs without names, and the last screen produces a text prompt
(votes first, then the reveal) to paste back to the lead.

## Build a session

```sh
python3 bench/blind/build_session.py \
  --contestant 'v1: algo=v1' --contestant 'v2: algo=v2' \
  --source footage/battlefield.mp4,genre=fps \
  --source footage/obs1.mp4,windows=8 \
  --source footage/cuts/racing_forza6_a.mp4,start=12.5 \
  --windows-per-source 4 --trials 40 --clip-len 4 --seed 1
```

- `--contestant 'name: set-string'`: fgtest `--set` keys (`algo`, `trustLow`, `ui`, ...). The base set
  `globalFallback=1,ui=crosshairv2` (whole-frame fallback off and CrosshairV2 UI protection, as the user plays) goes
  first, so a contestant can override a key (`ui=off`); `--base-set` changes it. The page's prompt header prints it.
- `--source PATH[,windows=N][,start=S][,genre=G]`. The genre defaults to the `<genre>_` prefix of the file name.
- **The arena: many windows per source.** Each source gives `--windows-per-source` clip windows (default 4; `windows=N`
  per source overrides it; `start=S` pins one window at S instead). The builder scans the whole source once at 240x136 grey,
  cuts it into N equal segments and picks one window per segment, seeded, among the top quarter by motion of the windows
  that pass the **motion check** (`--min-motion`):
  - motion = mean abs diff between consecutive 60 fps base frames (source frames 2 apart at 120 fps) over the window,
    >= 4.0 (calibrated: the median 4 s window is ~9.5 on battlefield, ~5.2 on obs1, ~8 on obs2, 6-8.5 on the 60 fps cuts);
  - no scene cut: no consecutive source-frame pair above 40;
  - no static stretch (menu, pause, frozen capture): no 0.5 s inside with a mean below 1.0.

  A segment with no passing window is filled from passing windows elsewhere, then from cut-free ones by motion (marked
  "below the check", warned). A source too short for N non-overlapping windows gets fewer (warned). Each window's source
  time and motion score go into the builder output, `manifest.json` and the end prompt.
  - Sources at 90 fps or more: the even frames are the 60 fps base, and the original frames take part as
    the hidden contestant `real`. Stills compare the held-out odd frame with each generated one.
  - Sources at 50-89 fps are interpolated 60 to 120 with no `real` contestant.
  - Sources under 50 fps are skipped.
- `--trials N` (fixed per session; about 10% are side-swapped repeats, `--repeat-frac`), `--still-frac`
  (0.5), `--stills` (instants per window, 3), `--clip-len` (3-5 s), `--seed`, `--out` (default
  `bench/blind/sessions/`, gitignored). Fresh trials cycle through (source, contestant pair); a clip trial then takes a
  window of that source that no clip trial has shown yet (a window repeats only once all of the source's windows are used,
  warned), a still trial the next unused (window, instant), round-robin over the windows. Repeats show the same window
  (and instant) with sides swapped. Every source gets at least one clip (video) trial and one still trial when
  N >= 2 x the number of sources (repeats are cut to make room); a smaller N prints a warning and places one clip trial
  per source first. The builder prints the trial mix per source and the clip trials per window.
- `--warm N`: unscored warm-up pairs rendered before each clip's start (default 2, or `uiStillFrames - 1` = 7 when UI
  protection is on: a fresh Interpolator masks a static crosshair only after 8 still frames). Fewer fit when the clip
  starts within N base frames of the file start (warning).

The builder rebuilds `gpu/build/fgtest` (`CCACHE_DISABLE=1`), records its git hash, and runs it once per
contestant per window: ONE call with the warm-up pairs and then all clip pairs in time order, so the Interpolator's state
(v2's motion history, the UI protection counters) runs continuously through the window, as in game. That keeps all
outputs of a clip on disk at once (about 8 MB per 1080p pair: 2-2.5 GB of outputs plus as much input for 4-5 s). Every clip, `real` included, gets the
same x264 encode (`-preset slow -crf 10 -g 60 -keyint_min 60 -sc_threshold 0`, yuv420p, bt709, 120 fps).
Stills are lossless PNG. Work files go under `bench/blind/work/` (one window at a time) and are deleted afterwards.
No thread caps (ffmpeg, ninja use their defaults); CPU limits are system-wide.

**Size and time.** Before rendering, the builder prints an estimate (windows, clip encodes, GB, minutes); at the end the
actual size and the measured render rate. A 4 s 1080p clip encode is ~30-60 MB (crf 10), a PNG still ~3.5 MB. Example:
3 sources x 3 windows, 2 contestants (+ real on battlefield) = 21 clip encodes, ~0.9 GB, 10-20 min depending on load;
4 windows on 8 sources with 3 contestants is about 3 GB.

Output: `sessions/<id>/index.html`, `session.js` (the manifest as `window.SESSION`, so it works from
`file://`), `manifest.json` (the same data with the reveal in plain text, for the lead), and `media/`
with random file names.

## Run it

Open `sessions/<id>/index.html` in Chrome or Firefox, full screen. With VRR on, clips play at their native
120 fps.

| Key | Action |
|---|---|
| `1` / `2` / `3` | Left better / can't tell / right better (then next trial) |
| `N`, `Esc` | type a note for this trial, leave the note field |
| `Space` | play or pause both videos (kept in sync, left is the clock) |
| `←` / `→` | step both videos one frame (1/120 s); pauses first |
| `Home`, `L`, `S` | back to frame 0, loop on/off, ¼ speed |
| `Z` | 1:1 pixels (one image pixel per screen pixel); the mouse position pans both sides together |
| `F`, `X` / `Tab` | flip view: one side fills the window, swap to the other side at the same spot |
| `B` | back one trial (to change a vote) |

Progress is saved in `localStorage`, so a reload resumes. After the last vote the page shows the reveal and
the prompt with a Copy button.
