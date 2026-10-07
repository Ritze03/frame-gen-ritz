# bench/blind

Blind A/B test of frame-generation versions, judged by eye. The builder renders every contestant on the
same clips, the page shows them in pairs without names, and the last screen produces a text prompt
(votes first, then the reveal) to paste back to the lead.

## Build a session

```sh
python3 bench/blind/build_session.py \
  --contestant 'v1: algo=v1' --contestant 'v2: algo=v2' \
  --source footage/battlefield.mp4,genre=fps \
  --source footage/cuts/racing_forza_1.mp4,start=12.5 \
  --trials 24 --clip-len 4 --seed 1
```

- `--contestant 'name: set-string'`: fgtest `--set` keys (`algo`, `trustLow`, `ui`, ...). The base set
  `globalFallback=1,ui=crosshairv2` (whole-frame fallback off and CrosshairV2 UI protection, as the user plays) goes
  first, so a contestant can override a key (`ui=off`); `--base-set` changes it. The page's prompt header prints it.
- `--source PATH[,start=S][,genre=G]`: without `start`, a seeded pick among the 3 highest-motion windows.
  The genre defaults to the `<genre>_` prefix of the file name.
  - Sources at 90 fps or more: the even frames are the 60 fps base, and the original frames take part as
    the hidden contestant `real`. Stills compare the held-out odd frame with each generated one.
  - Sources at 50-89 fps are interpolated 60 to 120 with no `real` contestant.
  - Sources under 50 fps are skipped.
- `--trials N` (fixed per session; about 10% are side-swapped repeats, `--repeat-frac`), `--still-frac`
  (0.5), `--stills` (instants per clip, 3), `--clip-len` (3-5 s), `--seed`, `--out` (default
  `bench/blind/sessions/`, gitignored). Every clip gets at least one clip (video) trial and one still trial when
  N >= 2 x the number of clips (repeats are cut to make room); a smaller N prints a warning and places one clip trial
  per clip first. The builder prints the trial mix per clip.
- `--warm N`: unscored warm-up pairs rendered before each clip's start (default 2, or `uiStillFrames - 1` = 7 when UI
  protection is on: a fresh Interpolator masks a static crosshair only after 8 still frames). Fewer fit when the clip
  starts within N base frames of the file start (warning).

The builder rebuilds `gpu/build/fgtest` (`CCACHE_DISABLE=1`), records its git hash, and runs it once per
contestant per clip: ONE call with the warm-up pairs and then all clip pairs in time order, so the Interpolator's state
(v2's motion history, the UI protection counters) runs continuously through the clip, as in game. That keeps all
outputs of a clip on disk at once (about 8 MB per 1080p pair: 2-2.5 GB of outputs plus as much input for 4-5 s). Every clip, `real` included, gets the
same x264 encode (`-preset slow -crf 10 -g 60 -keyint_min 60 -sc_threshold 0`, yuv420p, bt709, 120 fps).
Stills are lossless PNG. Work files go under `bench/blind/work/` and are deleted afterwards. A 4 s
1080p session with two clips and three contestants takes about 7 minutes, mostly x264.

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
