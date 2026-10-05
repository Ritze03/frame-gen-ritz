#pragma once

// pacing.h -- frame-generation PACING: which frame to show at which moment, and
// whether to generate at all. Host-agnostic: header-only C++17, no Vulkan, no OS
// calls, no clock of its own -- every time is passed in, as nanoseconds on any
// monotonic base. namespace framegen::pacing.
//
// The host (a compositor, a player, a test) feeds the Pacer three things --
// real-frame ARRIVALS (OnArrival), PAINTS (OnPaint, one per present tick, with the
// current Inputs) and, now and then, a STATUS request (TakeStatus) -- and applies
// the Decision each paint returns: which real frames / which t the renderer
// (framegen::Interpolator::recordSynth) is to generate, or that the real frame is
// to be shown as it is. The library's renderer needs none of it: this header only
// decides. INTEGRATION.md's "Pacing" section has the host contract and a minimal
// loop; gpu/pacingtest.cpp drives the whole thing with a fake clock.
//
// Vocabulary. A "present tick" is one display refresh: the moment the host can put
// a new image on screen (a vblank, a compositor frame callback, a timer standing
// in for either); its time, V, is when the image of THAT paint is shown. A
// "frame id" is any host-chosen identity of a real frame (a commit counter, a
// buffer sequence number): the same id means the same image, 0 means none.
//
// THREE FEATURES, ONE CONTENT TIMELINE. Frame generation, motion blur and the
// lag-spike buffer are separate switches that can all be on at once; each owns
// exactly one dimension of the same plan, so every combination falls out:
//   * the BUFFER   -- how far behind real time we show: the LAG-SPIKE BUFFER (below),
//                     plus the manual hook Inputs::extraDelayNs, both added to the base
//                     delay D (0 = none).
//   * FRAME GEN    -- which instants get an output frame (the cadence): fixed N,
//                     target fps, pause at refresh. Off (Inputs::frameGen false): one
//                     output per real frame, at that frame's own time plus D, and a gap
//                     the delay covers is filled by synths ("real-rate" planning).
//   * MOTION BLUR  -- what each output shows over time: a window [t0, t1] in the pair
//                     that brackets the output's content time (see BlurWindow).
// Every Decision carries {prevId, currId, t0, t1, showId, ...}: the host makes
// exactly ONE renderer call, recordSynthBlur( prev, curr, out, t0, t1 ), which is a
// plain synth when t0 == t1 and a real frame when it ends at 1. One estimate per
// real pair serves all of them; the cost guard budgets the estimate plus every
// synth of every blur sample; a reset resets all three; Report states each part
// separately. With frame generation off, no blur and no delay the pacer is a
// pass-through (Reason::Off, inert). Pause at refresh only concerns the cadence: a
// game that already reaches the output rate is planned real-rate when blur or the
// buffer is on (one blurred, delayed output per real frame), else passed through.
//
// THE MODEL: PER-TICK FRACTIONAL PACING.
// There is no "N" inside the pacer. Every output frame -- every present tick that
// the output cadence says should get a new image -- is generated for its OWN
// moment between two real frames:
//
//   * a real frame carries its arrival time (the earliest instant it could have
//     been shown: the game's frame fence signalling, see OnArrival); the frames
//     in play are A0 (before) and A1;
//   * an output frame painted for the present tick at time V shows CONTENT TIME
//         tau = V - D          (D = the delay, per priority, see below)
//     which lies between two real frames: t = (tau - A0) / (A1 - A0);
//   * t >= 1 (tau has caught up with the newest real frame): show that real
//     frame itself, no synth. Otherwise a synth at t.
//
// So "how many generated frames per real frame" is never chosen or held: it
// falls out of t from the latest arrivals, on the very next pair. A fixed
// multiplier N only sets the OUTPUT INTERVAL o = max(interval / N, 1/refresh)
// -- more frames than the display can show simply saturate at the refresh
// rate, nothing steps down. Target fps mode sets o = 1 / min(target, refresh).
// When o is longer than one tick only the ticks where a phase accumulator crosses
// o get a new image (Bresenham), so the average output rate is exact.
//
// DELAY D, per Priority (the user's "one for low latency and one for maximum
// smoothness"):
//   Low latency : D = max(0, i - o) + margin, i the smoothed interval, o the output
//     interval; margin = the 75th percentile of the arrival jitter, at most 1 ms and
//     20% of i (0 for a steady game). At a fixed N this is (N-1)/N of a game interval.
//     The latest interval replaces the smoothed one only when it is genuinely early
//     or late (more than max(2 ms, 15%) off): such a frame SNAPS the content time to
//     the new pair (latency-first rule), so uneven motion on really jittery frame
//     times is accepted. Minimal delay is the point: there is no floor of one o and
//     no allowance for the paint's lead, so a frame landing a little late may leave
//     one tick repeating its output (a few % of the ticks at +-1 ms jitter). Why not
//     the worst-case jitter + the lead + a floor (gamescope-ritz 8e806c4): they kept
//     every tick filled but put D at 8 ms at 120-210 fps on 280 Hz, no lower than
//     Smoothness -- "I feel like the input latency has become worse".
//   Smoothness  : D = median interval + margin (margin = max(2 ms, the spread
//     of recent intervals), capped at half an interval). tau advances evenly at
//     the output rate; the pair being played may be one OLDER than the newest
//     (so frames are queued by about one game interval). A late real frame
//     holds at t = 1 instead of hitching; an early one is absorbed by the
//     margin. Costs about one game frame of delay.
//   Why the pair can be older only in Smoothness: with D >= one interval the
//   newest real frame has usually ARRIVED while tau is still inside the pair
//   before it. The host's renderer therefore keeps the last kHistory (3) real
//   frames.
//
// tau is MONOTONIC: an output frame never shows content older than the one
// before it (tau >= tau_last), however D moves.
//
// PASS-THROUGH (the real frame, shown at once, renderer inert) only when:
//   - the game's rate already reaches the output rate (interval <= o, with a
//     ratio band for hysteresis: stop generating below 1.02, start again above
//     1.10 -- NOT a time hold);
//   - the renderer is unavailable (Inputs::rendererOk false -- it is still
//     driven, so it re-checks);
//   - the cost guard trips (below).
// VRR / tearing are NOT a pass-through reason: while generating, the host paces
// its own output frames on the tick timer exactly as on a fixed-refresh display
// (PaintTick() below is the one rule a host loop can use), and while passing
// through it behaves as it always did.
//
// COST GUARD ("Limit to GPU speed"). The renderer reports the estimate's and one
// synth's GPU time. Generation per game interval is estimate + the synths:
// when that would exceed 25% of the interval, the output rate is LOWERED
// (o raised) until it fits; if not even one generated frame per pair fits it
// passes through. A probe retries every 10 s. No timestamps, no guard.
// Inputs::costGuard false turns all of it off (costs are still read).
//
// LATENCY. A newer real frame is seen by the very next paint: nothing is ever
// queued except by Smoothness's deliberate delay.
//
// "The present tick" enters through Inputs::presentNs only, so a timer that is not
// the display's own can feed its own present time with no other change:
// tau = V - D is exactly as valid for it (see the next section).
//
// BEYOND THE REFRESH RATE (Inputs::pauseAtRefresh false). With pauseAtRefresh true
// (default) all of the above holds: the output is capped at the refresh and
// generation stops once the game alone reaches it. With it false the refresh cap
// is GONE:
//   * Fixed N : o = interval / N, no 1/refresh floor -- N-1 frames per real pair
//     however fast the game is (200 fps at 4x = 800 output frames a second);
//   * Target  : the target is not clamped to the refresh;
//   * pass-through only when the game itself reaches the output rate (never for
//     Fixed, since N >= 2; game fps >= target for Target), the renderer is
//     unavailable, or the cost guard trips -- the cost guard is what then bounds
//     the output.
// An output interval shorter than a tick cannot be driven by the display's tick,
// so the pacer asks for ExtraTimer(): the host arms a timer for each output
// frame's own time and paints on it (V = that time). That only makes sense where
// a present between ticks is shown or discarded without blocking -- a nested
// Wayland window (the host shows the newest buffer), or a tearing flip. Where it
// is not, the host says so with Inputs::canExceedRefresh false (a plain
// display without tearing, an offscreen sink, VR) and "off" then behaves
// exactly as "on": capped at the refresh, tick-paced. canExceedRefresh is a
// host-provided capability: the pacer cannot know what the display path allows.
//
// THE LAG-SPIKE BUFFER (Inputs::lagBuffer; off by default, then byte-identical).
// Some games hitch: a frame arrives one, two, three intervals late. The buffer keeps
// a small adaptive extra delay D_buf, so that when the next hitch comes the content
// time is still inside the pair that ends at the late frame and the gap is filled with
// generated frames instead of frozen. It is independent of the multiplier and of
// blur; it adds to D and to nothing else.
//   * DETECTION (OnArrival): an interval above kLagSpikeRatio (1.3; 1.75 until 2026-10-05) x
//     the smoothed interval AND at least kLagSpikeMinMs (2 ms) longer is a spike of size
//     interval - smoothed  ms ("missing frames x interval"). Kept in a ring of kLagSpikeMax
//     (time, size). Why 1.3: the user wanted a hiccup shorter than a missing frame (14 ms at
//     10 ms, 1.4x) to count and be bridged next time; ordinary jitter (60 fps +-4 ms peaks at
//     1.24x) stays under it, and the 2 ms floor keeps a fast game's noise (a 240 fps game's
//     4.2 ms interval x 1.3 is 1.2 ms) from being a spike. No jitter floor beyond that.
//   * JUDGEMENT: a spike larger than maxBufferMs is an OUTLIER (a shader compile, a
//     disk hitch: 10 fps): never sized for, only counted (Report::outliersIgnored).
//   * RECENCY: weight w(age) = 0.3 + 0.7 x (1 + cos(pi x age / lookback)) / 2 for
//     age <= lookback (1 when new, 0.3 at the end, smooth), 0 beyond. Base = max over
//     spikes of size x w: the newest big one dominates, older ones fade but are never
//     dropped inside the window, so a game that hitches every few minutes keeps a
//     steady buffer. While spikes recur (>= 1 in the window, >= 2 seen this session)
//     the base is at least kLagFloorFrac (0.5) x the session's EWMA of spike sizes
//     (alpha 0.15), so history is not thrown away. Target = base x 1.10 + 1 ms, clamped
//     to [0, maxBufferMs] (and to what kHistoryMax frames can hold at this game rate).
//   * STALL CREDIT (2026-10-05): a spike larger than the delay applied (m_flDelayMs, the
//     total D of the last paint; 0 when passing through) and within maxBufferMs grows D_buf
//     AT ONCE by the difference (NoteSpike notes it, UpdateBuffer spends it, never past the
//     target and so never past the history cap, ForceMax, or the instant-off). Why: the
//     display already stood still for the stall, so the content time may resume from where
//     it stopped (tau = V - D_new lands on the held instant; the existing tau monotonic
//     clamp keeps it from going back) instead of freezing again on the next spike. Measured
//     with the ramp alone (100 fps, max 50, 60 ms stalls at 3, 3.5, 4, 5 s from nothing):
//     freezes of 58, 42, 29 and 0 ms -- "it waits"; with the credit 58, 0, 0, 0. The user:
//     "it should still detect lags and based on that determine the buffer size, but
//     otherwise it should make sure that for example we have a buffer of 50 and usually we
//     get a frame every 10 milliseconds, but then there isn't a frame for 14 milliseconds,
//     then it should just generate the frames for that as soon as it can based on the next
//     frame, if the lag spike was bigger than the buffer, then it obviously can't do that.
//     But then it should increase the buffer and then it should start working again", and
//     "We need to keep ignoring everything above max, since thats what max is for."
//   * RAMP: D_buf otherwise never jumps. It moves toward the target by at most kLagRampEps (3 %)
//     of real time, i.e. content time plays at 0.97x while it grows and 1.03x while it
//     shrinks (tau stays monotonic): the margin above the credited size, a raised max, the
//     test modes and all shrinking. Shrinking starts only after the target has been
//     more than 3 ms below D_buf for 2 s (the look-back decay, not a flicker).
//     EXCEPTION: switching the buffer OFF (LagBufferSettings::enabled false) drops it to 0
//     on the very next paint: one forward step of tau, which stays monotonic.
//   * TEST MODE: ForceMin pins the target at 0, ForceMax at maxBufferMs (through the same
//     ramp, no band, no wait).
//   * HISTORY: HistoryDepth() = ceil((base delay + max(D_buf, target) + manual delay) /
//     interval) + 2, 3 with no delay: it grows AHEAD of the ramp and shrinks only
//     after a 2 s dwell and only past frames the content time has left.
// The buffer works because D >= spike: the late frame has arrived by the time the
// content time needs the gap filled. The first hitch of a session always freezes, and so does
// one larger than the delay then applied -- once: it grows the delay to its size on the spot.

// THE GENERATE / PASS DECISION (Pacer::Evaluate; 2026-10-05, from a user report).
// Whether the pacer generates or passes the game's frames through is decided on the ratio
// r = game interval / output interval: generating stops once r < kStopRatio (1.02, the game
// reaches the output rate) and starts again once r > kStartRatio (1.10). THE REPORT: the HUD
// in "Both" mode kept switching between "game > output" and the lone game number "for no
// apparent reason". THE MEASUREMENT: a 4-minute CS2 session at 280 Hz (Target = display
// refresh, Smoothness, Pause at refresh off) logged 246 alternating "generating" / "passing
// through" lines in 246 s (median 1.0 s apart, every pass-through "GameTooFast"), the game's
// measured rate wandering 210..330 fps around the 280 Hz target. THE CAUSE: r was taken from
// the SAME interval the pacing uses, the median of the last kIntervalWindow (8) arrival
// intervals: about 30 ms of history at 270 fps, which ordinary frame-time jitter swings by
// +-50 fps, so the 20 fps band (254.5 .. 274.5 fps at 280 Hz) was crossed constantly. A
// simulation driving this header (280 Hz, Target auto, 262 fps game, 10% jitter) generated
// 62% of the time with 660 flips a minute; a 25% jitter 1300; the same with an explicit
// target of 400 at 380 fps (900); a Fixed 3x with Pause at refresh on at 270 fps (500).
// THE FIX, two parts that act only on the DECISION (the 8-sample median still drives the
// pacing, tau, the margin, the cost guard and the blur: the output is as responsive as ever):
//   1. The ratio is judged on a STEADY rate (SteadyRate): a frame-weighted exponential mean of
//      the intervals, time constant kSteadyTauMs (300 ms), each interval clamped to
//      [mean / 2, 2 x mean] first (one 50 ms hitch among 3.7 ms frames moves it ~1%). The
//      band is kept: with 0.8% noise (10% jitter) instead of 15% it is now wider than the
//      noise, so it neither needs widening nor flaps.
//   2. A change of state must HOLD: the opposite condition continuously true for
//      kStopDwellMs (1000) to stop, kStartDwellMs (300) to start; leaving the condition
//      restarts the clock. Stopping late costs almost nothing (a generated frame the game
//      did not need); starting late costs smoothness, so it is the quicker of the two.
//   The first decision after a start or a change of settings uses the stop ratio alone, with
//   no dwell, as before. Video (OnFrame) keeps its exact PTS interval and no dwell.
// MEASURED, same simulation (mean of 10 seeds, 60 s each, flips per minute before -> after):
// 262 fps j10% 649 -> 0; 270 j10% 483 -> 0; 270 j25% 1254 -> 0; 270 with a +-15% slow drift
// 108 -> 1.5 (the drift genuinely crosses the band twice a period); Target 400 at 380 fps
// 864 -> 0; Fixed 3x Pause on at 270 fps 483 -> 0; a game wandering 210..330 fps with 1..3 s
// periods 137 -> 37 (a swing that fast and that wide is real, not noise; the dwell bounds it
// at one change per 1.3 s, and it only changes when sustained). A game that really drops
// 300 -> 150 fps starts generating 0.42 s later; one that really rises 150 -> 320 stops 1.40 s
// later; a 50 ms or 90 ms hitch changes nothing (a stall of kGapResetNs or more is the gap
// path, as before).
// THE STEADY "GENERATING" SIGNAL (Report::generating, steadyMultiplier): fgActive / activeN
// report the INSTANTANEOUS state. generating = the pacer is in the Generate state AND a generated
// frame (t < 1 from a pair) was presented within the last ~1 s: NO multiplier gate (2026-10-05).
// Why: it first demanded an effective multiplier of >= 1.2x to turn on (>= 1.1x to stay), and
// when the output was paint-limited (target 335, game ~177 fps, ~1.28x) the multiplier hovered
// at the gate so a HUD's "game > output" separator flipped to the lone number while frame
// generation was generating all along. The user: "If its generating any frames, it should show
// the '>'. Otherwise that differentiation is useless." steadyMultiplier stays, informational
// only (the real ratio, for a line like "1.28x"). A genuinely ~1x pass-through is not
// Generate (the decision's own band, see THE GENERATE / PASS DECISION), so it reads false.
// MEASURED (paint-limited extra-timer host, target 335, 177 fps game, steady multiplier 1.18 /
// 1.13 / 1.09 / 1.05 / 1.01): the old signal read "generating" 40% / 0% / 0% / 0% / 0% of
// the status samples while frames were being generated all along; the new one 100%.

// VIDEO / TIMESTAMPS MODE (Pacer::OnFrame; the default stays arrival-timed, byte-identical).
// A video's frames carry presentation timestamps (PTS) and the decoder has the next ones
// ready, so nothing needs to be guessed from arrival times and no latency is at stake. The
// host announces each frame with OnFrame( id, ptsNs, ... ) instead of OnArrival and gives
// the media clock in Inputs (mediaOffsetNs, playbackRate): the present time at which PTS p
// is to appear is  V = mediaOffsetNs + p / playbackRate. The paint at V then shows the
// content time  tau = (V - mediaOffsetNs) * playbackRate  -- the PTS that belongs to V, D = 0
// -- exactly: t = (tau - pts_k) / (pts_k+1 - pts_k) with no clamp, so 24 fps film on a
// 144 Hz display lands every output at its own content time (no 3:2-style judder). Frame
// generation's cadence (fixed N, target fps, pause at refresh), blur and the cost guard work
// as before; the lag-spike buffer, extraDelayNs and the delay priorities are inert (there are
// no arrival spikes to size a buffer for and D is 0). If the next frame is not there yet
// (a decoder hiccup) the last real frame is held at t = 1 -- never extrapolated -- and
// Report::reason is WaitingForDecoder. In this mode the pacer is never inert: even a
// pass-through must schedule (frames are announced AHEAD, so "the newest" is not what is
// due). Pause / seek: Discontinuity() (or Reset()). INTEGRATION.md has the host contract.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>

namespace framegen::pacing
{
	using Ns = uint64_t;

	constexpr Ns kNsPerMs = 1'000'000ull;

	// The game's frame interval = median of the last kIntervalWindow arrival
	// intervals, clamped to [1, 50] ms (1000 .. 20 fps).
	constexpr int    kIntervalWindow  = 8;
	constexpr int    kMinSamples      = 4;      // intervals needed before the estimate is trusted
	constexpr double kMinIntervalMs   = 1.0;
	constexpr double kMaxIntervalMs   = 50.0;
	// A gap longer than this between real frames drops the previous frame.
	constexpr Ns     kGapResetNs      = 100 * kNsPerMs;
	// Generation stops when game interval / output interval falls below
	// kStopRatio (the game reaches the output rate) and starts again once it is
	// back above kStartRatio. 98% .. 91% of the refresh at a fixed multiplier.
	constexpr double kStopRatio       = 1.02;
	constexpr double kStartRatio      = 1.10;
	// ... but that ratio is judged on the STEADY RATE (SteadyRate below), not on the 8-sample
	// median the pacing itself uses, and a change of state must also HOLD for a dwell. Why: a
	// game hovering near the output rate (a 210..330 fps CS2 at a 280 Hz target) swings its
	// 8-frame median (about 30 ms of history) by +-50 fps with ordinary frame-time jitter, which
	// crossed the 20 fps band once a second: the user's journal for a 4-minute session has 246
	// "generating" / "passing through" lines in 246 s (median 1.0 s apart, every pass-through
	// GameTooFast), and the HUD's "Both" readout flipped between "game > output" and the lone
	// game number "for no apparent reason". See "THE GENERATE / PASS DECISION".
	// Dwell to STOP generating is longer than to START: starting late costs smoothness (a
	// slow game shown unsmoothed), stopping late costs almost nothing (a generated frame in
	// a game that no longer needs it).
	constexpr double kStopDwellMs     = 1000.0;
	constexpr double kStartDwellMs    = 300.0;
	// The steady rate's time constant, and how far one interval may stray from the running
	// value before it is clamped (a hitch counts as two frame times, not as 50 ms).
	constexpr double kSteadyTauMs     = 300.0;
	constexpr double kSteadyClamp     = 2.0;
	// ExtraTimer() band: paint on its own timer once the output interval is more
	// than 2% shorter than a tick, back to the tick timer once it is more
	// than 2% longer (the same 2% the cadence in PaintGenerated() uses).
	constexpr double kExtraEnter      = 0.98;
	constexpr double kExtraLeave      = 1.02;
	// No output interval is shorter than this (2000 fps): a fixed 2x of a 1000 fps game, or any 4x+ of a 500+ fps one.
	constexpr double kMinOutputMs     = 0.5;
	// Generation (estimate + synths) may use at most this share of a game
	// interval.
	constexpr double kCostBudget      = 0.25;
	// After a cost-guard pass-through nothing is generated, hence nothing is
	// measured; retry this often (one generated frame per pair) to see whether
	// the load has gone, and give up on a probe that measured nothing after
	// kCostProbeMaxNs.
	constexpr Ns     kCostProbeNs     = 10'000'000'000ull;
	constexpr Ns     kCostProbeMaxNs  = 3'000'000'000ull;
	// Video mode: frames announced ahead of the one being shown that the pacer holds
	// (OnFrame returns false when full), and the t within which a synth snaps to a real frame.
	constexpr int    kVideoLookahead  = 8;
	constexpr double kPtsSnap         = 1e-4;
	// Status is worth publishing this often.
	constexpr Ns     kStatusPeriodNs  = 250 * kNsPerMs;
	// Report::generating (the steady "frame generation is visibly working" signal): the plan is
	// Generate and a generated frame was presented within kGeneratingRecentNs (no multiplier gate).
	// Report::steadyMultiplier (informational) is taken over the last kGeneratingWindowNs of status
	// windows.
	constexpr Ns     kGeneratingRecentNs = 1000 * kNsPerMs;
	constexpr Ns     kGeneratingWindowNs = 1000 * kNsPerMs;
	constexpr int    kStatusWindows      = 8;

	// The largest fixed multiplier a host should offer (Inputs::multiplier is 2..kMaxN).
	constexpr int    kMaxN            = 8;
	// Smoothness: margin = max(kMinMarginMs, spread), at most half an interval.
	constexpr double kMinMarginMs     = 2.0;
	// Low latency: margin = the P75 arrival jitter, at most 1 ms and at most 20% of an interval.
	constexpr double kLowLatencyMarginMaxMs = 1.0;
	constexpr double kLowLatencyMarginFrac  = 0.2;
	// Generated frames per real pair, well under the library's 32-set ring.
	constexpr int    kMaxSynthsPerPair = 24;
	// A synth is never made closer than this to either real frame; a newest-pair
	// t above kRealSnapT shows the real frame itself.
	constexpr double kTMin            = 0.02;
	constexpr double kRealSnapT       = 0.98;
	// Real frames kept for pair selection (the renderer's ring is the same size).
	constexpr int    kHistory         = 3;
	// With a delay (the lag-spike buffer, Inputs::extraDelayNs) the number of real frames kept
	// FOLLOWS the delay: ceil((base delay + buffer) / game interval) + kHistoryMargin
	// (HistoryDepth(), Report::historyFrames), never fewer than kHistory, never more than
	// kHistoryMax. The ceiling is NOT derived from kMinIntervalMs: 65 covers the largest buffer
	// (kLagMaxBufferMs, 250 ms) up to 250 fps (ceil(250/4) + 2); above that the buffer's cap
	// (LagTargetMs: (kHistoryMax - kHistory - kHistoryMargin) x interval) shrinks with the
	// interval instead (150 ms at 400 fps, 60 ms at 1000 fps), because each frame
	// the host keeps costs it one frame of memory (see INTEGRATION.md).
	constexpr int    kHistoryMargin   = 2;
	constexpr int    kHistoryMax      = 65;
	// Depth shrinks only after the wanted depth has stayed lower for this long (no allocating
	// back and forth), and only once the content time has moved past the frames to drop.
	constexpr double kDepthShrinkDwellMs = 2000.0;

	// ---- the lag-spike buffer's constants (see LagBufferSettings and "LAG-SPIKE BUFFER") ----
	constexpr double kLagSpikeRatio    = 1.3;     // an interval above this x the smoothed one is a spike (1.75 until 2026-10-05) ...
	constexpr double kLagSpikeMinMs    = 2.0;     // ... and at least this many ms longer than it (a 240 fps game's 4.2 ms x 1.3 is 1.2 ms of noise)
	constexpr int    kLagSpikeMax      = 64;      // spikes remembered (the lowest-weighted is replaced when full)
	constexpr double kLagWeightMin     = 0.3;     // a spike's weight at the end of the look-back window (1 when new, 0 beyond)
	constexpr double kLagMarginFrac    = 0.10;    // target = size x (1 + this) + kLagMarginMs
	constexpr double kLagMarginMs      = 1.0;
	constexpr double kLagFloorFrac     = 0.5;     // while spikes recur: the buffer is at least this x the long-run mean size
	constexpr double kLagEwmaAlpha     = 0.15;    // the long-run mean's weight of a new spike
	constexpr double kLagRampEps       = 0.03;    // the buffer changes by at most this per unit of real time (content speed 1 +- 3 %)
	constexpr double kLagRampMaxDtMs   = 100.0;   // one paint never moves the buffer by more than this x kLagRampEps
	constexpr double kLagShrinkBandMs  = 3.0;     // the buffer shrinks only once it is this far above its target ...
	constexpr double kLagShrinkDwellMs = 2000.0;  // ... for this long
	constexpr double kLagMaxBufferMs   = 250.0;   // LagBufferSettings::maxBufferMs's upper bound
	constexpr double kLagMaxLookbackSec = 600.0;  // ... and lookbackSec's (1 .. this)

	// The lag-spike buffer's test mode: pin the target instead of following the spikes (the
	// buffer still gets there through the same ramp), so a user can feel each end.
	enum class LagTestMode : uint8_t { Off, ForceMin, ForceMax };

	// The lag-spike buffer's settings. Off by default: the pacer is then byte-identical to one
	// without the buffer. Read fresh at every paint; every field is clamped to its range.
	struct LagBufferSettings
	{
		bool   enabled = false;
		double lookbackSec = 300.0;    // 1 .. kLagMaxLookbackSec: how far back spikes count
		double maxBufferMs = 50.0;     // 0 .. kLagMaxBufferMs: the buffer's cap; a spike larger than this is an outlier
		LagTestMode testMode = LagTestMode::Off;   // ForceMin: target 0; ForceMax: target maxBufferMs (needs enabled)
	};

	enum class Mode : uint8_t { Off, Fixed, Target };
	enum class Priority : uint8_t { LowLatency, Smoothness };

	// Why the output is (or is not) a generated frame; status only (Report::reason).
	// A host that has its own enum of the same list can static_assert the values.
	enum class Reason : uint8_t
	{
		Normal,
		Off,
		WarmingUp,
		CostGuard,
		GameTooFast,
		GameStalled,
		RendererUnavailable,
		WaitingForDecoder,   // video mode: the next frame is not announced yet; the last real frame is held
	};

	// ------------------------------------------------------------------
	//  Pure helpers
	// ------------------------------------------------------------------

	inline double ClampIntervalMs( double flMs )
	{
		return std::min( std::max( flMs, kMinIntervalMs ), kMaxIntervalMs );
	}

	// The sorted copy's helpers. n is tiny (<= 8): insertion sort.
	inline int SortedCopy( const double *v, int n, double *s )
	{
		n = std::min( n, kIntervalWindow );
		for ( int i = 0; i < n; i++ )
		{
			int j = i;
			while ( j > 0 && s[ j - 1 ] > v[ i ] )
			{
				s[ j ] = s[ j - 1 ];
				j--;
			}
			s[ j ] = v[ i ];
		}
		return n;
	}

	// The TYPICAL arrival jitter of v[0..n) (0 for fewer than 3): the 75th percentile
	// (nearest rank) of the absolute deviation from the median. Low latency's margin;
	// a hitch or two in the window do not move it, unlike a worst-case deviation.
	inline double P75DevOf( const double *v, int n );

	// Median of v[0..n): the middle value, or the mean of the two middle ones
	// for an even count.
	inline double MedianOf( const double *v, int n )
	{
		if ( n <= 0 )
			return 0.0;
		double s[ kIntervalWindow ] = {};
		n = SortedCopy( v, n, s );
		return ( n & 1 ) ? s[ n / 2 ] : 0.5 * ( s[ n / 2 - 1 ] + s[ n / 2 ] );
	}

	inline double P75DevOf( const double *v, int n )
	{
		if ( n < 3 )
			return 0.0;
		n = std::min( n, kIntervalWindow );
		const double flMed = MedianOf( v, n );
		double d[ kIntervalWindow ] = {};
		for ( int i = 0; i < n; i++ )
			d[ i ] = std::fabs( v[ i ] - flMed );
		double s[ kIntervalWindow ] = {};
		SortedCopy( d, n, s );
		return s[ ( 3 * n + 3 ) / 4 - 1 ];     // ceil( 0.75 n ) - 1
	}

	// A robust spread of v[0..n): second-largest minus median, so one hitch in
	// the window does not count but a jittery game does. 0 for fewer than 3.
	inline double SpreadOf( const double *v, int n )
	{
		if ( n < 3 )
			return 0.0;
		double s[ kIntervalWindow ] = {};
		n = SortedCopy( v, n, s );
		return std::max( 0.0, s[ n - 2 ] - MedianOf( v, n ) );
	}

	// The output interval o in milliseconds, before the cost guard.
	//   Fixed : max(interval / n, 1 / refresh)  -- never faster than the display
	//   Target: 1 / min(target, refresh); target 0 = the display's refresh
	// bCapAtRefresh false (Pause at refresh rate off, on a backend that can show
	// it) drops the refresh from both: Fixed = interval / n, Target = the target
	// itself (0 still means the display's refresh).
	inline double OutputIntervalMs( Mode eMode, int nMultiplier, int nTargetFps, double flIntervalMs, double flRefreshHz, bool bCapAtRefresh = true )
	{
		const double flTickMs = 1000.0 / std::max( flRefreshHz, 1.0 );
		if ( eMode == Mode::Target )
		{
			const double flTarget = nTargetFps > 0
				? ( bCapAtRefresh ? std::min( double( nTargetFps ), flRefreshHz ) : double( nTargetFps ) )
				: flRefreshHz;
			return std::max( 1000.0 / std::max( flTarget, 1.0 ), kMinOutputMs );
		}
		const int n = std::max( nMultiplier, 1 );
		const double flOut = flIntervalMs / double( n );
		return bCapAtRefresh ? std::max( flOut, flTickMs ) : std::max( flOut, kMinOutputMs );
	}

	// The time the next extra-timer output frame is painted for: one output
	// interval after the previous one's, never in the past (a late wakeup does
	// not make up lost slots with a burst), and at once when there is no previous.
	// ulLastTarget is the previous TARGET (the presentNs the paint was given), NOT the
	// time the paint ended: anchored to the target a paint shorter than the interval does
	// not stretch the cadence (period o); anchored to the paint's end the period would be
	// o + paint time (a user measured 1 / (o + 1.05 ms): 335 -> ~247 fps). Only a tick
	// more than a whole interval late restarts the grid at now.
	inline Ns NextExtraTargetNs( Ns ulLastTarget, Ns ulNow, double flOutputMs )
	{
		if ( ulLastTarget == 0 )
			return ulNow;
		return std::max( ulLastTarget + Ns( flOutputMs * 1e6 ), ulNow );
	}

	// Low latency: the latest real interval minus the output interval, floored.
	inline double LowLatencyDelayMs( double flLatestIntervalMs, double flOutputMs )
	{
		return std::max( 0.0, flLatestIntervalMs - flOutputMs );
	}

	// Low latency's interval for D (2026-10-04): the SMOOTHED interval, unless the
	// latest real interval is genuinely early or late -- off the smoothed one by
	// more than max(2 ms, 15%) -- in which case D snaps to the latest. Using the
	// latest interval always made D jump from pair to pair with ordinary arrival
	// jitter (a steady 30 fps game capped by MangoHud arrives +-1 ms), so the
	// content time overshot the newest frame early and an output frame was lost.
	inline double LowLatencyIntervalMs( double flSmoothMs, double flLatestMs )
	{
		if ( flLatestMs <= 0.0 )
			return flSmoothMs;
		return std::fabs( flLatestMs - flSmoothMs ) > std::max( 2.0, 0.15 * flSmoothMs ) ? flLatestMs : flSmoothMs;
	}

	// Low latency's jitter margin (2026-10-04, redone): the TYPICAL arrival jitter
	// (P75DevOf), at most 1 ms and at most 20% of the interval, so a steady game pays
	// nothing and a jittery one at most a millisecond. There is deliberately NO paint-
	// lead term and no floor of one output interval: Low latency means minimal delay,
	// D = max(0, i - o) + this, and a real frame that lands late may leave one tick
	// repeating the previous image (Smoothness is the priority that never does).
	inline double LowLatencyMarginMs( double flJitterP75Ms, double flIntervalMs )
	{
		return std::min( std::max( flJitterP75Ms, 0.0 ), std::min( kLowLatencyMarginMaxMs, kLowLatencyMarginFrac * std::max( flIntervalMs, 0.0 ) ) );
	}

	// Smoothness: the median interval plus the margin.
	inline double SmoothnessMarginMs( double flIntervalMs, double flSpreadMs )
	{
		return std::min( std::max( kMinMarginMs, flSpreadMs ), 0.5 * flIntervalMs );
	}
	inline double SmoothnessDelayMs( double flIntervalMs, double flSpreadMs )
	{
		return flIntervalMs + SmoothnessMarginMs( flIntervalMs, flSpreadMs );
	}

	// How many generated frames per game interval the GPU budget allows:
	// floor((25% of the interval - estimate) / one synth); 0 = not even one;
	// INT_MAX = unknown (no guard). Costs < 0 = no timings.
	inline int MaxSynthsPerPair( double flIntervalMs, double flEstimateMs, double flSynthMs )
	{
		if ( flEstimateMs < 0.0 || flSynthMs < 0.0 )
			return INT_MAX;
		const double flRoom = kCostBudget * flIntervalMs - flEstimateMs;
		if ( flRoom <= 0.0 )
			return 0;
		const double flSynth = std::max( flSynthMs, 1e-4 );
		return int( std::min( std::floor( flRoom / flSynth + 1e-9 ), double( kMaxSynthsPerPair ) ) );
	}

	// A host-loop helper: which main-loop iterations may paint. On a fixed-refresh
	// display only the present-tick timer's ticks do (bLoopWantsPaint ==
	// bFromTickTimer). Under VRR a loop typically treats EVERY iteration as a tick
	// (a frame arrival paints at once) and a tearing surface paints on arrival too;
	// while generating, the output frames would then go out at arrival times,
	// bunched, with the content times in the wrong places. So while generating
	// (bGenerating) only timer ticks paint -- the display is still asked for
	// nothing (VRR stays on, the flip is simply a normal one at the tick), and
	// while passing through the loop behaves exactly as without frame generation.
	//
	// When the pacer runs its own output timer (ExtraTimer(): pauseAtRefresh false
	// and the output interval shorter than a tick, bExtra) ONLY that timer's ticks
	// paint -- the tick timer's ticks would add output frames at times that are
	// not the output cadence's.
	inline bool PaintTick( bool bGenerating, bool bFromTickTimer, bool bLoopWantsPaint, bool bExtra = false, bool bFromExtraTimer = false )
	{
		if ( bGenerating && bExtra )
			return bFromExtraTimer;
		return bGenerating ? bFromTickTimer : bLoopWantsPaint;
	}

	// Ratio of the game's interval to the output interval: > 1 means there is
	// room to generate; the multiplier the display would be fed.
	inline double OutputRatio( double flIntervalMs, double flOutputMs )
	{
		return flOutputMs > 0.0 ? flIntervalMs / flOutputMs : 0.0;
	}

	// ------------------------------------------------------------------
	//  Motion-blur window
	// ------------------------------------------------------------------
	//
	// Where, inside the current real pair, the motion-blur pass is to average: the
	// span [t0, t1] of the pair's own timeline (0 = the previous real frame, 1 = the
	// newest), the arguments of the renderer's
	//     recordSynthBlur( cb, prev, curr, out, t0, t1 )
	// (the interpolation-based motion-blur pass of framegen::Interpolator; prev/curr
	// are the pair the output is made from, i.e. Decision::prevId/currId, which for a
	// real-frame output with blur is the previous real frame and that frame).
	// pacing.h does not depend on that pass; this function only does the arithmetic,
	// so every host's blur logic is the same. The Pacer calls it for every output
	// and puts the result in Decision::t0/t1.
	//
	// The window always ENDS at the instant the output shows (t): a trailing shutter.
	// It needs no future frame, so it adds no delay -- a centred window would need
	// the NEXT pair for the output that is the real frame itself (t = 1). It never
	// leaves [0, 1]. `amount` is the shutter, 0..1 (clamped), by what it is measured in:
	//
	//   Relative::ShownFrame -- in OUTPUT intervals (the time one displayed frame is
	//     on screen). The window is [t - s*o/g, t] with o the output interval and g
	//     the game interval, i.e. for output m of M per real pair (t = m/M, o/g = 1/M)
	//     [(m - s)/M, m/M]; s = 1 tiles the pair exactly, s = 0.5 is a 180 degree
	//     shutter. Blur only (no interpolation, o = g, t = 1): [1 - s, 1]. A host whose
	//     t does not sit on m/M (a fractional cadence) gets the lower end clamped at 0:
	//     the window is cut at the pair's start, never moved.
	//   Relative::GameFrame  -- in REAL-frame intervals: the "film look", a shutter as
	//     long as a camera at the game's own rate. Window [a, a + L], L = amount,
	//     a = clamp(t - L, 0, 1 - L): the length is kept and the window is slid into
	//     the pair (for t < L it looks slightly ahead of t, and neighbouring outputs'
	//     windows overlap). The intervals are not used.
	//
	// amount <= 0 (or NaN), or an unusable interval for ShownFrame, is a point,
	// t0 == t1 == clamp(t): no blur. t is clamped first; NaN counts as 0.
	enum class BlurRelative : uint8_t { ShownFrame, GameFrame };

	struct BlurSpan
	{
		float t0 = 0.0f;
		float t1 = 0.0f;
	};

	inline BlurSpan BlurWindow( double t, double outputIntervalNs, double gameIntervalNs, double amount, BlurRelative eRelative )
	{
		auto Clamp01 = []( double v ) { return v != v ? 0.0 : std::min( std::max( v, 0.0 ), 1.0 ); };
		const double flT = Clamp01( t );
		BlurSpan w;
		w.t0 = w.t1 = float( flT );
		if ( !( amount > 0.0 ) )
			return w;
		const double flS = std::min( amount, 1.0 );
		if ( eRelative == BlurRelative::GameFrame )
		{
			const double flA = std::min( std::max( flT - flS, 0.0 ), 1.0 - flS );
			w.t0 = float( flA );
			w.t1 = float( flA + flS );
			return w;
		}
		if ( !( outputIntervalNs > 0.0 ) || !( gameIntervalNs > 0.0 ) )
			return w;
		w.t0 = float( Clamp01( flT - flS * outputIntervalNs / gameIntervalNs ) );
		return w;
	}

	// ------------------------------------------------------------------
	//  Interval estimator
	// ------------------------------------------------------------------

	// Fed one real-frame ARRIVAL time per frame. Not display latch times: those are
	// quantised to the refresh (+-7 ms for 60 fps on 144 Hz), arrival is not.
	class IntervalEstimator
	{
	public:
		void Clear()
		{
			m_nCount = 0;
			m_nHead = 0;
		}

		void AddInterval( Ns ulIntervalNs )
		{
			m_flMs[ m_nHead ] = double( ulIntervalNs ) / double( kNsPerMs );
			m_nHead = ( m_nHead + 1 ) % kIntervalWindow;
			m_nCount = std::min( m_nCount + 1, kIntervalWindow );
		}

		int  Count() const { return m_nCount; }
		bool Valid() const { return m_nCount >= m_nMin; }
		// Video (timestamps) mode trusts exact PTS deltas after one interval.
		void SetMinSamples( int n ) { m_nMin = std::max( n, 1 ); }

		// Median of what we have, clamped. 0 when there is nothing yet.
		double IntervalMs() const
		{
			if ( m_nCount == 0 )
				return 0.0;
			return ClampIntervalMs( MedianOf( m_flMs, m_nCount ) );
		}

		// Jitter of the recent intervals (SpreadOf), in milliseconds.
		double SpreadMs() const { return SpreadOf( m_flMs, m_nCount ); }
		// The 75th percentile of the absolute deviation from the median, ms.
		double JitterP75Ms() const { return P75DevOf( m_flMs, m_nCount ); }

	private:
		double m_flMs[ kIntervalWindow ] = {};
		int m_nCount = 0;
		int m_nHead = 0;
		int m_nMin = kMinSamples;
	};

	// The STEADY game interval the generate / pass-through decision is made on (the pacing
	// maths keeps IntervalEstimator's 8-sample median: it must follow the game frame to
	// frame; the decision must not). An exponential mean of the real intervals with a time
	// constant of kSteadyTauMs, i.e. about 0.3 s of frames, whatever the frame rate:
	//   * it starts as the median of the first kIntervalWindow intervals (as robust as the
	//     estimator), then each interval x moves it by  w = max( mean / tau, 1 / n ) :
	//     a per-FRAME weight of "one tau's worth of frames" (n is the interval count: a
	//     running mean while the history is young). Frame-weighted on purpose: the mean of
	//     the intervals is the true frame rate's reciprocal, a time-weighted mean of them
	//     over-reads the interval by 1 + cv^2 (6% at 25% jitter);
	//   * x is first clamped to [mean / kSteadyClamp, mean x kSteadyClamp] (2): one 50 ms
	//     hitch among 3.7 ms frames then moves it by about 1% (and it decays with tau), while
	//     a game that really drops from 300 to 150 fps still pulls it all the way in ~0.1 s.
	// Noise is about cv / sqrt( 2 tau / interval ): 0.8% at 10% jitter, 2% at 25% (270 fps).
	// (An EMA rather than a median over a 0.5..1 s window: O(1) with no storage, where such a
	// window holds up to ~1000 intervals at 1000 fps, and the clamp gives it the median's
	// robustness to a hitch. See "THE GENERATE / PASS DECISION" for the measurements.)
	class SteadyRate
	{
	public:
		void Clear()
		{
			m_nN = 0;
			m_flMs = 0.0;
		}

		void AddInterval( double flIntervalMs )
		{
			const double x = ClampIntervalMs( flIntervalMs );
			if ( m_nN < kIntervalWindow )
			{
				m_flSeed[ m_nN++ ] = x;
				m_flMs = MedianOf( m_flSeed, m_nN );
				return;
			}
			m_nN = std::min( m_nN + 1, 1 << 20 );
			const double xc = std::min( std::max( x, m_flMs / kSteadyClamp ), m_flMs * kSteadyClamp );
			const double w = std::min( 1.0, std::max( m_flMs / kSteadyTauMs, 1.0 / double( m_nN ) ) );
			m_flMs += w * ( xc - m_flMs );
		}

		bool   Valid() const { return m_nN >= kMinSamples; }
		double IntervalMs() const { return m_nN == 0 ? 0.0 : ClampIntervalMs( m_flMs ); }

	private:
		double m_flSeed[ kIntervalWindow ] = {};
		double m_flMs = 0.0;
		int    m_nN = 0;
	};

	// ------------------------------------------------------------------
	//  The pacer
	// ------------------------------------------------------------------

	// What the real frames look like (size and pixel format, in whatever encoding
	// the host likes: a DRM fourcc, a VkFormat); a change is a reset, because two
	// differently shaped frames cannot be blended.
	struct FrameFormat
	{
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t format = 0;
		bool operator==( const FrameFormat &o ) const { return width == o.width && height == o.height && format == o.format; }
		bool operator!=( const FrameFormat &o ) const { return !( *this == o ); }
	};

	class Pacer
	{
	public:
		// Everything the host reads fresh at each paint.
		struct Inputs
		{
			Ns     now = 0;            // the clock, for gaps, probes and status
			Ns     presentNs = 0;       // V: the time this paint's image is shown (0 = now)
			Mode   mode = Mode::Fixed;
			int    multiplier = 2;     // Fixed: 2..kMaxN
			int    targetFps = 0;      // Target: 0 = the display's refresh, else 30..1000
			Priority priority = Priority::LowLatency;
			double refreshHz = 60.0;   // the refresh the present-tick timer paces against
			// "Pause at refresh rate" (true = capped at the refresh, the default) and
			// whether the host can show a present between ticks (a nested Wayland window,
			// a tearing flip). The cap applies unless the switch is off AND it can.
			bool   pauseAtRefresh = true;
			bool   canExceedRefresh = false;
			// "Limit to GPU speed" (the cost guard). false: costs are still read, but
			// the guard never caps the output, never passes through (CostGuard) and
			// never probes; switching it off lifts a cap at once.
			bool   costGuard = true;
			float  estimateMs = -1.0f; // RenderStatus::lastEstimateMs, < 0 = n/a
			float  synthMs = -1.0f;    // RenderStatus::lastSynthMs, < 0 = n/a
			uint32_t costSeq = 0;      // RenderStatus::costSeq: changes with every new measurement
			bool   rendererOk = true;  // RenderStatus::reason == Ok

			// --- the three features, each owning one dimension of the content timeline ---
			// Frame generation: WHICH INSTANTS get an output frame (the cadence). false =
			// off: one output per real frame, at that frame's own time (plus the delay
			// below); mode/multiplier/targetFps/priority/pauseAtRefresh are then unused.
			// With frame generation off, no blur and no delay the pacer is a pure
			// pass-through (Reason::Off, inert).
			bool   frameGen = true;
			// Lag-spike buffer: HOW FAR BEHIND real time frames are shown, sized from the
			// game's own frame-time spikes (see "THE LAG-SPIKE BUFFER"). Added to the base delay
			// (Low latency / Smoothness while frame generation is on, 0 while it is off).
			LagBufferSettings lagBuffer;
			// A fixed extra delay on top of the buffer (a host's manual hook; 0 = none). A
			// change is followed at once (the content time never goes backwards).
			Ns     extraDelayNs = 0;
			// Motion blur: WHAT EACH OUTPUT SHOWS OVER TIME (the shutter window, ending at
			// the output's content time). Active when blur && blurAmount > 0 &&
			// blurSamples >= 2. blurSamples only enters the cost guard (each output
			// costs blurSamples synths); the pacer returns the window, the renderer
			// averages it (recordSynthBlur).
			bool   blur = false;
			int    blurSamples = 0;
			double blurAmount = 0.5;   // 0..1, the shutter (see BlurWindow)
			BlurRelative blurRelative = BlurRelative::ShownFrame;

			// --- video / timestamps mode (only read after OnFrame; see the header) ---
			// The media clock: PTS p is to be shown at V = mediaOffsetNs + p / playbackRate.
			// Read fresh at each paint (a rate change or an A/V trim takes effect at once).
			int64_t mediaOffsetNs = 0;
			double  playbackRate = 1.0;
		};

		// What to do for this paint.
		struct Decision
		{
			uint64_t newestId = 0;     // the newest real frame (the renderer copies it into its ring on first sight)
			uint64_t prevId = 0;       // the pair the output is made from (0 when showing a real frame as it is, see t0/t1)
			uint64_t currId = 0;
			uint64_t outId = 0;        // the output frame's identity: the same outId is the same image
			float    t = 1.0f;         // the content instant this output shows, in the pair's own time: (0,1) a point between prevId and currId; 1: a real frame
			// THE HOST'S ONE RENDERER CALL. prevId != 0: recordSynthBlur( prev, curr, out, t0, t1 )
			// with the pair (prevId, currId); it collapses to recordSynth( t1 ) when t0 == t1
			// (no blur), and to the real frame when t1 is 1 and curr is shown. prevId == 0:
			// no renderer work, show the real frame showId. t0 == t1 == t when blur is off;
			// with blur [t0, t1] is BlurWindow() of t (it ends at t, except the film look
			// near the start of a pair, which slides the window into the pair).
			float    t0 = 1.0f;
			float    t1 = 1.0f;
			uint64_t showId = 0;       // the real frame to show when prevId == 0 (the newest, unless a lag-spike buffer delays)
			bool inert = false;        // the renderer does nothing at all (not even the ring copy)
			bool reset = false;        // reset the renderer (forget its previous real frames) BEFORE this paint
			bool repaintNext = false;  // another output is due: paint again at the next present tick
			bool newOutput = false;    // this paint presents a new output frame (counted in presentedFps)
			// A cadence-skip tick (the output interval is longer than a tick and no
			// new output is due): nothing to paint or commit, the display keeps the last
			// buffer. Only ever set when OnPaint() was told the paint is skippable. The
			// pacer consumed nothing: the next paint sees the same fresh real frame.
			bool skip = false;
		};

		// Status, for a host's HUD / log (TakeStatus).
		struct Report
		{
			float  gameFps = 0.0f;
			float  presentedFps = 0.0f;
			float  effectiveMultiplier = 0.0f; // presented / game while generating
			float  targetFps = 0.0f;           // Target mode: the rate aimed for (clamped to the refresh), else 0
			int    chosenN = 0;                // Fixed: the multiplier; Target: 0
			int    activeN = 0;                // round(effective multiplier), >= 2 while generating, else 0
			float  delayMs = 0.0f;             // D, the total delay (base + the buffer)
			bool   cappedAtRefresh = true;     // the output is capped at the refresh (Pause at refresh rate, or a backend that cannot go past it)
			Reason reason = Reason::Normal;
			// Each feature separately, for its own settings line.
			bool   fgActive = false;           // frame generation is on and generating (not passing through)
			bool   blurActive = false;         // motion blur is applied
			float  blurWindowMs = 0.0f;        // the last window's length in ms of content time (0 = a point)
			int    blurSamples = 0;            // Inputs::blurSamples while blurActive, else 0
			float  bufferDelayMs = 0.0f;       // the lag-spike buffer's part of delayMs (the live ramped buffer + Inputs::extraDelayNs)
			float  bufferTargetMs = 0.0f;      // what the buffer is ramping toward (0 with the buffer off)
			float  lastSpikeMs = 0.0f;         // the size of the newest spike inside the look-back window (0 = none)
			int    spikesInWindow = 0;         // spikes inside the window that size the buffer
			int    outliersIgnored = 0;        // spikes inside the window larger than maxBufferMs: ignored for sizing
			int    historyFrames = kHistory;   // HistoryDepth(): real frames the host keeps (and UI-protection clean copies)
			// THE STEADY "GENERATING" SIGNAL, for a HUD that must not flicker. fgActive / activeN
			// follow the pacer's instantaneous state; these two say whether frame generation is
			// adding frames:
			//   generating       -- fgActive (the plan is Generate, frame generation on, not a
			//                       pass-through) AND a generated frame (t < 1 from a pair) was
			//                       presented within the last ~1 s (kGeneratingRecentNs). NO
			//                       multiplier gate: any generated frame counts, so a paint-limited
			//                       output at ~1.1x-1.3x does not flip. Never true while passing
			//                       through; a blur window on a real frame is not a generated one.
			//   steadyMultiplier -- presented outputs per real frame over the last ~1 s of status
			//                       windows (kGeneratingWindowNs), 0 until a window has real frames.
			//                       Informational: it does not gate `generating`.
			float  steadyMultiplier = 0.0f;
			bool   generating = false;
		};

		// Is the output capped at the refresh rate under these inputs?
		static bool CapAtRefresh( const Inputs &in )
		{
			return in.pauseAtRefresh || !in.canExceedRefresh;
		}

		// Forget everything (frame generation was switched off/on, or a new
		// session). The next paint is "untracked" and starts clean.
		void Reset()
		{
			*this = Pacer();
		}

		// The real frames cannot take part right now (a fade, a menu on top, no
		// frame at all): the previous real frames must not be used for the next pair.
		void Discontinuity()
		{
			DropHistory( Cause::Other, false );
			m_ulNewestId = 0;
			m_bPending = false;
			if ( m_bPts )
				m_bHaveArrival = false;   // the next PTS is on a new timeline: no interval across a seek
		}

		// VIDEO MODE: a decoded frame, announced in presentation order, with its PTS in
		// nanoseconds (any origin, the media clock in Inputs maps it to V). Frames may be
		// announced AHEAD of the one on screen (up to kVideoLookahead; false = full, announce it
		// again later -- WantsFrame() says); the pacer promotes each when its time comes and
		// reports it in Decision::newestId, where the renderer copies it. A new stream key or
		// FrameFormat is a reset; a PTS that goes backwards is an implicit seek (a reset); an
		// equal PTS is a duplicate and ignored. Mixing with OnArrival resets.
		bool OnFrame( uint64_t ulFrameId, Ns ptsNs, uint64_t ulStreamKey, const FrameFormat &format )
		{
			if ( !m_bPts )
			{
				if ( m_bHaveStream )
					DropHistory( Cause::Stream, true );
				m_bPts = true;
			}
			else if ( m_bHaveStream && ( ulStreamKey != m_ulStreamKey || format != m_Format ) )
			{
				DropHistory( Cause::Stream, true );
				m_bHaveArrival = false;
			}
			m_Est.SetMinSamples( 1 );
			if ( m_bHaveArrival )
			{
				if ( ptsNs == m_ulLastArrival )
					return true;
				if ( ptsNs < m_ulLastArrival )
					DropHistory( Cause::Other, false );
				else if ( m_nPend >= kVideoLookahead )
					return false;
				else
					m_Est.AddInterval( ptsNs - m_ulLastArrival );
			}
			if ( m_nPend >= kVideoLookahead )
				return false;
			m_Pend[ m_nPend++ ] = HistFrame{ ulFrameId, int64_t( ptsNs ) };
			m_ulStreamKey = ulStreamKey;
			m_Format = format;
			m_bHaveStream = true;
			m_ulLastArrival = ptsNs;
			m_bHaveArrival = true;
			m_ulNewestId = ulFrameId;
			m_bPending = true;
			return true;
		}

		// A real frame became the newest usable one. "Arrival" = the earliest point
		// at which it could be displayed: the moment its acquire fence signalled
		// (a Vulkan/Wayland host stamps it there), NOT the time the host's loop
		// noticed it and NOT the present tick that latches it -- those are quantised
		// to the refresh (+-7 ms for 60 fps on 144 Hz), the arrival is not. tNs is
		// that time. ulStreamKey identifies the source (a window, a game); another
		// key, or another size/format, is a reset.
		void OnArrival( uint64_t ulFrameId, Ns tNs, uint64_t ulStreamKey, const FrameFormat &format )
		{
			if ( m_bPts )
			{
				// Back from timestamps to arrival timing: another kind of source.
				DropHistory( Cause::Stream, true );
				m_bPts = false;
			}
			m_Est.SetMinSamples( kMinSamples );
			m_flLastIntervalMs = 0.0;
			if ( m_bHaveStream && ( ulStreamKey != m_ulStreamKey || format != m_Format ) )
			{
				// Another source, or the same one at another size / format.
				DropHistory( Cause::Stream, true );
			}
			else if ( m_bHaveArrival && tNs > m_ulLastArrival + m_ulGapNs )
			{
				// A gap in the game's frames (a stall; the lag-spike buffer still counts it).
				NoteSpike( tNs, tNs - m_ulLastArrival );
				DropHistory( Cause::Gap, false );
			}
			else if ( m_bHaveArrival )
			{
				const Ns ulInterval = tNs >= m_ulLastArrival ? tNs - m_ulLastArrival : 0;
				NoteSpike( tNs, ulInterval );
				m_Est.AddInterval( ulInterval );
				m_Steady.AddInterval( double( ulInterval ) / double( kNsPerMs ) );
				m_flLastIntervalMs = ClampIntervalMs( double( ulInterval ) / double( kNsPerMs ) );
			}

			m_ulStreamKey = ulStreamKey;
			m_Format = format;
			m_bHaveStream = true;
			m_ulLastArrival = tNs;
			m_bHaveArrival = true;
			m_ulNewestId = ulFrameId;
			m_bPending = true;
			m_nArrivalsWin++;
		}

		// A paint is about to happen. `ulNewestId` is the real frame the host would
		// show for it unprocessed (the one OnArrival() last announced).
		// `bPresentTick` says whether it is a present tick that triggered it: on a
		// fixed-refresh display every paint is, and each one may consume one output
		// frame. A paint that is NOT a tick (a UI repaint) only advances when a new
		// real frame is waiting; otherwise it repeats what is being shown.
		//
		// `bSkippable`: nothing but frame generation itself asked for this paint (no
		// UI / cursor / fade repaint), so a tick that is not due a new output may
		// be dropped (Decision::skip) instead of re-presenting the cached image.
		Decision OnPaint( const Inputs &in, uint64_t ulNewestId, bool bPresentTick, bool bSkippable = false )
		{
			if ( !m_bPts && ulNewestId != m_ulNewestId )
			{
				// The newest frame changed behind our back (the source switched with no
				// arrival of its own, or this is the first paint): no interval, no
				// previous frame.
				DropHistory( Cause::Stream, true );
				m_ulNewestId = ulNewestId;
				m_bPending = true;
				m_bHaveArrival = false;
			}

			const bool bFresh = m_bPending;
			m_bPending = false;

			Evaluate( in );
			if ( m_bPts && m_Plan.state == State::Pass )
			{
				// Video: a pass-through still has to SCHEDULE (the newest announced frame is
				// not the due one): real-rate planning, never inert.
				m_Plan.state = State::Generate;
				m_Plan.realRate = true;
				m_Plan.inert = false;
				m_Plan.oMs = 0.0;
			}

			Decision d;
			d.newestId = ulNewestId;
			d.showId = ulNewestId;
			d.reset = m_bNeedReset;
			if ( d.reset )
				m_bNeedReset = false;

			switch ( m_Plan.state )
			{
				case State::Pass:
					ClearPlayState();
					d.t = 1.0f;
					d.inert = m_Plan.inert;
					d.newOutput = bFresh;
					m_flDelayMs = 0.0;
					break;

				case State::Warming:
					if ( m_bPts )
					{
						PaintVideo( in, d, bPresentTick, bSkippable );
						break;
					}
					// Ask for the ring copy so the first pair after warm-up has its
					// previous frame.
					PushNewest( ulNewestId );
					d.t = 1.0f;
					d.newOutput = bFresh;
					m_flDelayMs = 0.0;
					break;

				case State::Generate:
					if ( m_bPts )
					{
						PaintVideo( in, d, bPresentTick, bSkippable );
						break;
					}
					PaintGenerated( in, d, bFresh, bPresentTick, bSkippable );
					if ( d.skip )
						m_bPending = bFresh;
					break;
			}

			if ( d.newOutput )
			{
				m_nOutputs++;
				if ( d.t < 1.0f && d.prevId != 0 )
				{
					m_ulLastSynthAt = in.now;      // a generated frame (a blur window on a real frame has t = 1)
					m_bHaveSynth = true;
				}
			}
			if ( m_Plan.state == State::Generate )
				m_Last = d;
			return d;
		}

		bool StatusDue( Ns ulNow ) const
		{
			return !m_bStatusStarted || ulNow - m_ulStatusAt >= kStatusPeriodNs;
		}

		// Resets the presented-frames window, so call it only when it is going
		// to be published.
		Report TakeStatus( const Inputs &in, Ns ulNow )
		{
			Report s;
			s.bufferTargetMs = float( m_flTargetMs );
			s.lastSpikeMs = float( m_flLagShownMs );
			s.spikesInWindow = m_nLagInWindow;
			s.outliersIgnored = m_nLagOutliers;
			s.historyFrames = m_nDepth;
			s.chosenN = in.mode == Mode::Fixed ? in.multiplier : 0;
			s.cappedAtRefresh = CapAtRefresh( in );

			const bool bEst = m_Est.Valid();
			if ( bEst )
				s.gameFps = float( 1000.0 / m_Est.IntervalMs() );

			if ( m_bStatusStarted && ulNow > m_ulStatusAt )
				s.presentedFps = float( double( m_nOutputs ) * 1e9 / double( ulNow - m_ulStatusAt ) );
			if ( m_bStatusStarted )
			{
				if ( m_nWin == kStatusWindows )
				{
					for ( int i = 1; i < m_nWin; i++ )
						m_Win[ i - 1 ] = m_Win[ i ];
					m_nWin--;
				}
				m_Win[ m_nWin++ ] = StatusWin{ ulNow, m_nOutputs, m_nArrivalsWin };
			}
			{
				uint64_t ulOut = 0, ulArr = 0;
				for ( int i = 0; i < m_nWin; i++ )
				{
					if ( m_Win[ i ].end + kGeneratingWindowNs >= ulNow )
					{
						ulOut += m_Win[ i ].outputs;
						ulArr += m_Win[ i ].arrivals;
					}
				}
				if ( ulArr > 0 )
					s.steadyMultiplier = float( double( ulOut ) / double( ulArr ) );
			}
			m_nOutputs = 0;
			m_nArrivalsWin = 0;
			m_ulStatusAt = ulNow;
			m_bStatusStarted = true;

			if ( in.mode == Mode::Target )
			{
				const double flRefresh = std::max( in.refreshHz, 1.0 );
				s.targetFps = float( in.targetFps <= 0 ? flRefresh
					: ( s.cappedAtRefresh ? std::min( double( in.targetFps ), flRefresh ) : double( in.targetFps ) ) );
			}

			if ( !in.rendererOk )
			{
				s.reason = Reason::RendererUnavailable;
			}
			else if ( !m_bPts && m_bHaveArrival && ulNow > m_ulLastArrival + m_ulGapNs )
			{
				s.reason = Reason::GameStalled;
			}
			else if ( !bEst || m_Plan.state == State::Warming )
			{
				s.reason = ( m_Cause == Cause::Gap ) ? Reason::GameStalled : Reason::WarmingUp;
			}
			else
			{
				s.reason = ( m_bPts && m_bWaiting ) ? Reason::WaitingForDecoder : m_Plan.reason;
				if ( m_Plan.state == State::Generate )
				{
					s.delayMs = float( m_flDelayMs );
					s.bufferDelayMs = float( double( m_ulExtraNs ) / double( kNsPerMs ) );
					s.blurActive = BlurActive( in );
					s.blurWindowMs = float( m_flBlurMs );
					s.blurSamples = s.blurActive ? in.blurSamples : 0;
					if ( s.gameFps > 0.0f )
						s.effectiveMultiplier = s.presentedFps / s.gameFps;
					if ( !m_Plan.realRate )
					{
						s.fgActive = true;
						if ( s.gameFps > 0.0f )
							s.activeN = std::max( 2, int( std::lround( s.effectiveMultiplier ) ) );
						s.generating = m_bHaveSynth && ulNow >= m_ulLastSynthAt && ulNow - m_ulLastSynthAt <= kGeneratingRecentNs;
					}
				}
			}
			return s;
		}

		// --- read-only views, for tests and the host ---
		bool   EstimateValid() const { return m_Est.Valid(); }
		double IntervalMs() const { return m_Est.IntervalMs(); }
		bool   Generating() const { return m_Plan.state == State::Generate && !m_Plan.realRate; }
		// The planner is producing outputs (frame generation, or one per real frame for blur / the
		// buffer) rather than passing real frames through.
		bool   Planning() const { return m_Plan.state == State::Generate; }
		// One output per real frame (frame generation off, or the game reaches the output rate).
		bool   RealRate() const { return m_Plan.state == State::Generate && m_Plan.realRate; }
		// Real frames the host must keep addressable by id (and, with UI protection, the clean
		// copies to keep: Settings::uiHistory): kHistory, more with a delay; follows the
		// delay, see kHistoryMargin.
		int    HistoryDepth() const { return m_nDepth; }
		// Video mode: OnFrame was the last source, the pacer has room for another announced
		// frame, how many are held ahead, and whether the last paint held a frame for want of
		// the next one (Reason::WaitingForDecoder).
		bool   Timestamps() const { return m_bPts; }
		bool   WantsFrame() const { return m_nPend < kVideoLookahead; }
		int    PendingFrames() const { return m_nPend; }
		bool   WaitingForDecoder() const { return m_bPts && m_bWaiting; }
		bool   CostBlocked() const { return m_bCostBlocked; }
		// The effective output interval (after the cost guard), ms; 0 when not generating.
		double OutputMs() const { return m_Plan.state == State::Generate ? m_Plan.oMs : 0.0; }
		// The output interval is shorter than a tick and the display can show it
		// (pauseAtRefresh off): the host paints on its own timer, one tick per
		// output frame (V = that timer's target), instead of on the present-tick
		// timer. False while not generating.
		bool   ExtraTimer() const { return m_bExtra; }
		double DelayMs() const { return m_flDelayMs; }
		int    HistoryCount() const { return m_nHist; }
		// The lag-spike buffer: its live (ramped) size and its target, ms.
		double BufferMs() const { return m_flBufMs; }
		double BufferTargetMs() const { return m_flTargetMs; }
		// The content time of the last output frame (tau, ns; 0 before any).
		int64_t ContentNs() const { return m_bTauValid ? m_nTauLast : 0; }

	private:
		enum class Cause : uint8_t { Unknown, Stream, Gap, Other };
		enum class State : uint8_t { Warming, Pass, Generate };

		struct Plan
		{
			State  state = State::Warming;
			Reason reason = Reason::WarmingUp;
			bool   inert = false;   // Pass only: the renderer stays completely inert
			double oMs = 0.0;       // Generate: the effective output interval (0 in real-rate mode)
			// Generate only: one output per REAL frame, at its own time (plus the buffer's
			// delay), with the gaps between real frames filled by synths. That is the
			// planner's cadence when frame generation is off, or while the game reaches the
			// output rate (nothing to generate) and blur / the buffer still want a pair.
			bool   realRate = false;
		};

		struct HistFrame
		{
			uint64_t id = 0;
			int64_t  arrival = 0;   // ns
		};

		// Forget the previous real frames and the interval history. The
		// renderer is told through Decision::reset at the next paint.
		void DropHistory( Cause cause, bool bNewGame )
		{
			m_Est.Clear();
			m_Steady.Clear();
			m_bFlipPend = false;
			m_flLastIntervalMs = 0.0;
			ClearPlayState();
			m_bNeedReset = true;
			m_Cause = cause;
			if ( bNewGame )
			{
				// Another game / window: its rate and its cost are unrelated to the
				// last one's.
				m_bGenInit = false;
				m_bCostBlocked = false;
				m_bProbing = false;
				ClearLag();
			}
		}

		// ---- the lag-spike buffer ----

		struct Spike
		{
			Ns     t = 0;       // arrival time of the late frame
			double ms = 0.0;    // size: interval - smoothed interval
		};

		void ClearLag()
		{
			m_nSpikes = 0;
			m_nSpikesTotal = 0;
			m_flSpikeEwma = 0.0;
			m_flBufMs = 0.0;
			m_flTargetMs = 0.0;
			m_flLowMs = 0.0;
			m_flStarveMs = 0.0;
			m_ulExtraNs = 0;
			m_ulLagLastAt = 0;
			m_flLagShownMs = 0.0;
			m_nLagInWindow = m_nLagOutliers = 0;
		}

		static double SpikeWeight( double flAgeNs, double flLookbackNs )
		{
			if ( flAgeNs > flLookbackNs )
				return 0.0;
			return kLagWeightMin + ( 1.0 - kLagWeightMin ) * 0.5 * ( 1.0 + std::cos( 3.14159265358979323846 * flAgeNs / flLookbackNs ) );
		}

		// Called for every real interval BEFORE it enters the estimator: a spike is an
		// interval above kLagSpikeRatio x the smoothed one; its size is the excess.
		void NoteSpike( Ns tNs, Ns ulIntervalNs )
		{
			if ( !m_Est.Valid() )
				return;
			const double flSmooth = m_Est.IntervalMs();
			const double flMs = double( ulIntervalNs ) / double( kNsPerMs );
			const double flSize = flMs - flSmooth;
			if ( !( flMs > kLagSpikeRatio * flSmooth ) || flSize < kLagSpikeMinMs )
				return;
			m_ulLagLastAt = tNs;
			if ( flSize <= m_flLagMaxMs )
			{
				// STALL CREDIT: the display already stood still for this stall, so the delay may
				// grow by what it was short -- at once -- and the content time resumes from where
				// it stopped instead of freezing again on the next one. A spike above the max is
				// an outlier and earns none.
				if ( flSize > m_flDelayMs )
					m_flStarveMs = std::max( m_flStarveMs, flSize - m_flDelayMs );
				m_flSpikeEwma = m_nSpikesTotal == 0 ? flSize : m_flSpikeEwma + kLagEwmaAlpha * ( flSize - m_flSpikeEwma );
				m_nSpikesTotal++;
			}
			Spike sp;
			sp.t = tNs;
			sp.ms = flSize;
			if ( m_nSpikes < kLagSpikeMax )
			{
				m_Spikes[ m_nSpikes++ ] = sp;
				return;
			}
			// Full: the entry that contributes least right now (smallest size x weight) goes.
			int nWorst = 0;
			double flWorst = 1e300;
			for ( int i = 0; i < m_nSpikes; i++ )
			{
				const double flAge = tNs > m_Spikes[ i ].t ? double( tNs - m_Spikes[ i ].t ) : 0.0;
				const double flV = m_Spikes[ i ].ms * SpikeWeight( flAge, m_flLagLookbackNs );
				if ( flV < flWorst )
				{
					flWorst = flV;
					nWorst = i;
				}
			}
			m_Spikes[ nWorst ] = sp;
		}

		// The buffer's target for the settings, ms (also fills the window counters).
		double LagTargetMs( const Inputs &in, double flIntervalMs, Ns ulNow )
		{
			const LagBufferSettings &lb = in.lagBuffer;
			// What kHistoryMax frames can hold at this game rate.
			const double flCap = std::min( m_flLagMaxMs, double( kHistoryMax - kHistory - kHistoryMargin ) * flIntervalMs );
			m_nLagInWindow = m_nLagOutliers = 0;
			double flBase = 0.0;
			m_flLagShownMs = 0.0;
			int n = 0;
			for ( int i = 0; i < m_nSpikes; i++ )
			{
				const double flAge = ulNow > m_Spikes[ i ].t ? double( ulNow - m_Spikes[ i ].t ) : 0.0;
				if ( flAge > m_flLagLookbackNs )
					continue;                                  // expired
				m_Spikes[ n++ ] = m_Spikes[ i ];
				if ( m_Spikes[ i ].t == m_ulLagLastAt )
					m_flLagShownMs = m_Spikes[ i ].ms;
				if ( m_Spikes[ i ].ms > m_flLagMaxMs )
				{
					m_nLagOutliers++;                          // an outlier: never sized for
					continue;
				}
				m_nLagInWindow++;
				flBase = std::max( flBase, m_Spikes[ i ].ms * SpikeWeight( flAge, m_flLagLookbackNs ) );
			}
			m_nSpikes = n;
			if ( m_nLagInWindow >= 1 && m_nSpikesTotal >= 2 )
				flBase = std::max( flBase, kLagFloorFrac * m_flSpikeEwma );
			if ( lb.testMode == LagTestMode::ForceMax )
				return flCap;
			if ( lb.testMode == LagTestMode::ForceMin )
				return 0.0;
			return flBase > 0.0 ? std::min( flBase * ( 1.0 + kLagMarginFrac ) + kLagMarginMs, flCap ) : 0.0;
		}

		// Once per paint, before the plan: sizes the target, ramps D_buf toward it.
		void UpdateBuffer( const Inputs &in, int64_t V )
		{
			double flDtMs = 0.0;
			if ( m_bRampV && V > m_nRampV )
				flDtMs = std::min( double( V - m_nRampV ) / 1e6, kLagRampMaxDtMs );
			m_nRampV = V;
			m_bRampV = true;
			m_flDtMs = flDtMs;

			const LagBufferSettings &lb = in.lagBuffer;
			// Switched off: drop the buffer at once (one forward step of tau; tau stays
			// monotonic). Why: the user: "turning it off should immediately remove the buffer".
			if ( !lb.enabled )
			{
				// A paint that repeats the last output (cadence) does not recompute D, so
				// take the buffer off the reported delay now too.
				m_flDelayMs = std::max( 0.0, m_flDelayMs - m_flBufMs );
				m_flBufMs = 0.0;
				m_flLowMs = 0.0;
				m_flStarveMs = 0.0;
				m_flTargetMs = 0.0;
				m_ulGapNs = kGapResetNs;
				m_ulExtraNs = in.extraDelayNs;
				return;
			}
			auto Fin = []( double v, double lo, double hi ) { return v != v ? lo : std::min( std::max( v, lo ), hi ); };
			m_flLagMaxMs = Fin( lb.maxBufferMs, 0.0, kLagMaxBufferMs );
			m_flLagLookbackNs = Fin( lb.lookbackSec, 1.0, kLagMaxLookbackSec ) * 1e9;
			// A stall must not reset the pipeline before a spike the buffer is sized for.
			m_ulGapNs = lb.enabled ? std::max( kGapResetNs, Ns( m_flLagMaxMs * 1e6 ) + 50 * kNsPerMs ) : kGapResetNs;

			// No measured game interval yet (a cold start, or just after a stream change): the
			// cap (kHistoryMax frames at this rate), the history depth and the content timing
			// all need one, so hold the buffer where it is instead of sizing it from a guess.
			// Why: a ForceMax buffer was sized against the (then) 4 ms interval clamp and asked for
			// ~27 frames for the first paints (the depth then took two shrink dwells to settle).
			if ( !m_Est.Valid() )
			{
				m_flTargetMs = m_flBufMs;
				m_flLowMs = 0.0;
				m_ulExtraNs = in.extraDelayNs + Ns( m_flBufMs * 1e6 );
				return;
			}
			const double flInterval = m_Est.IntervalMs();
			m_flTargetMs = lb.enabled ? LagTargetMs( in, flInterval, in.now ) : 0.0;

			const double flStep = kLagRampEps * flDtMs;
			if ( m_flTargetMs > m_flBufMs )
			{
				// Growth is the ramp's step, plus the stall credit when a spike just starved the
				// display (NoteSpike), never past the target (which the history cap bounds).
				m_flBufMs = std::min( m_flBufMs + flStep + m_flStarveMs, m_flTargetMs );
				m_flLowMs = 0.0;
			}
			else if ( m_flBufMs > m_flTargetMs )
			{
				const bool bNow = !lb.enabled || lb.testMode != LagTestMode::Off;
				const bool bBelow = m_flTargetMs <= 0.0 ? m_flBufMs > 0.0 : m_flBufMs > m_flTargetMs + kLagShrinkBandMs;
				if ( bNow )
				{
					m_flBufMs = std::max( m_flBufMs - flStep, m_flTargetMs );
				}
				else if ( bBelow )
				{
					m_flLowMs += flDtMs;
					if ( m_flLowMs >= kLagShrinkDwellMs )
					{
						const double flGoal = m_flTargetMs > 0.0 ? m_flTargetMs + 0.5 * kLagShrinkBandMs : 0.0;
						m_flBufMs = std::max( m_flBufMs - flStep, flGoal );
					}
				}
				else
				{
					m_flLowMs = 0.0;
				}
			}
			else
			{
				m_flLowMs = 0.0;
			}
			m_flStarveMs = 0.0;
			m_ulExtraNs = in.extraDelayNs + Ns( m_flBufMs * 1e6 );
		}

		// Real frames to keep: kHistory with no delay, else what the delay reaches back (see
		// kHistoryMargin). Grows at once and AHEAD of the ramp (it counts the target), shrinks
		// after kDepthShrinkDwellMs and only past frames the content time has left.
		void UpdateDepth( const Inputs &in )
		{
			// Never size the depth from the buffer delay before the interval is measured
			// (dividing by the interval clamp balloons it); keep what it is (kHistory at start).
			if ( !m_Est.Valid() )
				return;
			const double flInterval = std::max( m_Est.IntervalMs(), kMinIntervalMs );
			const double flExtraMs = double( in.extraDelayNs ) / 1e6 + std::max( m_flBufMs, m_flTargetMs );
			int nWant = kHistory;
			if ( flExtraMs > 0.0 )
			{
				const double flBaseMs = ( m_Plan.state == State::Generate && !m_Plan.realRate ) ? DelayBaseFor( in ) : 0.0;
				nWant = std::min( kHistoryMax, std::max( kHistory, int( std::ceil( ( flBaseMs + flExtraMs ) / flInterval - 1e-9 ) ) + kHistoryMargin ) );
			}
			if ( nWant >= m_nDepth )
			{
				m_nDepth = nWant;
				m_flDepthLowMs = 0.0;
				return;
			}
			m_flDepthLowMs += m_flDtMs;
			if ( m_flDepthLowMs < kDepthShrinkDwellMs )
				return;
			if ( m_nHist > nWant && m_bTauValid && m_Hist[ m_nHist - nWant ].arrival > m_nTauLast )
				return;     // the content time is still inside the frames that would go
			m_nDepth = nWant;
			m_flDepthLowMs = 0.0;
		}

		// Everything about the output sequence in progress (the content time, the
		// cadence phase, the frames kept).
		void ClearPlayState()
		{
			m_nHist = 0;
			m_bTauValid = false;
			m_nNextDue = 0;
			m_bExpectPaint = false;
			m_bHavePair = false;
			m_nPairSynths = 0;
			m_flBlurMs = 0.0;
			m_Last = Decision();
			m_nPend = 0;
			m_bWaiting = false;
		}

		void PushNewest( uint64_t ulId )
		{
			if ( m_nHist > 0 && m_Hist[ m_nHist - 1 ].id == ulId )
				return;
			while ( m_nHist >= m_nDepth )
			{
				for ( int i = 1; i < m_nHist; i++ )
					m_Hist[ i - 1 ] = m_Hist[ i ];
				m_nHist--;
			}
			m_Hist[ m_nHist++ ] = HistFrame{ ulId, int64_t( m_ulLastArrival ) };
		}

		// Chooses what this paint does in general: pass-through (and why), warm-up,
		// or generation (and at what output interval).
		void Evaluate( const Inputs &in )
		{
			const bool bCap = CapAtRefresh( in );
			const uint32_t uKey = uint32_t( in.mode ) | ( uint32_t( in.multiplier ) << 2 ) |
				( uint32_t( in.targetFps ) << 8 ) | ( uint32_t( in.priority ) << 24 ) | ( uint32_t( bCap ) << 25 ) |
				( uint32_t( in.frameGen ) << 26 );
			if ( uKey != m_uCfgKey )
			{
				// The user picked something else: forget the content time and the
				// hysteresis memory and adopt the new target at once.
				m_uCfgKey = uKey;
				m_bGenInit = false;
				m_bFlipPend = false;
				m_bTauValid = false;
				m_nNextDue = 0;
				m_bExpectPaint = false;
				m_Last = Decision();
			}

			Plan p;
			m_bExtra = false;
			if ( m_bPts )
			{
				// Video: no arrival spikes to size a buffer for and D is 0 on the media clock.
				m_flBufMs = m_flTargetMs = 0.0;
				m_ulExtraNs = 0;
				m_nDepth = kHistory;
				m_flDtMs = 0.0;
			}
			else
			{
				UpdateBuffer( in, int64_t( in.presentNs ? in.presentNs : in.now ) );
				UpdateDepth( in );
			}
			const bool bFG = in.frameGen;
			const bool bBlur = BlurActive( in );
			const bool bDelay = m_ulExtraNs > 0;
			if ( !bFG && !bBlur && !bDelay )
			{
				// Frame generation off and nothing else asked for a pair: not a feature, a
				// pass-through. The host may also simply not call the pacer.
				p.state = State::Pass;
				p.reason = Reason::Off;
				p.inert = true;
				m_Plan = p;
				return;
			}
			if ( !in.rendererOk )
			{
				// Keep telling the renderer (not inert): it only re-checks whether it
				// can run when it is driven, and it would otherwise stay "unavailable"
				// forever after the cause (HDR, ...) went away.
				p.state = State::Pass;
				p.reason = Reason::RendererUnavailable;
				p.inert = false;
				m_Plan = p;
				return;
			}
			if ( !m_Est.Valid() )
			{
				p.state = State::Warming;
				p.reason = Reason::WarmingUp;
				m_Plan = p;
				return;
			}

			const double flInterval = m_Est.IntervalMs();
			const double flBaseMs = OutputIntervalMs( in.mode, in.multiplier, in.targetFps, flInterval, in.refreshHz, bCap );

			// The game reaches the output rate: nothing to generate. Judged on the STEADY rate
			// (video: the exact PTS interval), as a ratio band (kStopRatio / kStartRatio) that must
			// also HOLD for a dwell before the state changes (kStopDwellMs / kStartDwellMs; video:
			// none). The first decision after a start or a change uses the stop ratio alone.
			const double flSteadyMs = m_bPts ? flInterval : m_Steady.IntervalMs();
			const double flRatio = OutputRatio( flSteadyMs,
				OutputIntervalMs( in.mode, in.multiplier, in.targetFps, flSteadyMs, in.refreshHz, bCap ) );
			const bool bWantFlip = m_bGen ? flRatio < kStopRatio : flRatio > kStartRatio;
			if ( !m_bGenInit )
			{
				m_bGen = flRatio > kStopRatio;
				m_bGenInit = true;
				m_bFlipPend = false;
			}
			else if ( !bWantFlip )
			{
				m_bFlipPend = false;
			}
			else if ( m_bPts )
			{
				m_bGen = !m_bGen;
			}
			else
			{
				if ( !m_bFlipPend )
				{
					m_bFlipPend = true;
					m_ulFlipSince = in.now;
				}
				const Ns ulDwell = Ns( ( m_bGen ? kStopDwellMs : kStartDwellMs ) * 1e6 );
				if ( in.now >= m_ulFlipSince + ulDwell )
				{
					m_bGen = !m_bGen;
					m_bFlipPend = false;
				}
			}
			// Real-rate: no generated frames (off, or the game reaches the output rate),
			// but blur and / or the buffer still want a pair for every real frame.
			const bool bRealRate = !bFG || !m_bGen;
			if ( bRealRate && !bBlur && !bDelay )
			{
				p.state = State::Pass;
				p.reason = Reason::GameTooFast;
				p.inert = true;
				m_Plan = p;
				return;
			}
			// What one output costs, in synths: blur averages blurSamples of them.
			const float flSynthMs = ( in.synthMs >= 0.0f && bBlur ) ? in.synthMs * float( std::max( in.blurSamples, 1 ) ) : in.synthMs;

			double flOutMs = flBaseMs;
			Reason eReason = Reason::Normal;

			const bool bTimed = in.estimateMs >= 0.0f && in.synthMs >= 0.0f;
			if ( !bTimed || !in.costGuard )
			{
				m_bCostBlocked = false;
				m_bProbing = false;
			}
			else
			{
				if ( m_bCostBlocked && !m_bProbing && in.now >= m_ulCostBlockedAt + kCostProbeNs )
				{
					m_bProbing = true;
					m_ulProbeStart = in.now;
					m_uProbeSeq = in.costSeq;
				}
				if ( m_bProbing )
				{
					if ( in.costSeq != m_uProbeSeq )
					{
						// A measurement of a pair generated by the probe.
						m_bProbing = false;
						if ( MaxSynthsPerPair( flInterval, in.estimateMs, flSynthMs ) >= 1 )
							m_bCostBlocked = false;
						else
							m_ulCostBlockedAt = in.now;
					}
					else if ( in.now >= m_ulProbeStart + kCostProbeMaxNs )
					{
						// Nothing measured: stay blocked, try again later.
						m_bProbing = false;
						m_ulCostBlockedAt = in.now;
					}
				}

				if ( m_bCostBlocked )
				{
					if ( !m_bProbing )
					{
						p.state = State::Pass;
						p.reason = Reason::CostGuard;
						p.inert = true;
						m_Plan = p;
						return;
					}
					// The probe: one generated frame per real frame.
					flOutMs = std::max( flOutMs, flInterval * 0.5 );
					eReason = Reason::CostGuard;
				}
				else
				{
					const int nMax = MaxSynthsPerPair( flInterval, in.estimateMs, flSynthMs );
					if ( nMax < 1 )
					{
						m_bCostBlocked = true;
						m_ulCostBlockedAt = in.now;
						p.state = State::Pass;
						p.reason = Reason::CostGuard;
						p.inert = true;
						m_Plan = p;
						return;
					}
					const double flCostMs = flInterval / double( nMax + 1 );
					if ( !bRealRate && flCostMs > flOutMs + 1e-9 )
					{
						flOutMs = flCostMs;
						eReason = Reason::CostGuard;
					}
				}
			}

			p.state = State::Generate;
			p.reason = eReason != Reason::Normal ? eReason : ( !bRealRate ? Reason::Normal : ( bFG ? Reason::GameTooFast : Reason::Off ) );
			p.oMs = bRealRate ? 0.0 : flOutMs;
			p.realRate = bRealRate;
			m_Plan = p;

			// Beyond the refresh: only when the cap is off, with a band so an output
			// interval hovering at one tick does not flap between the two timers.
			if ( !bCap && !bRealRate )
			{
				const double flTickMs = 1000.0 / std::max( in.refreshHz, 1.0 );
				m_bExtra = flOutMs < flTickMs * ( m_bExtraPrev ? kExtraLeave : kExtraEnter );
			}
			m_bExtraPrev = m_bExtra;
		}

		static bool BlurActive( const Inputs &in )
		{
			return in.blur && in.blurAmount > 0.0 && in.blurSamples >= 2;
		}

		// The delay D for the current plan, ms: the base delay (Low latency /
		// Smoothness while generating, 0 in real-rate) plus the buffer's.
		double DelayFor( const Inputs &in ) const
		{
			const double flExtraMs = double( m_ulExtraNs ) / double( kNsPerMs );
			if ( m_Plan.realRate )
				return flExtraMs;
			return DelayBaseFor( in ) + flExtraMs;
		}

		double DelayBaseFor( const Inputs &in ) const
		{
			const double flInterval = m_Est.IntervalMs();
			if ( in.priority == Priority::Smoothness )
				return SmoothnessDelayMs( flInterval, m_Est.SpreadMs() );
			const double flI = LowLatencyIntervalMs( flInterval, m_flLastIntervalMs );
			const double flBase = LowLatencyDelayMs( flI, m_Plan.oMs );
			// i - o plus the typical jitter. No floor of one o and no paint lead: minimal
			// delay is the point (see LowLatencyMarginMs). Zero stays zero (game at the output rate).
			return flBase > 0.0 ? flBase + LowLatencyMarginMs( m_Est.JitterP75Ms(), flI ) : 0.0;
		}

		void Repeat( Decision &d, const Decision &last, bool bRepaintNext ) const
		{
			const bool bReset = d.reset;
			const uint64_t ulNewest = d.newestId;
			d = last;
			d.newestId = ulNewest;
			if ( !m_Plan.realRate || last.showId == 0 )
				d.showId = ulNewest;
			d.reset = bReset;
			d.newOutput = false;
			d.repaintNext = bRepaintNext;
		}

		void PaintGenerated( const Inputs &in, Decision &d, bool bFresh, bool bPresentTick, bool bSkippable )
		{
			// A cadence-skip tick, decided before anything is consumed.
			if ( bSkippable && bPresentTick && !d.reset && m_nHist >= 2 && m_nNextDue != 0 && m_Last.t < 1.0f && m_Last.outId != 0 )
			{
				const double flTickMs = 1000.0 / std::max( in.refreshHz, 1.0 );
				const int64_t V = int64_t( in.presentNs ? in.presentNs : in.now );
				if ( m_Plan.oMs > flTickMs * 1.02 && V + int64_t( flTickMs * 1e6 ) / 2 < m_nNextDue )
				{
					Repeat( d, m_Last, true );
					d.skip = true;
					return;
				}
			}

			PushNewest( d.newestId );

			if ( m_nHist < 2 )
			{
				// First frame after a reset / a pass-through: it is only copied;
				// generation starts with the next one.
				d.t = 1.0f;
				d.newOutput = bFresh;
				m_flDelayMs = 0.0;
				return;
			}

			if ( m_Plan.realRate )
			{
				PaintRealRate( in, d, bFresh, bPresentTick, bSkippable );
				return;
			}

			const bool bExpect = bFresh || m_bExpectPaint;
			if ( !bExpect || ( !bPresentTick && !bFresh ) )
			{
				// Nothing new to show (a UI / cursor repaint): the same output again.
				Repeat( d, m_Last, m_bExpectPaint );
				return;
			}

			const int64_t V = int64_t( in.presentNs ? in.presentNs : in.now );
			const double flTickMs = 1000.0 / std::max( in.refreshHz, 1.0 );
			const double flOutMs = m_Plan.oMs;
			const bool bCadence = flOutMs > flTickMs * 1.02;

			// Output cadence: a tick that does not cross the output interval keeps
			// the previous image. A real frame held (t = 1) is not repeatable once
			// the host has moved on, so only a cached synth may be skipped over.
			const bool bLastSynth = m_Last.t < 1.0f && m_Last.outId != 0;
			if ( bCadence )
			{
				const int64_t nTickNs = int64_t( flTickMs * 1e6 );
				const int64_t nOutNs = int64_t( flOutMs * 1e6 );
				const bool bDue = m_nNextDue != 0 && V + nTickNs / 2 >= m_nNextDue;
				if ( m_nNextDue != 0 && !bDue && bLastSynth )
				{
					Repeat( d, m_Last, true );
					return;
				}
				// A due tick advances the accumulator by one output interval, but
				// never lets it lag more than one tick behind (slots that lapsed
				// while holding a real frame are not made up for with a burst). An
				// output that was not due (a new real frame after a held one) starts
				// the phase afresh.
				if ( bDue )
					m_nNextDue = std::max( m_nNextDue + nOutNs, V + nOutNs - nTickNs );
				else
					m_nNextDue = V + nOutNs;
			}
			else
			{
				m_nNextDue = 0;
			}

			// Content time: tau = V - D, never older than the last output's.
			const double flDelayMs = DelayFor( in );
			m_flDelayMs = flDelayMs;
			int64_t tau = V - int64_t( flDelayMs * 1e6 );
			if ( m_bTauValid && tau < m_nTauLast )
				tau = m_nTauLast;
			m_nTauLast = tau;
			m_bTauValid = true;

			const HistFrame &newest = m_Hist[ m_nHist - 1 ];
			d.newOutput = true;

			// Low latency plays only the newest pair (a new real frame snaps the
			// content time to it); Smoothness may still be inside an older one.
			// With the buffer's delay older pairs are played too: that delay is the point.
			const int jMin = ( in.priority == Priority::LowLatency && m_ulExtraNs == 0 ) ? m_nHist - 2 : 0;
			bool bReal = tau >= newest.arrival;
			int j = m_nHist - 2;
			double t = 1.0;
			if ( !bReal )
			{
				while ( j > jMin && m_Hist[ j ].arrival > tau )
					j--;
				const int64_t nSpan = std::max<int64_t>( m_Hist[ j + 1 ].arrival - m_Hist[ j ].arrival, 1 );
				t = double( tau - m_Hist[ j ].arrival ) / double( nSpan );
				if ( j + 1 == m_nHist - 1 && t >= kRealSnapT )
					bReal = true;
				else
					t = std::min( std::max( t, kTMin ), kRealSnapT );
			}

			if ( bReal )
			{
				// The real frame itself: nothing more is due until a newer one arrives.
				FinishReal( in, d, m_nHist - 1 );
				d.repaintNext = false;
				m_bExpectPaint = false;
				m_bHavePair = false;
				return;
			}

			FinishSynth( in, d, j, t, bLastSynth );
			d.repaintNext = true;
			m_bExpectPaint = true;
		}

		// ---- video mode: one output per tick at the PTS that belongs to the tick ----

		void PaintVideo( const Inputs &in, Decision &d, bool bPresentTick, bool bSkippable )
		{
			const int64_t V = int64_t( in.presentNs ? in.presentNs : in.now );
			const double flRate = ( in.playbackRate > 0.0 && std::isfinite( in.playbackRate ) ) ? in.playbackRate : 1.0;
			const int64_t tau = int64_t( std::llround( double( V - in.mediaOffsetNs ) * flRate ) );
			m_flDelayMs = 0.0;
			const bool bSynth = m_Plan.state == State::Generate && !m_Plan.realRate;
			const bool bHaveLast = m_Last.outId != 0;

			auto RepeatLast = [&]( bool bSkip )
			{
				const bool bReset = d.reset;
				const uint64_t ulNewest = d.newestId;
				d = m_Last;
				d.newestId = ulNewest;
				d.reset = bReset;
				d.newOutput = false;
				d.repaintNext = true;
				d.skip = bSkip;
			};

			d.newestId = m_nHist > 0 ? m_Hist[ m_nHist - 1 ].id : 0;
			d.showId = d.newestId;
			d.t = d.t0 = d.t1 = 1.0f;
			if ( !bPresentTick && bHaveLast )
			{
				RepeatLast( false );    // a UI repaint between ticks: nothing new is due
				return;
			}

			// The output cadence (Fixed N / Target below the refresh): a tick that does not
			// cross the output interval keeps the previous image.
			const double flTickMs = 1000.0 / std::max( in.refreshHz, 1.0 );
			const bool bCadence = bSynth && m_Plan.oMs > flTickMs * 1.02;
			if ( bCadence )
			{
				const int64_t nTickNs = int64_t( flTickMs * 1e6 );
				const int64_t nOutNs = int64_t( m_Plan.oMs * 1e6 );
				const bool bDue = m_nNextDue != 0 && V + nTickNs / 2 >= m_nNextDue;
				if ( m_nNextDue != 0 && !bDue && bHaveLast )
				{
					RepeatLast( bSkippable );
					return;
				}
				m_nNextDue = bDue ? std::max( m_nNextDue + nOutNs, V + nOutNs - nTickNs ) : V + nOutNs;
			}
			else
			{
				m_nNextDue = 0;
			}

			// Promote announced frames until the newest known one is past tau (the upper end
			// of the pair that brackets it), or none is left.
			while ( m_nPend > 0 && ( m_nHist == 0 || m_Hist[ m_nHist - 1 ].arrival <= tau ) )
			{
				while ( m_nHist >= m_nDepth )
				{
					for ( int i = 1; i < m_nHist; i++ )
						m_Hist[ i - 1 ] = m_Hist[ i ];
					m_nHist--;
				}
				m_Hist[ m_nHist++ ] = m_Pend[ 0 ];
				for ( int i = 1; i < m_nPend; i++ )
					m_Pend[ i - 1 ] = m_Pend[ i ];
				m_nPend--;
			}
			d.newestId = m_nHist > 0 ? m_Hist[ m_nHist - 1 ].id : 0;
			d.showId = d.newestId;
			if ( m_nHist == 0 )
			{
				m_bWaiting = false;
				return;                 // nothing announced yet
			}

			int j = m_nHist - 1;        // the last frame whose time has come
			while ( j >= 0 && m_Hist[ j ].arrival > tau )
				j--;
			const bool bHaveNext = j >= 0 && j + 1 < m_nHist;
			// A frame whose time has come with none after it: the decoder is late (or the
			// stream ended): hold it, never extrapolate.
			m_bWaiting = j == m_nHist - 1 && tau > m_Hist[ j ].arrival;

			int jReal = std::max( j, 0 );          // before the first frame: show the first
			double t = 1.0;
			bool bReal = true;
			if ( bHaveNext )
			{
				// Within kPtsSnap of a frame's time is that frame (a tick time is rounded to
				// whole ns; the frame itself is not a tick late for it, and no synth that
				// close is worth making).
				const double flSpan = double( std::max<int64_t>( m_Hist[ j + 1 ].arrival - m_Hist[ j ].arrival, 1 ) );
				t = double( tau - m_Hist[ j ].arrival ) / flSpan;
				if ( t >= 1.0 - kPtsSnap )
					jReal = j + 1;
				else if ( t > kPtsSnap && bSynth )
					bReal = false;
			}

			m_nTauLast = tau;
			m_bTauValid = true;
			if ( bReal )
			{
				if ( bHaveLast && m_Last.prevId == 0 && m_Last.t0 == 1.0f && m_Last.showId == m_Hist[ jReal ].id && !BlurActive( in ) )
				{
					// the same real frame again: nothing new to present, and skippable when the
					// ring has nothing new to copy either (the newest announced frame is unchanged)
					RepeatLast( bSkippable && d.newestId == m_Last.newestId );
					return;
				}
				d.newOutput = true;
				FinishReal( in, d, jReal );
			}
			else
			{
				d.newOutput = true;
				FinishSynth( in, d, j, t, m_Last.t < 1.0f && bHaveLast );
			}
			d.repaintNext = true;
		}

		// ---- the shared end of every output: which pair, which window, which image ----

		// The output interval ShownFrame blur is measured in, ns.
		double BlurOutputNs() const
		{
			return ( m_Plan.realRate ? m_Est.IntervalMs() : m_Plan.oMs ) * 1e6;
		}

		// A real frame hist[jShown] is the output. With blur its window is cut from the
		// pair (jShown - 1, jShown), ending at the frame itself.
		void FinishReal( const Inputs &in, Decision &d, int jShown )
		{
			d.t = d.t0 = d.t1 = 1.0f;
			d.prevId = d.currId = 0;
			d.showId = m_Hist[ jShown ].id;
			d.outId = ++m_ulOutSeq;
			m_flBlurMs = 0.0;
			if ( BlurActive( in ) && jShown >= 1 )
			{
				const double flSpan = double( m_Hist[ jShown ].arrival - m_Hist[ jShown - 1 ].arrival );
				const BlurSpan w = BlurWindow( 1.0, BlurOutputNs(), flSpan, in.blurAmount, in.blurRelative );
				if ( w.t0 < w.t1 )
				{
					d.prevId = m_Hist[ jShown - 1 ].id;
					d.currId = m_Hist[ jShown ].id;
					d.t0 = w.t0;
					d.t1 = w.t1;
					m_flBlurMs = double( w.t1 - w.t0 ) * flSpan / 1e6;
				}
			}
		}

		// A synth at t between hist[j] and hist[j + 1], with its blur window.
		void FinishSynth( const Inputs &in, Decision &d, int j, double t, bool bLastSynth )
		{
			d.prevId = m_Hist[ j ].id;
			d.currId = m_Hist[ j + 1 ].id;
			d.t = float( t );

			if ( m_bHavePair && m_PairPrev == d.prevId && m_PairCurr == d.currId )
			{
				m_nPairSynths++;
			}
			else
			{
				m_bHavePair = true;
				m_PairPrev = d.prevId;
				m_PairCurr = d.currId;
				m_nPairSynths = 1;
			}

			// The same instant of the same pair needs no second synth (a clamped
			// content time), and a pair at its synth cap repeats its last one.
			const bool bSamePair = bLastSynth && m_Last.prevId == d.prevId && m_Last.currId == d.currId;
			if ( bSamePair && ( m_nPairSynths > kMaxSynthsPerPair || std::fabs( m_Last.t - d.t ) < 1e-4f ) )
			{
				d.t = m_Last.t;
				d.outId = m_Last.outId;
				m_nPairSynths = std::min( m_nPairSynths, kMaxSynthsPerPair );
			}
			else
			{
				d.outId = ++m_ulOutSeq;
			}

			d.t0 = d.t1 = d.t;
			m_flBlurMs = 0.0;
			if ( BlurActive( in ) )
			{
				const double flSpan = double( m_Hist[ j + 1 ].arrival - m_Hist[ j ].arrival );
				const BlurSpan w = BlurWindow( d.t, BlurOutputNs(), flSpan, in.blurAmount, in.blurRelative );
				d.t0 = w.t0;
				d.t1 = w.t1;
				m_flBlurMs = double( w.t1 - w.t0 ) * flSpan / 1e6;
			}
		}

		// REAL-RATE CADENCE: one output per real frame, at that frame's own time plus
		// the buffer's delay D (the frame whose time has come: the newest one with
		// arrival <= tau = V - D), and a gap between real frames -- a lag spike the
		// delay covers -- is filled by synths at one smoothed game interval spacing from
		// the last output, between the pair that brackets that instant. Without a
		// delay nothing is ever bracketed (every real frame is already due the moment it
		// exists), so the result is exactly one real frame per real frame. A real frame
		// that arrives within the snap tolerance max(2 ms, 15%) after a planned fill
		// instant is waited for (shown as itself) instead of being filled before.
		void PaintRealRate( const Inputs &in, Decision &d, bool bFresh, bool bPresentTick, bool bSkippable )
		{
			if ( !bPresentTick && !bFresh )
			{
				// Nothing new to show (a UI / cursor repaint): the same output again.
				Repeat( d, m_Last, m_bExpectPaint );
				return;
			}

			const int64_t V = int64_t( in.presentNs ? in.presentNs : in.now );
			const double flDelayMs = DelayFor( in );
			m_flDelayMs = flDelayMs;
			const int64_t tauNow = V - int64_t( flDelayMs * 1e6 );
			const double flIntervalMs = m_Est.IntervalMs();
			const int64_t nIntervalNs = int64_t( flIntervalMs * 1e6 );

			int jDue = -1;
			for ( int k = m_nHist - 1; k >= 0; k-- )
			{
				if ( m_Hist[ k ].arrival <= tauNow )
				{
					jDue = k;
					break;
				}
			}

			bool bOut = false;
			bool bReal = false;
			int j = 0;
			double t = 1.0;
			int64_t tau = 0;
			if ( jDue >= 0 && ( !m_bTauValid || m_Hist[ jDue ].arrival > m_nTauLast ) )
			{
				bOut = bReal = true;
				j = jDue;
				tau = m_Hist[ jDue ].arrival;
			}
			else if ( m_bTauValid )
			{
				// Past the planned instant of the next output and no real frame is due:
				// fill, if a later real frame is already known to bracket it.
				const int64_t tauPlan = std::max( m_nTauLast + nIntervalNs, tauNow - nIntervalNs );
				if ( tauNow >= tauPlan )
				{
					for ( int k = m_nHist - 2; k >= 0; k-- )
					{
						if ( m_Hist[ k ].arrival <= tauPlan && tauPlan < m_Hist[ k + 1 ].arrival )
						{
							const int64_t nTol = int64_t( std::max( 2.0, 0.15 * flIntervalMs ) * 1e6 );
							if ( m_Hist[ k + 1 ].arrival - tauPlan > nTol )
							{
								const int64_t nSpan = std::max<int64_t>( m_Hist[ k + 1 ].arrival - m_Hist[ k ].arrival, 1 );
								t = std::min( std::max( double( tauPlan - m_Hist[ k ].arrival ) / double( nSpan ), kTMin ), kRealSnapT );
								bOut = true;
								j = k;
								tau = tauPlan;
							}
							break;
						}
					}
				}
			}

			// Something known is still to be shown (its own time, or a fill before it):
			// ask for the next tick. Nothing known, nothing to wait for: the next
			// arrival wakes the host.
			const bool bPending = m_Hist[ m_nHist - 1 ].arrival > ( bOut ? tau : ( m_bTauValid ? m_nTauLast : INT64_MIN ) );

			if ( !bOut )
			{
				Repeat( d, m_Last, bPending );
				m_bExpectPaint = bPending;
				// Nothing new to show and no new real frame to ingest (its ring copy is
				// the paint a fresh frame needs even when it is not shown yet): a
				// present tick with nothing but a held frame is not worth compositing.
				d.skip = bSkippable && bPresentTick && !bFresh && !d.reset && m_Last.outId != 0;
				return;
			}

			m_nTauLast = tau;
			m_bTauValid = true;
			d.newOutput = true;
			if ( bReal )
			{
				FinishReal( in, d, j );
				m_bHavePair = false;
			}
			else
			{
				FinishSynth( in, d, j, t, m_Last.t < 1.0f && m_Last.outId != 0 );
			}
			d.repaintNext = bPending;
			m_bExpectPaint = bPending;
		}

		// -- arrivals / geometry --
		uint64_t m_ulNewestId = 0;
		bool     m_bPending = false;         // the newest arrival has not been painted yet
		bool     m_bHaveArrival = false;
		Ns       m_ulLastArrival = 0;
		double   m_flLastIntervalMs = 0.0;   // the most recent real interval (0 = none yet)
		bool     m_bHaveStream = false;
		uint64_t m_ulStreamKey = 0;
		FrameFormat m_Format;
		IntervalEstimator m_Est;
		SteadyRate m_Steady;                 // the generate / pass decision's rate (see SteadyRate)

		// -- video mode --
		bool     m_bPts = false;             // the stream is timestamped (OnFrame)
		bool     m_bWaiting = false;         // the last paint held a frame for want of the next
		HistFrame m_Pend[ kVideoLookahead ]; // announced, not yet promoted
		int      m_nPend = 0;

		// -- the output sequence in progress --
		HistFrame m_Hist[ kHistoryMax ];
		int      m_nDepth = kHistory;          // real frames kept: kHistory, more with a delay
		double   m_flDepthLowMs = 0.0;         // how long the wanted depth has been lower than m_nDepth
		double   m_flDtMs = 0.0;               // real time since the previous paint (capped), ms
		int64_t  m_nRampV = 0;
		bool     m_bRampV = false;
		int      m_nHist = 0;
		bool     m_bTauValid = false;
		int64_t  m_nTauLast = 0;
		int64_t  m_nNextDue = 0;             // the cadence accumulator, ns (0 = none)
		bool     m_bExpectPaint = false;     // we asked for the next tick's repaint
		bool     m_bHavePair = false;
		uint64_t m_PairPrev = 0, m_PairCurr = 0;
		int      m_nPairSynths = 0;
		uint64_t m_ulOutSeq = 0;
		bool     m_bNeedReset = false;
		double   m_flDelayMs = 0.0;
		double   m_flBlurMs = 0.0;           // the last output's blur window, ms of content time
		Decision m_Last;
		Cause    m_Cause = Cause::Unknown;

		// -- the plan --
		Plan     m_Plan;
		uint32_t m_uCfgKey = 0xFFFFFFFFu;
		bool     m_bGenInit = false;
		bool     m_bGen = false;
		bool     m_bFlipPend = false;    // the opposite of m_bGen has held since m_ulFlipSince
		Ns       m_ulFlipSince = 0;
		bool     m_bExtra = false;       // ExtraTimer(), as of the last Evaluate
		bool     m_bExtraPrev = false;   // its hysteresis memory (survives pass-through gaps)

		// -- the lag-spike buffer --
		Spike    m_Spikes[ kLagSpikeMax ];
		int      m_nSpikes = 0;
		int      m_nSpikesTotal = 0;           // inlier spikes this session
		double   m_flSpikeEwma = 0.0;          // their long-run mean size, ms
		double   m_flBufMs = 0.0;              // D_buf: the live, ramped buffer
		double   m_flTargetMs = 0.0;
		double   m_flLowMs = 0.0;              // how long D_buf has been above its target by the shrink band
		double   m_flStarveMs = 0.0;           // the stall credit: a spike's size beyond the delay applied, until the next UpdateBuffer
		Ns       m_ulExtraNs = 0;              // the manual extra delay + D_buf, as of the last Evaluate
		Ns       m_ulGapNs = kGapResetNs;      // a gap longer than this between real frames drops the history
		double   m_flLagMaxMs = 50.0;          // the settings last read
		double   m_flLagLookbackNs = 300e9;
		Ns       m_ulLagLastAt = 0;
		double   m_flLagShownMs = 0.0;         // ... if it is inside the window
		int      m_nLagInWindow = 0, m_nLagOutliers = 0;

		// -- cost guard --
		bool     m_bCostBlocked = false;
		Ns       m_ulCostBlockedAt = 0;
		bool     m_bProbing = false;
		Ns       m_ulProbeStart = 0;
		uint32_t m_uProbeSeq = 0;

		// -- status --
		struct StatusWin
		{
			Ns       end = 0;
			uint64_t outputs = 0;
			uint64_t arrivals = 0;
		};
		StatusWin m_Win[ kStatusWindows ];
		int      m_nWin = 0;
		uint64_t m_nArrivalsWin = 0;
		Ns       m_ulLastSynthAt = 0;
		bool     m_bHaveSynth = false;
		uint64_t m_nOutputs = 0;
		bool     m_bStatusStarted = false;
		Ns       m_ulStatusAt = 0;
	};
}
