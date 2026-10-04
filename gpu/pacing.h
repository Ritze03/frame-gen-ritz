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
//   Low latency : D = smoothed interval - o, floored at 0, plus the largest recent
//     arrival jitter (0 for a steady game, at most 4 ms) plus the lead by which the
//     paint wakes before its tick (at most 4 ms). At a fixed N this is
//     (N-1)/N of a game interval. The latest interval replaces the
//     smoothed one only when it is genuinely early or late (more than
//     max(2 ms, 15%) off): such a frame SNAPS the content time to the new pair
//     (latency-first rule), so uneven motion on really jittery frame times
//     is accepted. Why the smoothed interval and the jitter margin: with D from
//     the latest interval alone, ordinary arrival jitter (+-1 ms on a capped
//     game) moved D from pair to pair, and a late arrival left a tick with no
//     new real frame to show, losing an output a steady game should have had.
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>

namespace framegen::pacing
{
	using Ns = uint64_t;

	constexpr Ns kNsPerMs = 1'000'000ull;

	// The game's frame interval = median of the last kIntervalWindow arrival
	// intervals, clamped to [4, 50] ms (250 .. 20 fps).
	constexpr int    kIntervalWindow  = 8;
	constexpr int    kMinSamples      = 4;      // intervals needed before the estimate is trusted
	constexpr double kMinIntervalMs   = 4.0;
	constexpr double kMaxIntervalMs   = 50.0;
	// A gap longer than this between real frames drops the previous frame.
	constexpr Ns     kGapResetNs      = 100 * kNsPerMs;
	// Generation stops when game interval / output interval falls below
	// kStopRatio (the game reaches the output rate) and starts again once it is
	// back above kStartRatio. 98% .. 91% of the refresh at a fixed multiplier.
	constexpr double kStopRatio       = 1.02;
	constexpr double kStartRatio      = 1.10;
	// ExtraTimer() band: paint on its own timer once the output interval is more
	// than 2% shorter than a tick, back to the tick timer once it is more
	// than 2% longer (the same 2% the cadence in PaintGenerated() uses).
	constexpr double kExtraEnter      = 0.98;
	constexpr double kExtraLeave      = 1.02;
	// No output interval is shorter than this: a fixed 8x of a 250 fps game.
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
	// Status is worth publishing this often.
	constexpr Ns     kStatusPeriodNs  = 250 * kNsPerMs;

	// The largest fixed multiplier a host should offer (Inputs::multiplier is 2..kMaxN).
	constexpr int    kMaxN            = 8;
	// Smoothness: margin = max(kMinMarginMs, spread), at most half an interval.
	constexpr double kMinMarginMs     = 2.0;
	// Generated frames per real pair, well under the library's 32-set ring.
	constexpr int    kMaxSynthsPerPair = 24;
	// A synth is never made closer than this to either real frame; a newest-pair
	// t above kRealSnapT shows the real frame itself.
	constexpr double kTMin            = 0.02;
	constexpr double kRealSnapT       = 0.98;
	// Real frames kept for pair selection (the renderer's ring is the same size).
	constexpr int    kHistory         = 3;

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

	// The largest ORDINARY deviation from the median in v[0..n) (0 for fewer than
	// 3): the jitter an arrival can show, which Low latency's margin covers. A
	// deviation past max(2 ms, 15%) is a hitch and is left out.
	inline double MaxDevOf( const double *v, int n );

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

	inline double MaxDevOf( const double *v, int n )
	{
		if ( n < 3 )
			return 0.0;
		const double flMed = MedianOf( v, std::min( n, kIntervalWindow ) );
		double flMax = 0.0;
		for ( int i = 0; i < std::min( n, kIntervalWindow ); i++ )
		{
			// Only ordinary jitter: a deviation past the snap threshold is a hitch
			// (a real early/late frame), which Low latency snaps to, not pads for.
			const double flDev = std::fabs( v[ i ] - flMed );
			if ( flDev <= std::max( 2.0, 0.15 * flMed ) )
				flMax = std::max( flMax, flDev );
		}
		return flMax;
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

	// Low latency's jitter margin: D is the (interval - o) the content lags by, which
	// puts the real frame of a pair exactly where the NEXT real frame is expected
	// (a steady game: the last output of a pair is the real frame, one tick
	// later the next pair starts). An arrival later than that by even a little
	// leaves nothing to show for that tick and the output is lost. So D carries
	// the measured arrival jitter (the largest deviation from the median interval in the window, no floor)
	// on top: the latency Low latency adds is 'one interval minus o plus the jitter
	// actually seen', so a perfectly steady game pays nothing extra.
	// The paint also wakes BEFORE the tick it is for (a tick timer's draw-time
	// red zone, about 4 ms in a nested window): a real frame that arrives in that
	// lead cannot be used, however close to the tick it came, so the lead counts
	// like jitter. flLeadMs = tick time V minus the time of the paint.
	inline double LowLatencyMarginMs( double flJitterMs, double flLeadMs = 0.0 )
	{
		return std::min( std::max( flJitterMs, 0.0 ) + std::max( flLeadMs, 0.0 ), 8.0 );
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
		bool Valid() const { return m_nCount >= kMinSamples; }

		// Median of what we have, clamped. 0 when there is nothing yet.
		double IntervalMs() const
		{
			if ( m_nCount == 0 )
				return 0.0;
			return ClampIntervalMs( MedianOf( m_flMs, m_nCount ) );
		}

		// Jitter of the recent intervals (SpreadOf), in milliseconds.
		double SpreadMs() const { return SpreadOf( m_flMs, m_nCount ); }
		// The largest deviation from the median in the window, ms.
		double MaxDevMs() const { return MaxDevOf( m_flMs, m_nCount ); }

	private:
		double m_flMs[ kIntervalWindow ] = {};
		int m_nCount = 0;
		int m_nHead = 0;
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
		};

		// What to do for this paint.
		struct Decision
		{
			uint64_t newestId = 0;     // the newest real frame (the renderer copies it into its ring on first sight)
			uint64_t prevId = 0;       // the pair a synth is made from (0 when showing the real frame)
			uint64_t currId = 0;
			uint64_t outId = 0;        // the output frame's identity: the same outId is the same image
			float    t = 1.0f;         // (0,1): synth at t between prevId and currId; 1: the real frame newestId
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
			float  delayMs = 0.0f;             // D
			bool   cappedAtRefresh = true;     // the output is capped at the refresh (Pause at refresh rate, or a backend that cannot go past it)
			Reason reason = Reason::Normal;
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
			m_flLastIntervalMs = 0.0;
			if ( m_bHaveStream && ( ulStreamKey != m_ulStreamKey || format != m_Format ) )
			{
				// Another source, or the same one at another size / format.
				DropHistory( Cause::Stream, true );
			}
			else if ( m_bHaveArrival && tNs > m_ulLastArrival + kGapResetNs )
			{
				// A gap in the game's frames.
				DropHistory( Cause::Gap, false );
			}
			else if ( m_bHaveArrival )
			{
				const Ns ulInterval = tNs >= m_ulLastArrival ? tNs - m_ulLastArrival : 0;
				m_Est.AddInterval( ulInterval );
				m_flLastIntervalMs = ClampIntervalMs( double( ulInterval ) / double( kNsPerMs ) );
			}

			m_ulStreamKey = ulStreamKey;
			m_Format = format;
			m_bHaveStream = true;
			m_ulLastArrival = tNs;
			m_bHaveArrival = true;
			m_ulNewestId = ulFrameId;
			m_bPending = true;
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
			if ( ulNewestId != m_ulNewestId )
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

			Decision d;
			d.newestId = ulNewestId;
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
					// Ask for the ring copy so the first pair after warm-up has its
					// previous frame.
					PushNewest( ulNewestId );
					d.t = 1.0f;
					d.newOutput = bFresh;
					m_flDelayMs = 0.0;
					break;

				case State::Generate:
					PaintGenerated( in, d, bFresh, bPresentTick, bSkippable );
					if ( d.skip )
						m_bPending = bFresh;
					break;
			}

			if ( d.newOutput )
				m_nOutputs++;
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
			s.chosenN = in.mode == Mode::Fixed ? in.multiplier : 0;
			s.cappedAtRefresh = CapAtRefresh( in );

			const bool bEst = m_Est.Valid();
			if ( bEst )
				s.gameFps = float( 1000.0 / m_Est.IntervalMs() );

			if ( m_bStatusStarted && ulNow > m_ulStatusAt )
				s.presentedFps = float( double( m_nOutputs ) * 1e9 / double( ulNow - m_ulStatusAt ) );
			m_nOutputs = 0;
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
			else if ( m_bHaveArrival && ulNow > m_ulLastArrival + kGapResetNs )
			{
				s.reason = Reason::GameStalled;
			}
			else if ( !bEst || m_Plan.state == State::Warming )
			{
				s.reason = ( m_Cause == Cause::Gap ) ? Reason::GameStalled : Reason::WarmingUp;
			}
			else
			{
				s.reason = m_Plan.reason;
				if ( m_Plan.state == State::Generate )
				{
					s.delayMs = float( m_flDelayMs );
					if ( s.gameFps > 0.0f )
					{
						s.effectiveMultiplier = s.presentedFps / s.gameFps;
						s.activeN = std::max( 2, int( std::lround( s.effectiveMultiplier ) ) );
					}
				}
			}
			return s;
		}

		// --- read-only views, for tests and the host ---
		bool   EstimateValid() const { return m_Est.Valid(); }
		double IntervalMs() const { return m_Est.IntervalMs(); }
		bool   Generating() const { return m_Plan.state == State::Generate; }
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
			double oMs = 0.0;       // Generate: the effective output interval
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
			}
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
			m_Last = Decision();
		}

		void PushNewest( uint64_t ulId )
		{
			if ( m_nHist > 0 && m_Hist[ m_nHist - 1 ].id == ulId )
				return;
			if ( m_nHist == kHistory )
			{
				for ( int i = 1; i < kHistory; i++ )
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
				( uint32_t( in.targetFps ) << 8 ) | ( uint32_t( in.priority ) << 24 ) | ( uint32_t( bCap ) << 25 );
			if ( uKey != m_uCfgKey )
			{
				// The user picked something else: forget the content time and the
				// hysteresis memory and adopt the new target at once.
				m_uCfgKey = uKey;
				m_bGenInit = false;
				m_bTauValid = false;
				m_nNextDue = 0;
				m_bExpectPaint = false;
				m_Last = Decision();
			}

			Plan p;
			m_bExtra = false;
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
			const double flRatio = OutputRatio( flInterval, flBaseMs );

			// The game reaches the output rate: nothing to generate. Hysteresis is a
			// ratio band, not a time hold (the first decision after a start or a
			// change uses the stop ratio alone).
			if ( !m_bGenInit )
			{
				m_bGen = flRatio > kStopRatio;
				m_bGenInit = true;
			}
			else if ( m_bGen )
			{
				if ( flRatio < kStopRatio )
					m_bGen = false;
			}
			else if ( flRatio > kStartRatio )
			{
				m_bGen = true;
			}
			if ( !m_bGen )
			{
				p.state = State::Pass;
				p.reason = Reason::GameTooFast;
				p.inert = true;
				m_Plan = p;
				return;
			}

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
						if ( MaxSynthsPerPair( flInterval, in.estimateMs, in.synthMs ) >= 1 )
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
					const int nMax = MaxSynthsPerPair( flInterval, in.estimateMs, in.synthMs );
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
					if ( flCostMs > flOutMs + 1e-9 )
					{
						flOutMs = flCostMs;
						eReason = Reason::CostGuard;
					}
				}
			}

			p.state = State::Generate;
			p.reason = eReason;
			p.oMs = flOutMs;
			m_Plan = p;

			// Beyond the refresh: only when the cap is off, with a band so an output
			// interval hovering at one tick does not flap between the two timers.
			if ( !bCap )
			{
				const double flTickMs = 1000.0 / std::max( in.refreshHz, 1.0 );
				m_bExtra = flOutMs < flTickMs * ( m_bExtraPrev ? kExtraLeave : kExtraEnter );
			}
			m_bExtraPrev = m_bExtra;
		}

		// The delay D for the current plan, ms.
		double DelayFor( const Inputs &in ) const
		{
			const double flInterval = m_Est.IntervalMs();
			if ( in.priority == Priority::Smoothness )
				return SmoothnessDelayMs( flInterval, m_Est.SpreadMs() );
			double flBase = LowLatencyDelayMs( LowLatencyIntervalMs( flInterval, m_flLastIntervalMs ), m_Plan.oMs );
			// At a high ratio (the game just under the output rate: 1.1 .. 2) i - o is
			// less than one output interval and no generated frame would fit between two
			// real ones; the floor of one o makes room for at least one.
			if ( m_Plan.oMs > 0.0 && flInterval / m_Plan.oMs >= kStartRatio )
				flBase = std::max( flBase, m_Plan.oMs );
			const double flLeadMs = in.presentNs > in.now ? double( in.presentNs - in.now ) / double( kNsPerMs ) : 0.0;
			return flBase > 0.0 ? flBase + LowLatencyMarginMs( std::min( m_Est.MaxDevMs(), 4.0 ), std::min( flLeadMs, 4.0 ) ) : 0.0;
		}

		void Repeat( Decision &d, const Decision &last, bool bRepaintNext ) const
		{
			const bool bReset = d.reset;
			const uint64_t ulNewest = d.newestId;
			d = last;
			d.newestId = ulNewest;
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
			const int jMin = in.priority == Priority::LowLatency ? m_nHist - 2 : 0;
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
				d.t = 1.0f;
				d.prevId = d.currId = 0;
				d.outId = ++m_ulOutSeq;
				d.repaintNext = false;
				m_bExpectPaint = false;
				m_bHavePair = false;
				return;
			}

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
			d.repaintNext = true;
			m_bExpectPaint = true;
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

		// -- the output sequence in progress --
		HistFrame m_Hist[ kHistory ];
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
		Decision m_Last;
		Cause    m_Cause = Cause::Unknown;

		// -- the plan --
		Plan     m_Plan;
		uint32_t m_uCfgKey = 0xFFFFFFFFu;
		bool     m_bGenInit = false;
		bool     m_bGen = false;
		bool     m_bExtra = false;       // ExtraTimer(), as of the last Evaluate
		bool     m_bExtraPrev = false;   // its hysteresis memory (survives pass-through gaps)

		// -- cost guard --
		bool     m_bCostBlocked = false;
		Ns       m_ulCostBlockedAt = 0;
		bool     m_bProbing = false;
		Ns       m_ulProbeStart = 0;
		uint32_t m_uProbeSeq = 0;

		// -- status --
		uint64_t m_nOutputs = 0;
		bool     m_bStatusStarted = false;
		Ns       m_ulStatusAt = 0;
	};

	// ------------------------------------------------------------------
	//  Motion-blur window
	// ------------------------------------------------------------------
	//
	// Where, inside the current real pair, the motion-blur pass is to average: the
	// span [t0, t1] of the pair's own timeline (0 = the previous real frame, 1 = the
	// newest), the arguments of the renderer's
	//     recordSynthBlur( cb, prev, curr, out, t0, t1 )
	// (the motion-blur pass of framegen::Interpolator; prev/curr are the pair the
	// output is made from, i.e. Decision::prevId/currId, or, when the real frame
	// itself is being shown (Decision::t == 1), the previous real frame and the
	// newest one). This function only does the arithmetic, so the host's blur logic
	// is the same in every host.
	//
	// The blur amount is a fraction of one interval, like a camera's shutter angle:
	//   Relative::ShownFrame -- of the OUTPUT interval o (the time one displayed
	//                           frame is on screen; "180 degrees" = 0.5);
	//   Relative::GameFrame  -- of the GAME interval (the time between two real
	//                           frames). The same amount blurs the same stretch of
	//                           game motion whatever the output rate is.
	// and the window is that long in time, then turned into a fraction s of the pair
	// (s = width / game interval; the pair spans one game interval):
	//   Anchor::Trailing -- [t - s, t]: the blur looks BACK from the shown instant.
	//                       Blur without interpolation: t = 1 (the real frame), so
	//                       [1 - s, 1], and nothing is delayed to make room for it.
	//   Anchor::Centered -- [t - s/2, t + s/2]: the window straddles the generated
	//                       instant, as it does with interpolation, where the later
	//                       half of the window already exists as the next frame.
	// Both ends are clamped to [0, 1] (the window never leaves the pair, it shortens
	// at an edge instead of moving: a Centered window at t = 1 is [1 - s/2, 1]).
	// amount <= 0 (or an unusable game interval) is a point, t0 == t1 == clamp(t):
	// no blur. t is clamped first; NaN counts as 0.
	enum class BlurRelative : uint8_t { ShownFrame, GameFrame };
	enum class BlurAnchor : uint8_t { Trailing, Centered };

	struct BlurSpan
	{
		float t0 = 0.0f;
		float t1 = 0.0f;
	};

	inline BlurSpan BlurWindow( double t, double outputIntervalNs, double gameIntervalNs, double amount,
		BlurRelative eRelative, BlurAnchor eAnchor )
	{
		auto Clamp01 = []( double v ) { return v != v ? 0.0 : std::min( std::max( v, 0.0 ), 1.0 ); };
		const double flT = Clamp01( t );
		BlurSpan w;
		w.t0 = w.t1 = float( flT );
		if ( !( amount > 0.0 ) || !( gameIntervalNs > 0.0 ) )
			return w;
		const double flInterval = eRelative == BlurRelative::ShownFrame ? outputIntervalNs : gameIntervalNs;
		if ( !( flInterval > 0.0 ) )
			return w;
		const double flWidth = std::min( amount, 1.0 ) * flInterval / gameIntervalNs;   // fraction of the pair
		const double flBack = eAnchor == BlurAnchor::Trailing ? flWidth : 0.5 * flWidth;
		const double flFwd = eAnchor == BlurAnchor::Trailing ? 0.0 : 0.5 * flWidth;
		w.t0 = float( Clamp01( flT - flBack ) );
		w.t1 = float( Clamp01( flT + flFwd ) );
		return w;
	}
}
