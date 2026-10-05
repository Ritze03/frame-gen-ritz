// Unit tests for gpu/pacing.h -- the pure half of frame-generation pacing:
// per-tick fractional pacing (content time tau = V - D, the output interval o,
// Low latency / Smoothness), pass-through, the cost guard, the resets and the
// interval estimator, plus the motion-blur window. No Vulkan, no GPU, no clock:
// every test drives the pacer with times it picks itself. Build target `pacingtest`
// (meson test pacing); the tiny Catch2 stand-in is gpu/minitest.h.
//
// `Driver` below stands in for a host's main loop on a fixed-refresh display: real
// frames arrive at times the test lists, a present tick paints iff a frame arrived
// since the last paint or the last decision asked for a repaint (repaintNext), V is
// the tick's time, and what each paint decided is logged. (Ported unchanged in
// intent from gamescope-ritz's tests/test_framegen_pacing.cpp, where the same
// logic was first written; `Vblank()` here is one present tick.)

#include "minitest.h"

#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "pacing.h"

using namespace framegen::pacing;

namespace
{
	constexpr Ns MS = kNsPerMs;

	struct Painted
	{
		Ns t;
		Pacer::Decision d;
		double bufMs = 0.0;     // the lag-spike buffer after this paint
		int    depth = 0;       // HistoryDepth() after this paint
		double delayMs = 0.0;   // the pacer's total delay D after this paint
		int64_t tau = 0;        // the content time after this paint
		bool   gen = false;     // Pacer::Generating() after this paint
	};

	struct Driver
	{
		Pacer pacer;
		double refreshHz = 125.0;      // 8 ms vblanks: integer-ms arithmetic in most tests
		Mode mode = Mode::Fixed;
		int multiplier = 2;
		int targetFps = 0;
		Priority priority = Priority::LowLatency;
		// The renderer's reported cost (stale values persist); < 0 = no timings.
		float estimateMs = -1.0f;
		float synthMs = -1.0f;
		uint32_t costSeq = 0;
		bool rendererOk = true;
		// "Pause at refresh rate" and whether the backend can show a present between
		// vblanks (the defaults are the capped, vblank-paced behaviour).
		bool pauseAtRefresh = true;
		bool canExceed = false;
		bool costGuard = true;         // "Limit to GPU speed" (the product default is off; the pacing tests exercise the guard, so the Driver turns it on)
		// The main loop lets frame generation skip a vblank that is not due a new
		// output when nothing else (UI, cursor, fade) asked for the paint.
		bool skipVblanks = false;
		// The paint wakes this long BEFORE the vblank it is for (the vblank timer's
		// lead): V is the vblank, `now` is earlier, and an arrival in between is not
		// seen by that paint.
		Ns leadNs = 0;
		bool uiRepaints = false;       // a cursor / overlay repaint on every vblank
		// The three features beyond the cadence: frame generation on/off, the lag-spike
		// buffer's extra delay and motion blur.
		bool frameGen = true;
		Ns extraDelayNs = 0;
		LagBufferSettings lag;         // the lag-spike buffer (off by default)
		bool blur = false;
		int blurSamples = 4;
		double blurAmount = 0.5;
		BlurRelative blurRelative = BlurRelative::ShownFrame;
		uint64_t focusKey = 1;
		FrameFormat layer{ 1920, 1080, 87 };

		Ns now = 0;
		uint64_t vblankIdx = 0;
		uint64_t newestId = 0;
		bool wantPaint = false;
		bool forced = false;
		Ns lastArrival = 0;
		std::vector<Painted> log;

		Ns VblankAt( uint64_t i ) const { return Ns( double( i ) * 1e9 / refreshHz ); }
		Ns NextVblank() const { return VblankAt( vblankIdx ); }

		Pacer::Inputs Inputs( Ns t ) const
		{
			Pacer::Inputs in;
			in.now = t >= leadNs ? t - leadNs : 0;
			in.presentNs = t;
			in.mode = mode;
			in.multiplier = multiplier;
			in.targetFps = targetFps;
			in.priority = priority;
			in.refreshHz = refreshHz;
			in.estimateMs = estimateMs;
			in.synthMs = synthMs;
			in.costSeq = costSeq;
			in.rendererOk = rendererOk;
			in.pauseAtRefresh = pauseAtRefresh;
			in.canExceedRefresh = canExceed;
			in.costGuard = costGuard;
			in.frameGen = frameGen;
			in.extraDelayNs = extraDelayNs;
			in.lagBuffer = lag;
			in.blur = blur;
			in.blurSamples = blurSamples;
			in.blurAmount = blurAmount;
			in.blurRelative = blurRelative;
			return in;
		}

		void Arrive( Ns t )
		{
			newestId++;
			lastArrival = t;
			pacer.OnArrival( newestId, t, focusKey, layer );
			wantPaint = true;
		}

		Pacer::Decision Paint( Ns t, bool bPresentTick = true )
		{
			const Pacer::Decision d = pacer.OnPaint( Inputs( t ), newestId, bPresentTick, skipVblanks && !uiRepaints );
			log.push_back( { t, d, pacer.BufferMs(), pacer.HistoryDepth(), pacer.DelayMs(), pacer.ContentNs(), pacer.Generating() } );
			forced = d.repaintNext;
			return d;
		}

		void Vblank()
		{
			if ( wantPaint || forced || uiRepaints )
			{
				const bool bHadArrival = wantPaint;
				wantPaint = false;
				const Pacer::Decision d = Paint( NextVblank() );
				if ( d.skip )
					wantPaint = bHadArrival;   // hasRepaint is not cleared by a skipped paint
			}
			vblankIdx++;
		}

		// An arrival at the same instant as a vblank goes first (it was ready before
		// the loop looked).
		void RunArrivals( const std::vector<Ns> &arrivals, Ns tEnd )
		{
			size_t i = 0;
			for ( ;; )
			{
				const Ns tA = i < arrivals.size() ? arrivals[ i ] : ~Ns( 0 );
				if ( tA + leadNs <= NextVblank() && tA < tEnd )
				{
					now = tA;
					Arrive( tA );
					i++;
				}
				else if ( NextVblank() < tEnd )
				{
					now = NextVblank() >= leadNs ? NextVblank() - leadNs : 0;
					Vblank();
				}
				else
					break;
			}
			now = tEnd;
		}

		static std::vector<Ns> Periodic( Ns first, double intervalMs, Ns tEnd )
		{
			std::vector<Ns> v;
			for ( uint64_t n = 0;; n++ )
			{
				const Ns t = first + Ns( double( n ) * intervalMs * double( MS ) );
				if ( t >= tEnd )
					break;
				v.push_back( t );
			}
			return v;
		}

		// Game frames every intervalMs from the current time until now + dur.
		void Run( Ns dur, double intervalMs, Ns firstOffset = 3 * MS )
		{
			const Ns tEnd = now + dur;
			RunArrivals( Periodic( now + firstOffset, intervalMs, tEnd ), tEnd );
		}

		// Vblanks only (no new frames).
		void Settle( Ns dur ) { RunArrivals( {}, now + dur ); }

		// A real frame arrives at t (the vblanks before it run first) and the next
		// vblank paints. Returns what that paint decided.
		Pacer::Decision ArriveAndPaint( Ns t )
		{
			while ( NextVblank() < t )
			{
				now = NextVblank();
				Vblank();
			}
			now = t;
			Arrive( t );
			now = NextVblank();
			Vblank();
			return log.back().d;
		}

		int CountOutputs( Ns t0, Ns t1 ) const
		{
			int n = 0;
			for ( const Painted &p : log )
				n += ( p.t >= t0 && p.t < t1 && p.d.newOutput ) ? 1 : 0;
			return n;
		}

		// Paints that reach the display (a skipped vblank commits nothing).
		int CountCommits( Ns t0, Ns t1 ) const
		{
			int n = 0;
			for ( const Painted &p : log )
				n += ( p.t >= t0 && p.t < t1 && !p.d.skip ) ? 1 : 0;
			return n;
		}

		// Distinct output images in [t0, t1).
		int CountDistinct( Ns t0, Ns t1 ) const
		{
			std::set<uint64_t> ids;
			for ( const Painted &p : log )
				if ( p.t >= t0 && p.t < t1 && p.d.newOutput )
					ids.insert( p.d.outId );
			return int( ids.size() );
		}

		Pacer::Report Status() { return pacer.TakeStatus( Inputs( now ), now ); }
	};

	// A warmed-up driver: `dur` of frames at intervalMs.
	void WarmUp( Driver &drv, double intervalMs = 16.0, Ns dur = 1000 * MS, Ns firstOffset = 3 * MS )
	{
		drv.Run( dur, intervalMs, firstOffset );
	}
}

// ---------------------------------------------------------------------------
//  Pure helpers
// ---------------------------------------------------------------------------

TEST_CASE( "pacing: output interval for a fixed multiplier and for a target", "[pacing]" )
{
	// Fixed: interval / N, never faster than the display.
	CHECK( OutputIntervalMs( Mode::Fixed, 2, 0, 16.0, 125.0 ) == 8.0 );
	CHECK( OutputIntervalMs( Mode::Fixed, 4, 0, 16.0, 125.0 ) == 8.0 );       // 4 ms wanted, 8 ms vblank
	CHECK( std::fabs( OutputIntervalMs( Mode::Fixed, 2, 0, 10.0, 144.0 ) - 1000.0 / 144.0 ) < 1e-9 );
	// Target: 1 / min(target, refresh), 0 = the refresh.
	CHECK( OutputIntervalMs( Mode::Target, 2, 100, 16.0, 240.0 ) == 10.0 );
	CHECK( std::fabs( OutputIntervalMs( Mode::Target, 2, 0, 16.0, 240.0 ) - 1000.0 / 240.0 ) < 1e-9 );
	CHECK( std::fabs( OutputIntervalMs( Mode::Target, 2, 500, 16.0, 240.0 ) - 1000.0 / 240.0 ) < 1e-9 );   // capped to the refresh
}

TEST_CASE( "pacing: delay per priority", "[pacing]" )
{
	// Low latency: the latest interval minus the output interval, floored at 0.
	CHECK( LowLatencyDelayMs( 16.0, 8.0 ) == 8.0 );          // (N-1)/N of an interval at 2x
	CHECK( std::fabs( LowLatencyDelayMs( 12.0, 4.0 ) - 8.0 ) < 1e-9 );   // 2/3 at 3x
	CHECK( LowLatencyDelayMs( 6.0, 8.0 ) == 0.0 );
	// Smoothness: the median interval plus max(2 ms, spread), at most half an interval.
	CHECK( SmoothnessDelayMs( 16.0, 0.0 ) == 18.0 );
	CHECK( SmoothnessDelayMs( 16.0, 5.0 ) == 21.0 );
	CHECK( SmoothnessDelayMs( 16.0, 30.0 ) == 24.0 );        // margin capped at 8 ms
	CHECK( SmoothnessDelayMs( 3.0, 0.0 ) == 4.5 );           // 2 ms margin capped at 1.5
}

TEST_CASE( "pacing: the jitter spread ignores one hitch", "[pacing]" )
{
	IntervalEstimator e;
	CHECK( e.SpreadMs() == 0.0 );
	for ( int i = 0; i < 8; i++ )
		e.AddInterval( 16 * MS );
	CHECK( e.SpreadMs() == 0.0 );
	e.AddInterval( 60 * MS );                  // one hitch
	CHECK( e.SpreadMs() == 0.0 );
	e.AddInterval( 20 * MS );
	e.AddInterval( 22 * MS );
	CHECK( e.SpreadMs() > 3.0 );               // a jittery game does show
}

TEST_CASE( "pacing: the GPU budget in synths per pair", "[pacing]" )
{
	// 25% of 16 ms = 4 ms; estimate 1 ms leaves 3 ms.
	CHECK( MaxSynthsPerPair( 16.0, 1.0, 1.0 ) == 3 );
	CHECK( MaxSynthsPerPair( 16.0, 1.0, 1.5 ) == 2 );
	CHECK( MaxSynthsPerPair( 16.0, 3.9, 0.5 ) == 0 );
	CHECK( MaxSynthsPerPair( 16.0, 5.0, 0.5 ) == 0 );
	CHECK( MaxSynthsPerPair( 16.0, -1.0, 0.5 ) == INT_MAX );   // no timings, no guard
	CHECK( MaxSynthsPerPair( 16.0, 1.0, -1.0 ) == INT_MAX );
	CHECK( MaxSynthsPerPair( 50.0, 0.1, 0.01 ) == kMaxSynthsPerPair );
}

TEST_CASE( "pacing: interval is the median of the last 8, clamped to 1..50 ms", "[pacing]" )
{
	IntervalEstimator e;
	CHECK_FALSE( e.Valid() );
	CHECK( e.IntervalMs() == 0.0 );

	for ( int i = 0; i < 3; i++ )
		e.AddInterval( 10 * MS );
	CHECK_FALSE( e.Valid() );          // 3 < kMinSamples
	e.AddInterval( 10 * MS );
	CHECK( e.Valid() );
	CHECK( e.IntervalMs() == 10.0 );

	// An outlier does not move a median.
	e.AddInterval( 90 * MS );
	CHECK( e.IntervalMs() == 10.0 );

	// 10 x4, 90 x3 -> sorted 10 10 10 10 90 90 90 -> 10.
	e.AddInterval( 90 * MS );
	e.AddInterval( 90 * MS );
	CHECK( e.IntervalMs() == 10.0 );
	// Eighth: 10 x4, 90 x4 -> mean(10, 90) = 50.
	e.AddInterval( 90 * MS );
	CHECK( e.Count() == 8 );
	CHECK( e.IntervalMs() == 50.0 );

	// Only the LAST 8 count.
	for ( int i = 0; i < 9; i++ )
		e.AddInterval( 20 * MS );
	CHECK( e.Count() == 8 );
	CHECK( e.IntervalMs() == 20.0 );
}

TEST_CASE( "pacing: the interval estimate is clamped", "[pacing]" )
{
	IntervalEstimator fast;
	for ( int i = 0; i < 8; i++ )
		fast.AddInterval( 1 * MS );       // a batch of commits: ~1 ms apart
	CHECK( fast.IntervalMs() == kMinIntervalMs );

	IntervalEstimator slow;
	for ( int i = 0; i < 8; i++ )
		slow.AddInterval( 80 * MS );      // 12.5 fps (still under the 100 ms gap reset)
	CHECK( slow.IntervalMs() == kMaxIntervalMs );

	CHECK( kMinIntervalMs == 1.0 );
	CHECK( ClampIntervalMs( 0.99 ) == 1.0 );
	CHECK( ClampIntervalMs( 2.94 ) == 2.94 );      // 340 fps is measured, not read as 250
	CHECK( ClampIntervalMs( 3.99 ) == 3.99 );
	CHECK( ClampIntervalMs( 50.01 ) == 50.0 );
	CHECK( ClampIntervalMs( 16.6 ) == 16.6 );

	CHECK( MedianOf( nullptr, 0 ) == 0.0 );
	const double v[ 3 ] = { 3.0, 1.0, 2.0 };
	CHECK( MedianOf( v, 3 ) == 2.0 );
}

TEST_CASE( "pacing: the interval comes from arrival times, not from when vblanks latch them", "[pacing]" )
{
	// 60 fps on 144 Hz: latch times are quantised to 6.94 ms, arrivals are not.
	Driver drv;
	drv.refreshHz = 144.0;
	drv.Run( 1000 * MS, 1000.0 / 60.0, 2 * MS );
	REQUIRE( drv.pacer.EstimateValid() );
	CHECK( std::fabs( drv.pacer.IntervalMs() - 1000.0 / 60.0 ) < 0.001 );

	const Pacer::Report s = drv.Status();
	CHECK( std::fabs( s.gameFps - 60.0f ) < 0.01f );
}

// ---------------------------------------------------------------------------
//  Content time: t for both priorities
// ---------------------------------------------------------------------------

TEST_CASE( "pacing: low latency 2x is synth 0.5 then the real frame", "[pacing]" )
{
	// 125 Hz (8 ms vblank), 62.5 fps (16 ms), arrivals ON the vblank grid:
	// o = 8, D = 16 - 8 = 8, tau = V - 8.
	Driver drv;
	WarmUp( drv, 16.0, 1000 * MS, 0 );
	drv.log.clear();
	drv.Run( 400 * MS, 16.0, 0 );
	// Find an arrival-aligned pair (past the first few, where the run starts).
	int nChecked = 0;
	for ( size_t i = 6; i + 1 < drv.log.size(); i++ )
	{
		const Pacer::Decision &a = drv.log[ i ].d;
		const Pacer::Decision &b = drv.log[ i + 1 ].d;
		if ( a.t < 1.0f && !a.inert && b.t >= 1.0f && drv.log[ i + 1 ].t - drv.log[ i ].t == 8 * MS )
		{
			CHECK( std::fabs( a.t - 0.5f ) < 0.01f );
			CHECK( a.currId == a.newestId );
			CHECK( a.prevId == a.newestId - 1 );
			CHECK( a.repaintNext );
			CHECK( b.newestId == a.newestId );
			CHECK( b.prevId == 0 );
			CHECK_FALSE( b.repaintNext );
			CHECK( b.newOutput );
			nChecked++;
		}
	}
	CHECK( nChecked >= 10 );
}

TEST_CASE( "pacing: low latency 3x is thirds then the real frame", "[pacing]" )
{
	// 250 Hz (4 ms), 12 ms frames, 3x: o = 4, D = 12 - 4 = 8.
	Driver drv;
	drv.refreshHz = 250.0;
	drv.multiplier = 3;
	drv.Run( 1000 * MS, 12.0, 0 );
	drv.log.clear();
	drv.Run( 300 * MS, 12.0, 0 );

	int nChecked = 0;
	for ( size_t i = 0; i + 2 < drv.log.size(); i++ )
	{
		const Pacer::Decision &a = drv.log[ i ].d;
		const Pacer::Decision &b = drv.log[ i + 1 ].d;
		const Pacer::Decision &c = drv.log[ i + 2 ].d;
		if ( a.t < 1.0f && b.t < 1.0f && c.t >= 1.0f && a.newestId == c.newestId )
		{
			CHECK( std::fabs( a.t - 1.0f / 3.0f ) < 0.01f );
			CHECK( std::fabs( b.t - 2.0f / 3.0f ) < 0.01f );
			nChecked++;
		}
	}
	CHECK( nChecked >= 10 );
}

TEST_CASE( "pacing: the content time never goes backwards", "[pacing]" )
{
	for ( int prio = 0; prio < 2; prio++ )
	{
		Driver drv;
		drv.priority = prio ? Priority::Smoothness : Priority::LowLatency;
		INFO( "priority " << prio );
		WarmUp( drv, 16.0, 400 * MS );

		// +-5 ms of jitter on every frame, plus the odd late one.
		std::vector<Ns> arr;
		Ns t = drv.now + 3 * MS;
		for ( int i = 0; i < 120; i++ )
		{
			arr.push_back( t );
			const int64_t jitter = ( ( i * 7919 ) % 11 ) - 5;     // deterministic -5..+5
			t += Ns( ( 16 + jitter + ( i % 17 == 0 ? 9 : 0 ) ) * int64_t( MS ) );
		}

		int64_t last = drv.pacer.ContentNs();
		int nMoved = 0;
		const Ns tEnd = arr.back() + 100 * MS;
		size_t ia = 0;
		while ( true )
		{
			const Ns tA = ia < arr.size() ? arr[ ia ] : ~Ns( 0 );
			if ( tA <= drv.NextVblank() && tA < tEnd )
			{
				drv.now = tA;
				drv.Arrive( tA );
				ia++;
			}
			else if ( drv.NextVblank() < tEnd )
			{
				drv.now = drv.NextVblank();
				const size_t nBefore = drv.log.size();
				drv.Vblank();
				if ( drv.log.size() > nBefore )
				{
					const int64_t c = drv.pacer.ContentNs();
					REQUIRE( c >= last );
					nMoved += c > last ? 1 : 0;
					last = c;
					const Pacer::Decision &d = drv.log.back().d;
					REQUIRE( d.t > 0.0f );
					REQUIRE( d.t <= 1.0f );
				}
			}
			else
				break;
		}
		CHECK( nMoved > 100 );
	}
}

// ---------------------------------------------------------------------------
//  Saturation, cadence, target
// ---------------------------------------------------------------------------

TEST_CASE( "pacing: a fixed multiplier saturates at the refresh and never steps down", "[pacing]" )
{
	// 2x at 100 fps on 144 Hz: 200 wanted, 144 shown. Before the rework this
	// passed through (100 x 2 > 1.2 x 144); now every vblank gets an output.
	Driver drv;
	drv.refreshHz = 144.0;
	drv.multiplier = 2;
	WarmUp( drv, 10.0, 1000 * MS );
	drv.log.clear();
	const Ns t0 = drv.now;
	drv.Run( 2000 * MS, 10.0 );

	for ( const Painted &p : drv.log )
		REQUIRE_FALSE( p.d.inert );
	const int nOut = drv.CountOutputs( t0, t0 + 2000 * MS );
	CHECK( nOut >= 2 * 138 );
	CHECK( nOut <= 2 * 146 );

	const Pacer::Report s = drv.Status();
	CHECK( s.reason == Reason::Normal );
	CHECK( s.activeN >= 2 );
	CHECK( drv.pacer.Generating() );
	CHECK( std::fabs( drv.pacer.OutputMs() - 1000.0 / 144.0 ) < 1e-6 );
}

TEST_CASE( "pacing: no 60% cutoff -- generating keeps going until the game reaches the refresh", "[pacing]" )
{
	// 2x at 86 fps on 144 Hz (60% of refresh) used to pass through.
	Driver drv;
	drv.refreshHz = 144.0;
	WarmUp( drv, 1000.0 / 86.0, 1000 * MS );
	CHECK( drv.pacer.Generating() );
	// ...and at 130 fps (90% of the refresh) too.
	Driver d2;
	d2.refreshHz = 144.0;
	WarmUp( d2, 1000.0 / 130.0, 1000 * MS );
	CHECK( d2.pacer.Generating() );
}

TEST_CASE( "pacing: below the refresh the output rate is exactly N x the game", "[pacing]" )
{
	struct Case { double fps; int n; };
	const Case cases[] = { { 60.0, 2 }, { 40.0, 3 }, { 30.0, 4 }, { 24.0, 5 } };   // all 120 / 120 / 120 / 120 on 144 Hz
	for ( const Case &c : cases )
	{
		Driver drv;
		drv.refreshHz = 144.0;
		drv.multiplier = c.n;
		INFO( "fps " << c.fps << " n " << c.n );
		WarmUp( drv, 1000.0 / c.fps, 1000 * MS );
		const Ns t0 = drv.now;
		drv.Run( 3000 * MS, 1000.0 / c.fps );
		const int nOut = drv.CountOutputs( t0, t0 + 3000 * MS );
		const double expect = c.fps * c.n * 3.0;
		CHECK( std::fabs( double( nOut ) - expect ) <= 4.0 );
	}
}

TEST_CASE( "pacing: target mode follows the latest interval on the very next pair", "[pacing]" )
{
	// Target 240 on a 240 Hz display: 120 fps -> 2 per game frame; the game
	// drops to 100 fps -> 2.4 per game frame at once, no multiplier to re-pick.
	Driver drv;
	drv.refreshHz = 240.0;
	drv.mode = Mode::Target;
	drv.targetFps = 240;
	WarmUp( drv, 1000.0 / 120.0, 1000 * MS );
	{
		const Ns t0 = drv.now;
		drv.Run( 1000 * MS, 1000.0 / 120.0 );
		const int n = drv.CountOutputs( t0, t0 + 1000 * MS );
		CHECK( n >= 236 );
		CHECK( n <= 242 );
	}

	// The drop. At 120 fps every pair got exactly 2 outputs; from the very first
	// pair after the drop some get 3 (2.4 on average) -- no multiplier to re-pick,
	// the count falls out of t.
	const Ns tDrop = drv.now;
	drv.log.clear();
	drv.Run( 100 * MS, 10.0 );
	std::map<uint64_t, int> perFrame;
	for ( const Painted &p : drv.log )
		perFrame[ p.d.newestId ] += p.d.newOutput ? 1 : 0;
	int nThree = 0;
	for ( const auto &kv : perFrame )
		nThree += kv.second >= 3 ? 1 : 0;
	CHECK( nThree >= 2 );
	CHECK( drv.CountOutputs( tDrop, tDrop + 100 * MS ) >= 22 );

	drv.Status();
	const Ns t1 = drv.now;
	drv.Run( 1000 * MS, 10.0 );
	const int n = drv.CountOutputs( t1, t1 + 1000 * MS );
	CHECK( n >= 236 );
	CHECK( n <= 242 );

	const Pacer::Report s = drv.Status();
	CHECK( s.chosenN == 0 );
	CHECK( s.targetFps == 240.0f );
	CHECK( std::fabs( s.gameFps - 100.0f ) < 0.5f );
	CHECK( std::fabs( s.effectiveMultiplier - 2.4f ) < 0.15f );
	CHECK( s.activeN == 2 );
}

TEST_CASE( "pacing: target 0 aims at the display refresh and a target above it is capped", "[pacing]" )
{
	for ( int target : { 0, 500 } )
	{
		Driver drv;
		drv.refreshHz = 144.0;
		drv.mode = Mode::Target;
		drv.targetFps = target;
		WarmUp( drv, 1000.0 / 60.0, 1000 * MS );
		const Ns t0 = drv.now;
		drv.Run( 2000 * MS, 1000.0 / 60.0 );
		const int n = drv.CountOutputs( t0, t0 + 2000 * MS );
		INFO( "target " << target << " n " << n );
		// Low latency shows each real frame as soon as the content time reaches it,
		// so a vblank that falls between that and the next real frame re-presents
		// the same image (not counted): a little under the refresh, never over.
		CHECK( n >= 2 * 125 );
		CHECK( n <= 2 * 146 );
		CHECK( drv.Status().targetFps == 144.0f );
	}

	// Smoothness queues about one game frame, so every vblank has content: the
	// full refresh.
	{
		Driver drv;
		drv.refreshHz = 144.0;
		drv.mode = Mode::Target;
		drv.priority = Priority::Smoothness;
		WarmUp( drv, 1000.0 / 60.0, 1000 * MS );
		const Ns t0 = drv.now;
		drv.Run( 2000 * MS, 1000.0 / 60.0 );
		const int n = drv.CountOutputs( t0, t0 + 2000 * MS );
		INFO( "smoothness n " << n );
		CHECK( n >= 2 * 140 );
		CHECK( n <= 2 * 146 );
	}
}

TEST_CASE( "pacing: an uncapped game just under the refresh still generates", "[pacing]" )
{
	// 200 fps on 240 Hz, target = refresh: ratio 1.2.
	Driver drv;
	drv.refreshHz = 240.0;
	drv.mode = Mode::Target;
	drv.targetFps = 0;
	WarmUp( drv, 5.0, 1000 * MS );
	CHECK( drv.pacer.Generating() );
	CHECK( drv.Status().activeN >= 2 );
}

// ---------------------------------------------------------------------------
//  Pass-through
// ---------------------------------------------------------------------------

TEST_CASE( "pacing: pass-through only when the game reaches the output rate, with a ratio band", "[pacing]" )
{
	// 2x on 60 Hz: the output rate is 60. Interval / output interval is the ratio.
	Driver drv;
	drv.refreshHz = 60.0;
	auto Phase = [&]( double fps ) { drv.Run( 2000 * MS, 1000.0 / fps ); };

	Phase( 50.0 );                                  // ratio 1.2
	CHECK( drv.pacer.Generating() );
	Phase( 58.5 );                                  // 1.026: still above the stop ratio
	CHECK( drv.pacer.Generating() );
	Phase( 59.4 );                                  // 1.0101: the game fills the display
	CHECK_FALSE( drv.pacer.Generating() );
	{
		const Pacer::Report s = drv.Status();
		CHECK( s.reason == Reason::GameTooFast );
		CHECK( s.activeN == 0 );
		CHECK( s.delayMs == 0.0f );
	}
	Phase( 57.0 );                                  // 1.0526: inside the band, stays off
	CHECK_FALSE( drv.pacer.Generating() );
	Phase( 54.0 );                                  // 1.111: above the start ratio
	CHECK( drv.pacer.Generating() );

	// A game above the refresh is plainly passed through.
	Driver d2;
	d2.refreshHz = 60.0;
	WarmUp( d2, 1000.0 / 90.0, 1000 * MS );
	CHECK_FALSE( d2.pacer.Generating() );
}

TEST_CASE( "pacing: settled pass-through shows each real frame at once with the renderer inert", "[pacing]" )
{
	Driver drv;
	drv.refreshHz = 60.0;
	WarmUp( drv, 1000.0 / 70.0, 1000 * MS );
	drv.log.clear();
	drv.Run( 500 * MS, 1000.0 / 70.0 );
	REQUIRE( drv.log.size() > 20 );
	for ( const Painted &p : drv.log )
	{
		CHECK( p.d.inert );
		CHECK( p.d.t >= 1.0f );
		CHECK_FALSE( p.d.repaintNext );
		CHECK( p.d.newOutput );             // every real frame shown counts
	}
	// The status number is the real frame rate, not a paint count.
	const Pacer::Report s = drv.Status();
	CHECK( s.reason == Reason::GameTooFast );
}

TEST_CASE( "pacing: the first real frame after a pass-through is only copied", "[pacing]" )
{
	Driver drv;
	drv.refreshHz = 60.0;
	WarmUp( drv, 1000.0 / 70.0, 1000 * MS );
	REQUIRE_FALSE( drv.pacer.Generating() );
	// The game slows to 40 fps: generating again. The frame that starts it has
	// no previous frame: real, renderer NOT inert (it copies), nothing queued.
	drv.log.clear();
	drv.Run( 1000 * MS, 25.0 );
	REQUIRE( drv.pacer.Generating() );
	bool bSawCopyOnly = false;
	for ( size_t i = 0; i < drv.log.size(); i++ )
	{
		if ( !drv.log[ i ].d.inert && drv.pacer.HistoryCount() >= 1 )
		{
			// the first non-inert decision is a copy-only real frame
			CHECK( drv.log[ i ].d.t >= 1.0f );
			CHECK_FALSE( drv.log[ i ].d.repaintNext );
			bSawCopyOnly = true;
			break;
		}
	}
	CHECK( bSawCopyOnly );
}

TEST_CASE( "pacing: an unavailable renderer passes through but keeps driving it", "[pacing]" )
{
	Driver drv;
	WarmUp( drv, 16.0, 800 * MS );
	REQUIRE( drv.pacer.Generating() );

	drv.rendererOk = false;
	drv.log.clear();
	drv.Run( 400 * MS, 16.0 );
	REQUIRE_FALSE( drv.log.empty() );
	for ( const Painted &p : drv.log )
	{
		CHECK_FALSE( p.d.inert );          // still driven, so the renderer re-checks
		CHECK( p.d.t >= 1.0f );
		CHECK_FALSE( p.d.repaintNext );
	}
	CHECK( drv.Status().reason == Reason::RendererUnavailable );

	// The cause goes away: generation resumes (one real frame to copy first).
	drv.rendererOk = true;
	drv.Run( 800 * MS, 16.0 );
	CHECK( drv.pacer.Generating() );
	CHECK( drv.Status().reason == Reason::Normal );
}

TEST_CASE( "pacing: variable refresh and tearing still generate, painted on timer ticks only", "[pacing]" )
{
	// The main loop under VRR: every iteration counts as a vblank (a commit
	// arrival paints at once), but while the pacer generates only vblank-TIMER
	// ticks may paint -- PaintTick() is the rule steamcompmgr.cpp applies.
	CHECK( PaintTick( true, true, true ) );
	CHECK_FALSE( PaintTick( true, false, true ) );       // VRR arrival iteration while generating
	CHECK( PaintTick( false, false, true ) );            // passing through: as always
	CHECK_FALSE( PaintTick( false, false, false ) );
	CHECK( PaintTick( true, true, false ) );

	// An uncapped 200 fps game on a 240 Hz VRR panel, target = refresh.
	Pacer pacer;
	const double flRefresh = 240.0;
	const Ns vblank = Ns( 1e9 / flRefresh );
	const FrameFormat layer{ 1920, 1080, 87 };
	Ns tNextTick = 0;
	Ns tNextArrival = 3 * MS;
	uint64_t id = 0;
	bool bWant = false, bForced = false;
	int nPaintsOnTicks = 0, nPaintsOffTicks = 0, nGeneratingPaints = 0;
	Pacer::Inputs in;
	in.refreshHz = flRefresh;
	in.mode = Mode::Target;
	in.targetFps = 0;
	for ( int step = 0; step < 4000; step++ )
	{
		const bool bArrival = tNextArrival <= tNextTick;
		const Ns t = bArrival ? tNextArrival : tNextTick;
		in.now = in.presentNs = t;
		if ( bArrival )
		{
			id++;
			pacer.OnArrival( id, t, 1, layer );
			bWant = true;
			tNextArrival += 5 * MS;                       // 200 fps
		}
		else
		{
			tNextTick += vblank;
		}
		// VRR: the loop's `vblank` is true on every iteration.
		const bool bTick = !bArrival;
		const bool bGenBefore = pacer.Generating();
		if ( PaintTick( bGenBefore, bTick, true ) && ( bWant || bForced ) )
		{
			const Pacer::Decision d = pacer.OnPaint( in, id, true );
			bWant = false;
			bForced = d.repaintNext;
			if ( bGenBefore )
			{
				nGeneratingPaints++;
				( bTick ? nPaintsOnTicks : nPaintsOffTicks )++;
			}
		}
	}
	CHECK( pacer.Generating() );
	CHECK( nGeneratingPaints > 400 );
	CHECK( nPaintsOffTicks == 0 );
	CHECK( nPaintsOnTicks == nGeneratingPaints );
}

// ---------------------------------------------------------------------------
//  Low latency vs Smoothness
// ---------------------------------------------------------------------------

TEST_CASE( "pacing: low latency snaps to an early real frame, smoothness plays the pair it was in", "[pacing]" )
{
	for ( int prio = 0; prio < 2; prio++ )
	{
		Driver drv;
		drv.priority = prio ? Priority::Smoothness : Priority::LowLatency;
		WarmUp( drv, 16.0, 1000 * MS, 0 );
		INFO( "priority " << prio );

		// The next frame comes 2 ms early (it is painted at the vblank 2 ms later).
		const Pacer::Decision d = drv.ArriveAndPaint( drv.lastArrival + 14 * MS );
		REQUIRE( d.t < 1.0f );
		if ( prio == 0 )
		{
			// Snap: the new pair is the one played at once.
			CHECK( d.currId == d.newestId );
			CHECK( d.t > 0.3f );
		}
		else
		{
			// Queued: the pair before it is still playing.
			CHECK( d.currId != d.newestId );
			CHECK( d.currId == d.newestId - 1 );
		}
	}
}

TEST_CASE( "pacing: a late real frame holds at t = 1 in low latency, smoothness absorbs it", "[pacing]" )
{
	auto HoldBefore = [&]( Priority prio, int nLateMs )
	{
		Driver drv;
		drv.priority = prio;
		WarmUp( drv, 16.0, 1000 * MS, 0 );
		drv.log.clear();
		// Let the vblanks run up to the late frame, which arrives nLateMs late.
		const Ns tLate = drv.lastArrival + Ns( 16 + nLateMs ) * MS;
		drv.RunArrivals( {}, tLate - 1 );
		bool bHold = false;
		for ( const Painted &p : drv.log )
			bHold = bHold || ( p.d.t >= 1.0f && !p.d.repaintNext && !p.d.inert );
		const Pacer::Decision d = drv.ArriveAndPaint( tLate );
		return std::make_pair( bHold, d );
	};

	// 6 ms late: low latency has already shown the newest real frame and waits.
	{
		const auto r = HoldBefore( Priority::LowLatency, 6 );
		CHECK( r.first );
		CHECK( r.second.t < 1.0f );            // the late frame is then played from its pair
	}
	// Smoothness absorbs it (its queue is about one game frame): no hold...
	{
		const auto r = HoldBefore( Priority::Smoothness, 6 );
		CHECK_FALSE( r.first );
		CHECK( r.second.t < 1.0f );
	}
	// ...until the frame is later than the queue: 12 ms late holds at t = 1 once,
	// and the content then carries on from where it was (no skip back).
	{
		Driver drv;
		drv.priority = Priority::Smoothness;
		WarmUp( drv, 16.0, 1000 * MS, 0 );
		const Ns tLate = drv.lastArrival + 28 * MS;
		drv.RunArrivals( {}, tLate - 1 );
		bool bHold = false;
		for ( const Painted &p : drv.log )
			bHold = bHold || ( p.d.t >= 1.0f && !p.d.repaintNext && !p.d.inert && p.t > drv.lastArrival );
		CHECK( bHold );
		const int64_t before = drv.pacer.ContentNs();
		const Pacer::Decision d = drv.ArriveAndPaint( tLate );
		CHECK( drv.pacer.ContentNs() >= before );
		CHECK( d.t < 1.0f );
	}
}

TEST_CASE( "pacing: the delay in the status is D for each priority", "[pacing]" )
{
	Driver ll;
	WarmUp( ll, 16.0, 1000 * MS );
	CHECK( std::fabs( ll.Status().delayMs - 8.0f ) < 0.01f );          // 16 - 8

	Driver sm;
	sm.priority = Priority::Smoothness;
	WarmUp( sm, 16.0, 1000 * MS );
	CHECK( std::fabs( sm.Status().delayMs - 18.0f ) < 0.01f );         // 16 + 2 ms margin

	Driver t3;
	t3.refreshHz = 250.0;
	t3.multiplier = 3;
	WarmUp( t3, 12.0, 1000 * MS );
	CHECK( std::fabs( t3.Status().delayMs - 8.0f ) < 0.01f );          // 12 - 4: 2/3 of a frame
}

// ---------------------------------------------------------------------------
//  Status
// ---------------------------------------------------------------------------

TEST_CASE( "pacing: presented fps ignores repaints of an already-shown output", "[pacing]" )
{
	Driver clean;
	WarmUp( clean, 16.0, 1000 * MS );
	clean.Status();
	clean.Run( 2000 * MS, 16.0 );
	const float flClean = clean.Status().presentedFps;

	// The same game with a cursor/overlay repaint on every single vblank.
	Driver noisy;
	noisy.uiRepaints = true;
	WarmUp( noisy, 16.0, 1000 * MS );
	noisy.Status();
	noisy.Run( 2000 * MS, 16.0 );
	const float flNoisy = noisy.Status().presentedFps;

	// 62.5 fps x 2.
	CHECK( std::fabs( flClean - 125.0f ) < 4.0f );
	CHECK( std::fabs( flNoisy - flClean ) < 1.0f );

	// A repaint is the same output again: same id, not counted, nothing queued.
	clean.Settle( 100 * MS );
	Pacer::Decision before = clean.log.back().d;
	clean.now = clean.NextVblank();
	const Pacer::Decision rep = clean.Paint( clean.now );
	CHECK_FALSE( rep.newOutput );
	CHECK( rep.outId == before.outId );
	CHECK( rep.t == before.t );
	CHECK( rep.prevId == before.prevId );
	CHECK( rep.currId == before.currId );
}

TEST_CASE( "pacing: status numbers", "[pacing]" )
{
	Driver drv;
	drv.multiplier = 3;
	drv.refreshHz = 250.0;
	WarmUp( drv, 12.0, 1000 * MS );
	drv.Status();
	drv.Run( 1000 * MS, 12.0 );
	const Pacer::Report s = drv.Status();
	CHECK( std::fabs( s.gameFps - 83.33f ) < 0.1f );
	CHECK( std::fabs( s.presentedFps - 250.0f ) < 6.0f );
	CHECK( s.chosenN == 3 );
	CHECK( s.activeN == 3 );
	CHECK( std::fabs( s.effectiveMultiplier - 3.0f ) < 0.1f );
	CHECK( s.targetFps == 0.0f );
	CHECK( s.reason == Reason::Normal );

	// Before there is anything to report.
	Pacer fresh;
	Pacer::Inputs in;
	const Pacer::Report f = fresh.TakeStatus( in, 0 );
	CHECK( f.activeN == 0 );
	CHECK( f.reason == Reason::WarmingUp );
	CHECK( f.gameFps == 0.0f );
}

TEST_CASE( "pacing: changing the mode or the multiplier takes effect at once", "[pacing]" )
{
	Driver drv;
	drv.refreshHz = 250.0;
	drv.multiplier = 2;
	WarmUp( drv, 12.0, 1000 * MS );
	CHECK( std::fabs( drv.pacer.OutputMs() - 6.0 ) < 1e-9 );
	drv.multiplier = 3;
	drv.Run( 100 * MS, 12.0 );
	CHECK( std::fabs( drv.pacer.OutputMs() - 4.0 ) < 1e-9 );
	drv.mode = Mode::Target;
	drv.targetFps = 100;
	drv.Run( 100 * MS, 12.0 );
	CHECK( std::fabs( drv.pacer.OutputMs() - 10.0 ) < 1e-9 );
	CHECK( drv.Status().chosenN == 0 );
}

// ---------------------------------------------------------------------------
//  Synth clamp
// ---------------------------------------------------------------------------

TEST_CASE( "pacing: at most kMaxSynthsPerPair distinct synths per real pair", "[pacing]" )
{
	// 1000 Hz, 20 fps, aiming at 1000 fps: 50 outputs per pair wanted.
	Driver drv;
	drv.refreshHz = 1000.0;
	drv.mode = Mode::Target;
	drv.targetFps = 1000;
	WarmUp( drv, 50.0, 1000 * MS );
	drv.log.clear();
	drv.Run( 1000 * MS, 50.0 );

	std::map<std::pair<uint64_t, uint64_t>, std::set<uint64_t>> perPair;
	for ( const Painted &p : drv.log )
	{
		if ( p.d.t < 1.0f && !p.d.inert )
		{
			REQUIRE( p.d.t > 0.0f );
			perPair[ { p.d.prevId, p.d.currId } ].insert( p.d.outId );
		}
	}
	REQUIRE( perPair.size() >= 10 );
	size_t nMax = 0;
	for ( const auto &kv : perPair )
		nMax = std::max( nMax, kv.second.size() );
	CHECK( nMax == size_t( kMaxSynthsPerPair ) );
}

// ---------------------------------------------------------------------------
//  Cost guard
// ---------------------------------------------------------------------------

TEST_CASE( "pacing: the cost guard lowers the output rate, then passes through", "[pacing]" )
{
	// 4x at 60 fps on 240 Hz: o = 4.17 ms, three generated frames per pair.
	// Budget 25% of 16.67 = 4.17 ms.
	Driver drv;
	drv.refreshHz = 240.0;
	drv.multiplier = 4;
	WarmUp( drv, 1000.0 / 60.0, 1000 * MS );
	REQUIRE( drv.pacer.Generating() );
	CHECK( std::fabs( drv.pacer.OutputMs() - 1000.0 / 240.0 ) < 1e-6 );

	// Estimate 1 ms + 1.5 ms per synth: room for 2 synths -> 3 outputs per pair.
	drv.estimateMs = 1.0f;
	drv.synthMs = 1.5f;
	drv.costSeq++;
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	CHECK( drv.pacer.Generating() );
	CHECK( std::fabs( drv.pacer.OutputMs() - ( 1000.0 / 60.0 ) / 3.0 ) < 1e-6 );
	{
		const Pacer::Report s = drv.Status();
		CHECK( s.reason == Reason::CostGuard );
		CHECK( s.activeN >= 2 );
	}

	// Cheaper: back to the full rate.
	drv.synthMs = 0.5f;
	drv.costSeq++;
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	CHECK( std::fabs( drv.pacer.OutputMs() - 1000.0 / 240.0 ) < 1e-6 );
	CHECK( drv.Status().reason == Reason::Normal );

	// The estimate alone is over budget: not even one generated frame fits.
	drv.estimateMs = 5.0f;
	drv.costSeq++;
	drv.log.clear();
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	CHECK_FALSE( drv.pacer.Generating() );
	CHECK( drv.pacer.CostBlocked() );
	CHECK( drv.log.back().d.inert );
	{
		const Pacer::Report s = drv.Status();
		CHECK( s.reason == Reason::CostGuard );
		CHECK( s.activeN == 0 );
	}
}

TEST_CASE( "pacing: with the GPU limit off an over-budget cost neither caps nor passes through", "[pacing]" )
{
	Driver drv;
	drv.costGuard = true;   // explicit: the guard is on until the switch below
	drv.refreshHz = 240.0;
	drv.multiplier = 4;
	WarmUp( drv, 1000.0 / 60.0, 800 * MS );

	// Over budget for one generated frame: with the limit on this passes through.
	drv.estimateMs = 5.0f;
	drv.synthMs = 0.5f;
	drv.costSeq++;
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	REQUIRE( drv.pacer.CostBlocked() );
	REQUIRE_FALSE( drv.pacer.Generating() );

	// Off: the cap lifts at once and it generates at the full rate, no CostGuard, no probe.
	drv.costGuard = false;
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	CHECK_FALSE( drv.pacer.CostBlocked() );
	CHECK( drv.pacer.Generating() );
	CHECK( std::fabs( drv.pacer.OutputMs() - 1000.0 / 240.0 ) < 1e-6 );
	CHECK( drv.Status().reason == Reason::Normal );
	drv.Run( 12 * 1000 * MS, 1000.0 / 60.0 );
	CHECK( drv.pacer.Generating() );
	CHECK( drv.Status().reason == Reason::Normal );

	// A cost that would only lower the rate (not pass through) leaves it alone too.
	drv.estimateMs = 1.0f;
	drv.synthMs = 1.5f;
	drv.costSeq++;
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	CHECK( std::fabs( drv.pacer.OutputMs() - 1000.0 / 240.0 ) < 1e-6 );
	CHECK( drv.Status().reason == Reason::Normal );

	// Back on: it caps again at once.
	drv.costGuard = true;
	drv.costSeq++;
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	CHECK( std::fabs( drv.pacer.OutputMs() - ( 1000.0 / 60.0 ) / 3.0 ) < 1e-6 );
	CHECK( drv.Status().reason == Reason::CostGuard );
}

TEST_CASE( "pacing: a cost-guard pass-through is retried after a while", "[pacing]" )
{
	Driver drv;
	drv.refreshHz = 240.0;
	drv.multiplier = 4;
	WarmUp( drv, 1000.0 / 60.0, 800 * MS );
	drv.estimateMs = 5.0f;
	drv.synthMs = 0.5f;
	drv.costSeq++;
	drv.Run( 500 * MS, 1000.0 / 60.0 );
	REQUIRE( drv.pacer.CostBlocked() );
	REQUIRE_FALSE( drv.pacer.Generating() );

	// Nothing generates, so nothing is measured: the stale numbers stay and the
	// guard holds for the retry period...
	drv.Run( 8 * 1000 * MS, 1000.0 / 60.0 );
	CHECK_FALSE( drv.pacer.Generating() );

	// ...then a probe generates one frame per pair, to take a new measurement.
	drv.Run( 3 * 1000 * MS, 1000.0 / 60.0 );
	CHECK( drv.pacer.Generating() );
	CHECK( drv.pacer.OutputMs() >= 0.5 * ( 1000.0 / 60.0 ) - 1e-6 );
	CHECK( drv.Status().reason == Reason::CostGuard );

	// The probe's pair measured cheap: the guard lifts for good.
	drv.estimateMs = 0.5f;
	drv.costSeq++;
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	CHECK_FALSE( drv.pacer.CostBlocked() );
	CHECK( std::fabs( drv.pacer.OutputMs() - 1000.0 / 240.0 ) < 1e-6 );
}

TEST_CASE( "pacing: a probe that is still too expensive blocks again", "[pacing]" )
{
	Driver drv;
	drv.refreshHz = 240.0;
	drv.multiplier = 4;
	WarmUp( drv, 1000.0 / 60.0, 800 * MS );
	drv.estimateMs = 5.0f;
	drv.synthMs = 0.5f;
	drv.costSeq++;
	drv.Run( 500 * MS, 1000.0 / 60.0 );
	REQUIRE( drv.pacer.CostBlocked() );
	drv.Run( 10 * 1000 * MS + 200 * MS, 1000.0 / 60.0 );
	REQUIRE( drv.pacer.Generating() );              // probing
	drv.costSeq++;                                  // a fresh measurement: still 5 ms
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	CHECK_FALSE( drv.pacer.Generating() );
	CHECK( drv.pacer.CostBlocked() );
}

TEST_CASE( "pacing: no timings means no cost guard", "[pacing]" )
{
	Driver drv;
	drv.refreshHz = 240.0;
	drv.multiplier = 4;
	WarmUp( drv, 1000.0 / 60.0, 800 * MS );
	drv.estimateMs = 5.0f;
	drv.synthMs = 5.0f;
	drv.costSeq++;
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	REQUIRE_FALSE( drv.pacer.Generating() );

	// The GPU cannot time it (or the renderer forgot): the guard is off.
	drv.estimateMs = -1.0f;
	drv.synthMs = -1.0f;
	drv.Run( 300 * MS, 1000.0 / 60.0 );
	CHECK( drv.pacer.Generating() );
	CHECK_FALSE( drv.pacer.CostBlocked() );
	CHECK( drv.Status().reason == Reason::Normal );
}

// ---------------------------------------------------------------------------
//  D12: resets
// ---------------------------------------------------------------------------

TEST_CASE( "pacing: a gap longer than 100 ms resets", "[pacing]" )
{
	Driver drv;
	WarmUp( drv, 16.0 );
	REQUIRE( drv.pacer.Generating() );

	// 99 ms between real frames is a slow frame, not a gap.
	Pacer::Decision d = drv.ArriveAndPaint( drv.lastArrival + 99 * MS );
	CHECK_FALSE( d.reset );
	CHECK( drv.pacer.EstimateValid() );

	// 101 ms is.
	d = drv.ArriveAndPaint( drv.lastArrival + 101 * MS );
	CHECK( d.reset );                      // fghost::Reset() before this composite
	CHECK( d.t >= 1.0f );                  // the first frame after the gap goes out directly
	CHECK_FALSE( d.inert );                // ...but is copied
	CHECK_FALSE( d.repaintNext );
	CHECK_FALSE( drv.pacer.EstimateValid() );   // history gone
	CHECK( drv.pacer.HistoryCount() == 1 );

	// The status says why, until the estimate has settled again.
	Pacer::Report s = drv.Status();
	CHECK( s.reason == Reason::GameStalled );
	CHECK( s.activeN == 0 );

	// Frames keep coming at 30 fps: real frames only until the interval is known
	// again (kMinSamples intervals), then generation resumes.
	drv.Run( 1500 * MS, 1000.0 / 30.0 );
	CHECK( drv.pacer.EstimateValid() );
	CHECK( drv.pacer.Generating() );
	s = drv.Status();
	CHECK( s.reason == Reason::Normal );
	CHECK( s.activeN >= 2 );

	// Exactly one reset in the whole run since the gap.
	int nReset = 0;
	for ( const Painted &p : drv.log )
		nReset += p.d.reset ? 1 : 0;
	CHECK( nReset == 1 );
}

TEST_CASE( "pacing: a size or format change of layer 0 resets", "[pacing]" )
{
	for ( int what = 0; what < 3; what++ )
	{
		Driver drv;
		WarmUp( drv, 16.0 );
		INFO( "what = " << what );

		if ( what == 0 )
			drv.layer.width = 2560;
		else if ( what == 1 )
			drv.layer.height = 1440;
		else
			drv.layer.format = 88;

		const Pacer::Decision d = drv.ArriveAndPaint( drv.lastArrival + 33 * MS );
		CHECK( d.reset );
		CHECK( d.t >= 1.0f );
		CHECK_FALSE( drv.pacer.EstimateValid() );
		CHECK_FALSE( d.repaintNext );
	}
}

TEST_CASE( "pacing: a focus change resets", "[pacing]" )
{
	// with an arrival of its own
	{
		Driver drv;
		WarmUp( drv, 16.0 );
		drv.focusKey = 2;
		const Pacer::Decision d = drv.ArriveAndPaint( drv.lastArrival + 33 * MS );
		CHECK( d.reset );
		CHECK( d.t >= 1.0f );
		CHECK_FALSE( drv.pacer.EstimateValid() );
	}
	// layer 0 swapped for another window's commit with no arrival at all
	// (determine_and_apply_focus() picks that window's last done commit)
	{
		Driver drv;
		WarmUp( drv, 16.0 );
		drv.newestId += 1000;
		const Pacer::Decision d = drv.Paint( drv.NextVblank() );
		CHECK( d.reset );
		CHECK( d.t >= 1.0f );
		CHECK( d.newestId == drv.newestId );
		CHECK_FALSE( d.inert );            // warm-up frame: the renderer still copies it
	}
	// layer 0 not a game frame (fade, Steam UI): the glue reports a discontinuity
	{
		Driver drv;
		WarmUp( drv, 16.0 );
		drv.pacer.Discontinuity();
		drv.forced = false;                // the glue drops its pending repaint too
		drv.wantPaint = false;
		CHECK( drv.pacer.HistoryCount() == 0 );
		// the game's next frame is the first of a fresh sequence
		const Pacer::Decision d = drv.ArriveAndPaint( drv.lastArrival + 33 * MS );
		CHECK( d.reset );
		CHECK( d.t >= 1.0f );
	}
}

TEST_CASE( "pacing: after a reset the next frame is copied and generation resumes after warm-up", "[pacing]" )
{
	Driver drv;
	WarmUp( drv, 16.0 );
	drv.focusKey = 7;
	drv.layer.width = 1280;
	drv.layer.height = 720;
	drv.log.clear();
	drv.Run( 1000 * MS, 1000.0 / 30.0 );
	REQUIRE( drv.pacer.Generating() );
	REQUIRE_FALSE( drv.log.empty() );
	int nReset = 0;
	for ( const Painted &p : drv.log )
		nReset += p.d.reset ? 1 : 0;
	CHECK( nReset == 1 );
	// Warm-up frames are real and not inert; generated frames come after.
	bool bGen = false;
	for ( const Painted &p : drv.log )
	{
		if ( p.d.t < 1.0f )
			bGen = true;
		else if ( !bGen )
			CHECK_FALSE( p.d.inert );
	}
	CHECK( bGen );
}

TEST_CASE( "pacing: Reset forgets everything", "[pacing]" )
{
	Driver drv;
	WarmUp( drv, 16.0 );
	REQUIRE( drv.pacer.EstimateValid() );
	drv.pacer.Reset();
	CHECK_FALSE( drv.pacer.EstimateValid() );
	CHECK_FALSE( drv.pacer.Generating() );
	CHECK( drv.pacer.HistoryCount() == 0 );
	CHECK( drv.pacer.ContentNs() == 0 );
	const Pacer::Report s = drv.Status();
	CHECK( s.reason == Reason::WarmingUp );
}

// ---------------------------------------------------------------------------
//  Pause at refresh rate (Pacer::Inputs::pauseAtRefresh / canExceedRefresh)
// ---------------------------------------------------------------------------

TEST_CASE( "pacing: output interval without the refresh cap", "[pacing]" )
{
	// Fixed: interval / N with no 1/refresh floor (200 fps at 4x = 800/s on 144 Hz).
	CHECK( std::fabs( OutputIntervalMs( Mode::Fixed, 4, 0, 5.0, 144.0, false ) - 1.25 ) < 1e-9 );
	// ...while the capped form is unchanged.
	CHECK( std::fabs( OutputIntervalMs( Mode::Fixed, 4, 0, 5.0, 144.0, true ) - 1000.0 / 144.0 ) < 1e-9 );
	CHECK( std::fabs( OutputIntervalMs( Mode::Fixed, 4, 0, 5.0, 144.0 ) - 1000.0 / 144.0 ) < 1e-9 );
	// Target: not clamped to the refresh (500 fps on 144 Hz = 2 ms); 0 is still the refresh.
	CHECK( OutputIntervalMs( Mode::Target, 2, 500, 16.0, 144.0, false ) == 2.0 );
	CHECK( std::fabs( OutputIntervalMs( Mode::Target, 2, 0, 16.0, 144.0, false ) - 1000.0 / 144.0 ) < 1e-9 );
	CHECK( std::fabs( OutputIntervalMs( Mode::Target, 2, 500, 16.0, 144.0, true ) - 1000.0 / 144.0 ) < 1e-9 );
	// Never below the 0.5 ms floor (8x of a 250 fps game).
	CHECK( OutputIntervalMs( Mode::Fixed, 8, 0, 4.0, 60.0, false ) == kMinOutputMs );
}

TEST_CASE( "pacing: the extra timer's next target is one output interval on, never in the past", "[pacing]" )
{
	CHECK( NextExtraTargetNs( 0, 50 * MS, 1.25 ) == 50 * MS );                    // no previous: at once
	CHECK( NextExtraTargetNs( 100 * MS, 100 * MS, 1.25 ) == 100 * MS + 1250000 ); // on schedule
	CHECK( NextExtraTargetNs( 100 * MS, 130 * MS, 1.25 ) == 130 * MS );           // late: no burst of lost slots
}

// The extra timer's cadence reaches the target (2026-10-05). THE REPORT: in the user's session
// (nested Wayland, Hyprland at 280 Hz, the output timer active above the refresh) the presented
// rate was 1 / (o + ~1.05 ms), o = 1000 / target: target 335 gave ~247, 420 gave ~290, 1000
// gave ~500-580, independent of blur samples, game load and fps. THE SUSPECT was that the next
// tick is anchored to "now" after the paint (period = o + paint time). THE FINDING: it is not
// in this header: NextExtraTargetNs() has anchored to the PREVIOUS TARGET (never to the time
// the paint ended) since it was written, and the host (steamcompmgr.cpp FrameGen_ArmExtraTimer)
// passes the previous tick's target (ulLastV = Inputs::presentNs) as it. These tests pin that
// with a host that arms its timer AFTER a paint that takes 1.05 ms.

namespace
{
	// Ticks per second of a host that arms the next tick after the paint, which takes flPaintMs.
	// bAnchorNow models the suspected bug (the next tick is one interval after the paint's end).
	double ExtraTickRate( double flTarget, double flPaintMs, bool bAnchorNow )
	{
		const double flOutMs = 1000.0 / flTarget;
		Ns tTarget = 1 * MS;
		int n = 0;
		const Ns tEnd = 3000 * MS;
		while ( tTarget < tEnd )
		{
			n++;
			const Ns tPaintEnd = tTarget + Ns( flPaintMs * 1e6 );
			tTarget = bAnchorNow ? tPaintEnd + Ns( flOutMs * 1e6 ) : NextExtraTargetNs( tTarget, tPaintEnd, flOutMs );
		}
		return double( n ) / 3.0;
	}
}

TEST_CASE( "pacing: a paint that takes less than the output interval does not stretch the extra timer's cadence", "[pacing][extratimer]" )
{
	for ( double flTarget : { 335.0, 420.0, 600.0 } )
	{
		const double flRate = ExtraTickRate( flTarget, 1.05, false );
		const double flOld = ExtraTickRate( flTarget, 1.05, true );
		INFO( "target " << flTarget << ": anchored to the previous target " << flRate << "/s, anchored to the paint's end " << flOld << "/s" );
		CHECK( std::fabs( flRate - flTarget ) <= 0.02 * flTarget );
		CHECK( flOld < 0.85 * flTarget );        // the 1 / (o + 1.05 ms) of the report: the test has teeth
	}
	// a paint longer than the interval is the limit, as it must be: no catching up with a burst
	const double flSlow = ExtraTickRate( 1000.0, 1.05, false );
	CHECK( flSlow < 960.0 );
	CHECK( flSlow > 900.0 );
}

TEST_CASE( "pacing: a late extra tick (more than an interval) restarts the grid and never bursts the lost slots", "[pacing][extratimer]" )
{
	const double flOutMs = 1000.0 / 335.0;
	Ns tTarget = 100 * MS;
	tTarget = NextExtraTargetNs( tTarget, tTarget + 1 * MS, flOutMs );               // on schedule
	CHECK( tTarget == 100 * MS + Ns( flOutMs * 1e6 ) );
	// the host stalls 20 ms (6 intervals) in this paint: the next target is "now", not 6 slots in the past
	const Ns tLate = tTarget + 20 * MS;
	Ns t1 = NextExtraTargetNs( tTarget, tLate, flOutMs );
	CHECK( t1 == tLate );
	// and from there the grid carries on one interval at a time: no tick closer than o
	Ns t2 = NextExtraTargetNs( t1, t1 + 1 * MS, flOutMs );
	CHECK( t2 - t1 == Ns( flOutMs * 1e6 ) );
	// a slightly late one (under an interval) keeps the grid: the next target is still one o after it
	const Ns t3 = t2 + Ns( flOutMs * 1e6 ) / 2;
	CHECK( NextExtraTargetNs( t2, t3, flOutMs ) == t2 + Ns( flOutMs * 1e6 ) );
}

namespace
{
	// The pacer driven as steamcompmgr.cpp drives it on the extra timer: arrivals noticed at once,
	// a paint per tick at V = the tick's target taking flPaintMs, the next tick armed after the
	// paint (while the decision asks for one or a real frame arrived), one output interval after
	// the previous TARGET. Outputs per second once settled.
	double ExtraPacedOutputRate( int nTarget, double flGameFps, Priority ePriority, double flPaintMs )
	{
		Pacer pacer;
		Pacer::Inputs in;
		in.mode = Mode::Target;
		in.targetFps = nTarget;
		in.priority = ePriority;
		in.refreshHz = 280.0;
		in.pauseAtRefresh = false;
		in.canExceedRefresh = true;
		const Ns tEnd = 8000 * MS, tMeasure = 3000 * MS;
		const double flIntervalMs = 1000.0 / flGameFps;
		uint64_t id = 0;
		Ns nextArr = 1 * MS, now = 1 * MS, lastV = 0, armedTarget = 0;
		bool bArmed = false, bWantNext = false, bRepaint = false;
		int nOut = 0;
		while ( now < tEnd )
		{
			while ( nextArr <= now )
			{
				pacer.OnArrival( ++id, nextArr, 1, FrameFormat{ 1920, 1080, 87 } );
				bRepaint = true;
				nextArr += Ns( flIntervalMs * 1e6 );
			}
			if ( !bArmed && ( bRepaint || bWantNext ) )
			{
				armedTarget = pacer.ExtraTimer() ? NextExtraTargetNs( lastV, now, pacer.OutputMs() ) : now;
				bArmed = true;
			}
			if ( !bArmed || nextArr < armedTarget )
			{
				now = std::max( now, nextArr );
				continue;
			}
			now = std::max( now, armedTarget );
			bArmed = false;
			in.now = now;
			in.presentNs = armedTarget;
			const Pacer::Decision d = pacer.OnPaint( in, id, true, false );
			if ( armedTarget > tMeasure && d.newOutput )
				nOut++;
			lastV = armedTarget;
			bRepaint = false;
			bWantNext = d.repaintNext;
			now += Ns( flPaintMs * 1e6 );
		}
		return double( nOut ) / double( tEnd - tMeasure ) * 1e9;
	}
}

TEST_CASE( "pacing: the pacer on the extra timer reaches its target with a 1.05 ms paint (335, 420, 600 at 177 and 100 fps)", "[pacing][extratimer]" )
{
	for ( Priority ePriority : { Priority::Smoothness, Priority::LowLatency } )
		for ( int nTarget : { 335, 420, 600 } )
			for ( double flFps : { 177.0, 100.0 } )
			{
				const double flRate = ExtraPacedOutputRate( nTarget, flFps, ePriority, 1.05 );
				INFO( ( ePriority == Priority::Smoothness ? "smoothness" : "low latency" ) << " target " << nTarget << " game " << flFps << " fps: " << flRate << " outputs/s" );
				CHECK( std::fabs( flRate - nTarget ) <= 0.02 * nTarget );
			}
}

TEST_CASE( "pacing: only the extra timer paints while it paces the output", "[pacing]" )
{
	// Not generating: the loop as without frame generation.
	CHECK( PaintTick( false, false, true, true, false ) );
	CHECK_FALSE( PaintTick( false, false, false, true, true ) );
	// Generating, no extra timer: vblank-timer ticks only (unchanged).
	CHECK( PaintTick( true, true, true ) );
	CHECK_FALSE( PaintTick( true, false, true ) );
	// Generating on the extra timer: its ticks only, vblank ticks do not paint.
	CHECK( PaintTick( true, false, false, true, true ) );
	CHECK_FALSE( PaintTick( true, true, true, true, false ) );
}

TEST_CASE( "pacing: Off, fixed 4x at 200 fps on 144 Hz outputs 800/s and asks for the extra timer", "[pacing]" )
{
	Driver drv;
	drv.refreshHz = 144.0;
	drv.multiplier = 4;
	drv.pauseAtRefresh = false;
	drv.canExceed = true;
	WarmUp( drv, 5.0 );

	REQUIRE( drv.pacer.Generating() );
	CHECK( std::fabs( drv.pacer.OutputMs() - 1.25 ) < 1e-6 );
	CHECK( std::fabs( 1000.0 / drv.pacer.OutputMs() - 800.0 ) < 1e-3 );
	CHECK( drv.pacer.ExtraTimer() );
	CHECK( drv.Status().reason == Reason::Normal );

	// The same game with the switch ON passes through: it reaches the refresh by itself.
	Driver on;
	on.refreshHz = 144.0;
	on.multiplier = 4;
	WarmUp( on, 5.0 );
	CHECK_FALSE( on.pacer.Generating() );
	CHECK_FALSE( on.pacer.ExtraTimer() );
	CHECK( on.Status().reason == Reason::GameTooFast );
}

TEST_CASE( "pacing: Off on a backend that cannot exceed the refresh is capped like On", "[pacing]" )
{
	// 100 fps, 4x, 144 Hz: o = max(2.5, 6.94) = one vblank, vblank-paced.
	Driver drv;
	drv.refreshHz = 144.0;
	drv.multiplier = 4;
	drv.pauseAtRefresh = false;
	drv.canExceed = false;     // e.g. DRM without tearing
	WarmUp( drv, 10.0 );
	REQUIRE( drv.pacer.Generating() );
	CHECK( std::fabs( drv.pacer.OutputMs() - 1000.0 / 144.0 ) < 1e-6 );
	CHECK_FALSE( drv.pacer.ExtraTimer() );

	// ...and a 200 fps game still passes through (the refresh cap's ratio rule).
	Driver fast;
	fast.refreshHz = 144.0;
	fast.multiplier = 4;
	fast.pauseAtRefresh = false;
	fast.canExceed = false;
	WarmUp( fast, 5.0 );
	CHECK_FALSE( fast.pacer.Generating() );
	CHECK( fast.Status().reason == Reason::GameTooFast );

	// With the backend able to exceed it, the same Off setting generates and uses the timer.
	Driver can;
	can.refreshHz = 144.0;
	can.multiplier = 4;
	can.pauseAtRefresh = false;
	can.canExceed = true;
	WarmUp( can, 10.0 );
	REQUIRE( can.pacer.Generating() );
	CHECK( std::fabs( can.pacer.OutputMs() - 2.5 ) < 1e-6 );
	CHECK( can.pacer.ExtraTimer() );
}

TEST_CASE( "pacing: Off + target 500 on 144 Hz aims at 500/s", "[pacing]" )
{
	Driver drv;
	drv.refreshHz = 144.0;
	drv.mode = Mode::Target;
	drv.targetFps = 500;
	drv.pauseAtRefresh = false;
	drv.canExceed = true;
	WarmUp( drv, 10.0 );
	REQUIRE( drv.pacer.Generating() );
	CHECK( std::fabs( drv.pacer.OutputMs() - 2.0 ) < 1e-6 );
	CHECK( drv.pacer.ExtraTimer() );
	CHECK( drv.Status().targetFps == 500.0f );

	// On: capped to the refresh, as before.
	Driver on;
	on.refreshHz = 144.0;
	on.mode = Mode::Target;
	on.targetFps = 500;
	WarmUp( on, 10.0 );
	REQUIRE( on.pacer.Generating() );
	CHECK( std::fabs( on.pacer.OutputMs() - 1000.0 / 144.0 ) < 1e-6 );
	CHECK_FALSE( on.pacer.ExtraTimer() );
	CHECK( std::fabs( on.Status().targetFps - 144.0f ) < 1e-3f );

	// Off passes through only when the game itself reaches the target.
	Driver game;
	game.refreshHz = 144.0;
	game.mode = Mode::Target;
	game.targetFps = 200;
	game.pauseAtRefresh = false;
	game.canExceed = true;
	WarmUp( game, 5.0 );     // a 200 fps game, a 200 fps target
	CHECK_FALSE( game.pacer.Generating() );
	CHECK( game.Status().reason == Reason::GameTooFast );
}

TEST_CASE( "pacing: the cost guard still bounds Off", "[pacing]" )
{
	// 200 fps at 4x uncapped asks for 1.25 ms; the guard (25% of 5 ms = 1.25 ms of
	// GPU per pair) raises it to what the GPU affords: estimate 0.25 + 2 synths of 0.5.
	Driver drv;
	drv.refreshHz = 144.0;
	drv.multiplier = 4;
	drv.pauseAtRefresh = false;
	drv.canExceed = true;
	drv.estimateMs = 0.25f;
	drv.synthMs = 0.5f;
	drv.costSeq++;
	WarmUp( drv, 5.0 );
	REQUIRE( drv.pacer.Generating() );
	CHECK( std::fabs( drv.pacer.OutputMs() - 5.0 / 3.0 ) < 1e-6 );
	CHECK( drv.pacer.ExtraTimer() );
	CHECK( drv.Status().reason == Reason::CostGuard );

	// Too expensive for even one generated frame per pair: pass-through, no timer.
	Driver heavy;
	heavy.refreshHz = 144.0;
	heavy.multiplier = 4;
	heavy.pauseAtRefresh = false;
	heavy.canExceed = true;
	heavy.estimateMs = 0.9f;
	heavy.synthMs = 1.0f;
	heavy.costSeq++;
	WarmUp( heavy, 5.0 );
	CHECK_FALSE( heavy.pacer.Generating() );
	CHECK_FALSE( heavy.pacer.ExtraTimer() );
	CHECK( heavy.pacer.CostBlocked() );

	// A guard that lifts the output interval past a vblank hands the pacing back
	// to the vblank timer: 50 fps, 8x, one synth affordable = o 10 ms > 6.94 ms.
	Driver slow;
	slow.refreshHz = 144.0;
	slow.multiplier = 8;
	slow.pauseAtRefresh = false;
	slow.canExceed = true;
	slow.estimateMs = 2.0f;
	slow.synthMs = 2.0f;
	slow.costSeq++;
	WarmUp( slow, 20.0 );
	REQUIRE( slow.pacer.Generating() );
	CHECK( std::fabs( slow.pacer.OutputMs() - 10.0 ) < 1e-6 );
	CHECK_FALSE( slow.pacer.ExtraTimer() );
}

TEST_CASE( "pacing: Off + extra timer produces a distinct output frame per tick", "[pacing]" )
{
	// Drive the pacer the way the glue does on the extra timer: one paint per
	// output interval, V = the timer's target. 100 fps, 4x, 144 Hz -> 2.5 ms.
	Pacer pacer;
	Pacer::Inputs in;
	in.mode = Mode::Fixed;
	in.multiplier = 4;
	in.refreshHz = 144.0;
	in.pauseAtRefresh = false;
	in.canExceedRefresh = true;

	uint64_t id = 0;
	int nNew = 0;
	const Ns tEnd = 1500 * MS;
	Ns nextArrival = 0;
	Ns t = 1 * MS;   // not 0: NextExtraTargetNs() reads a 0 previous target as "none yet"
	std::set<uint64_t> outs;
	while ( t < tEnd )
	{
		// Real frames every 10 ms.
		while ( nextArrival <= t )
		{
			pacer.OnArrival( ++id, nextArrival, 1, FrameFormat{ 1920, 1080, 87 } );
			nextArrival += 10 * MS;
		}
		in.now = t;
		in.presentNs = t;
		const Pacer::Decision d = pacer.OnPaint( in, id, true );
		if ( t > 1000 * MS && d.newOutput )
		{
			nNew++;
			outs.insert( d.outId );
		}
		t = NextExtraTargetNs( t, t, pacer.Generating() ? pacer.OutputMs() : 2.5 );
	}
	REQUIRE( pacer.Generating() );
	CHECK( pacer.ExtraTimer() );
	// 0.5 s at 400 output frames a second.
	CHECK( nNew > 180 );
	CHECK( nNew < 220 );
	CHECK( outs.size() > 150 );
}

// ---------------------------------------------------------------------------
//  Steady-state output rate (cadence-skip vblanks, Low latency's smoothed D)
// ---------------------------------------------------------------------------

namespace
{
	// A game with deterministic arrival jitter (a MangoHud-capped one).
	std::vector<Ns> JitteredArrivals( double intervalMs, double jitterMs, Ns tEnd )
	{
		std::vector<Ns> v;
		uint32_t seed = 12345;
		for ( uint64_t n = 1;; n++ )
		{
			seed = seed * 1664525u + 1013904223u;
			const double j = ( double( ( seed >> 8 ) & 0xFFFF ) / 65535.0 * 2.0 - 1.0 ) * jitterMs;
			const Ns t = Ns( ( double( n ) * intervalMs + j ) * double( MS ) );
			if ( t >= tEnd )
				break;
			v.push_back( t );
		}
		return v;
	}
}

TEST_CASE( "pacing: a cadence-skip vblank commits nothing (fixed 2x, 30 fps, 120 Hz -> 60 commits/s)", "[pacing]" )
{
	for ( bool bSkip : { false, true } )
	{
		Driver drv;
		drv.refreshHz = 120.0;
		drv.multiplier = 2;
		drv.skipVblanks = bSkip;
		drv.RunArrivals( Driver::Periodic( 5 * MS, 1000.0 / 30.0, 6000 * MS ), 6000 * MS );
		REQUIRE( drv.pacer.Generating() );
		const int nCommits = drv.CountCommits( 3000 * MS, 5000 * MS );
		const int nOutputs = drv.CountOutputs( 3000 * MS, 5000 * MS );
		INFO( "skip " << bSkip << " commits " << nCommits << " outputs " << nOutputs );
		CHECK( std::abs( nOutputs - 120 ) <= 4 );    // 60/s for 2 s
		if ( bSkip )
			CHECK( std::abs( nCommits - 120 ) <= 6 );
		else
			CHECK( nCommits > 150 );                 // the cached output was re-presented
	}
}

TEST_CASE( "pacing: skipping is only for a repaint nobody else asked for", "[pacing]" )
{
	Driver drv;
	drv.refreshHz = 120.0;
	drv.multiplier = 2;
	drv.uiRepaints = true;       // a UI repaint on every vblank (the driver then passes bSkippable false)
	drv.skipVblanks = true;
	drv.RunArrivals( Driver::Periodic( 5 * MS, 1000.0 / 30.0, 2500 * MS ), 2500 * MS );
	REQUIRE( drv.pacer.Generating() );
	for ( const Painted &p : drv.log )
		CHECK_FALSE( p.d.skip );
}


TEST_CASE( "pacing: Low latency keeps (nearly) every tick distinct for a steady jittered game", "[pacing]" )
{
	// 30 fps +-1 ms, fixed 8x on 120 Hz: o = one vblank -> 120 distinct outputs/s at best.
	// Low latency pays minimal delay, so a frame that lands late may leave one tick
	// repeating (about 5% here); Smoothness is the priority that never does.
	Driver drv;
	drv.refreshHz = 120.0;
	drv.multiplier = 8;
	drv.skipVblanks = true;
	drv.RunArrivals( JitteredArrivals( 1000.0 / 30.0, 1.0, 8000 * MS ), 8000 * MS );
	REQUIRE( drv.pacer.Generating() );
	const int nDistinct = drv.CountDistinct( 4000 * MS, 6000 * MS );
	INFO( "distinct outputs in 2 s: " << nDistinct );
	CHECK( nDistinct >= 216 );                        // >= 90% of 240
	CHECK( nDistinct <= 244 );

	// The same for Target 120.
	Driver tgt;
	tgt.refreshHz = 120.0;
	tgt.mode = Mode::Target;
	tgt.targetFps = 120;
	tgt.skipVblanks = true;
	tgt.RunArrivals( JitteredArrivals( 1000.0 / 30.0, 1.0, 8000 * MS ), 8000 * MS );
	const int nTgt = tgt.CountDistinct( 4000 * MS, 6000 * MS );
	INFO( "target 120 distinct outputs in 2 s: " << nTgt );
	CHECK( nTgt >= 216 );
	CHECK( nTgt <= 244 );

	// Smoothness, same game: every tick.
	Driver smo;
	smo.refreshHz = 120.0;
	smo.multiplier = 8;
	smo.priority = Priority::Smoothness;
	smo.skipVblanks = true;
	smo.RunArrivals( JitteredArrivals( 1000.0 / 30.0, 1.0, 8000 * MS ), 8000 * MS );
	CHECK( std::abs( smo.CountDistinct( 4000 * MS, 6000 * MS ) - 240 ) <= 4 );
}

TEST_CASE( "pacing: Low latency, steady game: 60 fps on 144 Hz, target = refresh, >= 95% of ticks distinct", "[pacing]" )
{
	Driver drv;
	drv.refreshHz = 144.0;
	drv.mode = Mode::Target;
	drv.targetFps = 0;
	drv.skipVblanks = true;
	drv.RunArrivals( Driver::Periodic( 5 * MS, 1000.0 / 60.0, 8000 * MS ), 8000 * MS );
	REQUIRE( drv.pacer.Generating() );
	const int nDistinct = drv.CountDistinct( 4000 * MS, 6000 * MS );
	INFO( "distinct outputs in 2 s: " << nDistinct );
	CHECK( nDistinct >= 274 );                        // >= 95% of 288
	CHECK( nDistinct <= 290 );

	// With 0.5 ms of arrival jitter some ticks may repeat.
	Driver jit;
	jit.refreshHz = 144.0;
	jit.mode = Mode::Target;
	jit.targetFps = 0;
	jit.skipVblanks = true;
	jit.RunArrivals( JitteredArrivals( 1000.0 / 60.0, 0.5, 8000 * MS ), 8000 * MS );
	const int nJit = jit.CountDistinct( 4000 * MS, 6000 * MS );
	INFO( "jittered distinct outputs in 2 s: " << nJit );
	CHECK( nJit >= 259 );                             // >= 90%
}

TEST_CASE( "pacing: Low latency still snaps on a genuinely early or late frame", "[pacing]" )
{
	// The smoothed interval is used while the latest one is within max(2 ms, 15%).
	CHECK( LowLatencyIntervalMs( 33.3, 34.3 ) == 33.3 );
	CHECK( LowLatencyIntervalMs( 33.3, 31.0 ) == 33.3 );
	CHECK( LowLatencyIntervalMs( 33.3, 20.0 ) == 20.0 );
	CHECK( LowLatencyIntervalMs( 33.3, 45.0 ) == 45.0 );
	CHECK( LowLatencyIntervalMs( 8.0, 9.9 ) == 8.0 );      // 2 ms floor
	CHECK( LowLatencyIntervalMs( 8.0, 10.5 ) == 10.5 );
	CHECK( LowLatencyIntervalMs( 16.0, 0.0 ) == 16.0 );    // no latest interval yet
}

TEST_CASE( "pacing: P75DevOf is the typical jitter, not the worst", "[pacing]" )
{
	const double steady[ 8 ] = { 10, 10, 10, 10, 10, 10, 10, 10 };
	CHECK( P75DevOf( steady, 8 ) == 0.0 );
	// One late frame in eight does not move it (the worst-case deviation would be 1.5).
	const double hitch[ 8 ] = { 10, 10, 10, 10, 10, 10, 10, 11.5 };
	CHECK( P75DevOf( hitch, 8 ) == 0.0 );
	// A genuinely jittery game does: deviations 1,0,1,0,1,1,0,0 -> the 6th smallest is 1.
	const double jittery[ 8 ] = { 9, 10, 11, 10, 9, 11, 10, 10 };
	CHECK( P75DevOf( jittery, 8 ) == 1.0 );
	CHECK( P75DevOf( jittery, 2 ) == 0.0 );                 // too few samples
}

TEST_CASE( "pacing: Low latency's margin is the typical jitter, at most 1 ms and 20% of an interval", "[pacing]" )
{
	CHECK( LowLatencyMarginMs( 0.0, 16.0 ) == 0.0 );        // a steady game pays nothing
	CHECK( LowLatencyMarginMs( 0.5, 16.0 ) == 0.5 );
	CHECK( LowLatencyMarginMs( 3.0, 16.0 ) == 1.0 );        // capped at 1 ms
	CHECK( std::fabs( LowLatencyMarginMs( 0.8, 3.0 ) - 0.6 ) < 1e-9 );   // capped at 20% of 3 ms
	CHECK( LowLatencyMarginMs( -1.0, 16.0 ) == 0.0 );
}

namespace
{
	// The mean total delay D over the new outputs in [t0, t1).
	double MeanDelayMs( const Driver &drv, Ns t0, Ns t1 )
	{
		double sum = 0.0;
		int n = 0;
		for ( const Painted &p : drv.log )
			if ( p.t >= t0 && p.t < t1 && p.d.newOutput )
			{
				sum += p.delayMs;
				n++;
			}
		return n ? sum / n : 0.0;
	}

	Driver Run280( double gameFps, double jitterMs, Priority prio, Ns leadNs = 0 )
	{
		Driver drv;
		drv.refreshHz = 280.0;
		drv.mode = Mode::Target;
		drv.targetFps = 0;
		drv.priority = prio;
		drv.skipVblanks = true;
		drv.leadNs = leadNs;
		drv.RunArrivals( JitteredArrivals( 1000.0 / gameFps, jitterMs, 8000 * MS ), 8000 * MS );
		return drv;
	}
}

TEST_CASE( "pacing: Low latency adds about i - o (+ typical jitter) at 120 / 160 / 210 fps on 280 Hz, well below Smoothness", "[pacing]" )
{
	// The user's CS2 session ("I feel like the input latency has become worse"): Low
	// latency measured 7.9-9.2 ms, no lower than Smoothness's 7.7-8.7.
	for ( double fps : { 120.0, 160.0, 210.0 } )
	{
		const double flI = 1000.0 / fps, flO = 1000.0 / 280.0;
		Driver low = Run280( fps, 1.0, Priority::LowLatency );
		Driver smo = Run280( fps, 1.0, Priority::Smoothness );
		REQUIRE( low.pacer.Generating() );
		const double dLow = MeanDelayMs( low, 4000 * MS, 6000 * MS );
		const double dSmo = MeanDelayMs( smo, 4000 * MS, 6000 * MS );
		INFO( fps << " fps: i - o " << flI - flO << " ms, Low latency D " << dLow << " ms, Smoothness D " << dSmo << " ms" );
		CHECK( dLow <= ( flI - flO ) + 1.5 );
		CHECK( dLow >= ( flI - flO ) - 0.01 );               // no floor, but never below i - o either
		CHECK( dLow + 3.0 <= dSmo );
		// No floor of one output interval: at 210 fps D is below one tick (3.57 ms).
		if ( fps >= 210.0 )
			CHECK( dLow < flO );
	}

	// A steady 60 fps on 144 Hz: exactly i - o, no margin at all.
	Driver s60;
	s60.refreshHz = 144.0;
	s60.mode = Mode::Target;
	s60.targetFps = 0;
	s60.skipVblanks = true;
	s60.RunArrivals( Driver::Periodic( 5 * MS, 1000.0 / 60.0, 8000 * MS ), 8000 * MS );
	CHECK( std::fabs( MeanDelayMs( s60, 4000 * MS, 6000 * MS ) - ( 1000.0 / 60.0 - 1000.0 / 144.0 ) ) < 0.05 );
}

TEST_CASE( "pacing: Low latency's D does not grow with the paint's lead before the vblank", "[pacing]" )
{
	// The lead used to be added to D (it kept every tick filled). Low latency now
	// leaves it out on purpose: minimal delay is the priority, and a frame landing
	// inside the lead is simply seen one tick later.
	const double flI = 1000.0 / 160.0, flO = 1000.0 / 280.0;
	for ( Ns lead : { Ns( 0 ), 2 * MS, 4 * MS } )
	{
		Driver low = Run280( 160.0, 1.0, Priority::LowLatency, lead );
		const double d = MeanDelayMs( low, 4000 * MS, 6000 * MS );
		INFO( "lead " << lead / MS << " ms: D " << d << " ms" );
		CHECK( d <= ( flI - flO ) + 1.5 );
	}
}

// ---------------------------------------------------------------------------
//  High ratios on a fast display (the user's CS2 session: 280 Hz, 212 / 234 fps)
// ---------------------------------------------------------------------------

namespace
{
	// Target = refresh at 280 Hz. Smoothness: the presented rate never falls below the
	// game's own. Low latency (minimal delay): a late frame may repeat one tick, so the
	// output rate is only required to stay near the refresh (nMinOut) -- but with the
	// vblank seeing every arrival (lead 0) it still never falls below the game's rate,
	// and real frames are still shown (the renderer shows the real frame itself, t = 1).
	void CheckHighRatio( double gameFps, Priority prio, Ns leadNs, double jitterMs, int nMinOut, double flMinGameShare = 1.0 )
	{
		Driver drv;
		drv.refreshHz = 280.0;
		drv.mode = Mode::Target;
		drv.targetFps = 0;
		drv.priority = prio;
		drv.skipVblanks = true;
		drv.leadNs = leadNs;
		const std::vector<Ns> arr = JitteredArrivals( 1000.0 / gameFps, jitterMs, 8000 * MS );
		drv.RunArrivals( arr, 8000 * MS );
		const Ns t0 = 4000 * MS, t1 = 6000 * MS;
		int nGame = 0;
		for ( Ns a : arr )
			nGame += ( a >= t0 && a < t1 ) ? 1 : 0;
		const int nOut = drv.CountOutputs( t0, t1 );
		INFO( "game " << gameFps << " fps, prio " << int( prio ) << ", lead " << leadNs / MS << " ms, jitter " << jitterMs << " ms: arrivals " << nGame
			<< " / 2 s, outputs " << nOut << " / 2 s, generating " << drv.pacer.Generating() );
		CHECK( double( nOut ) >= flMinGameShare * double( nGame ) );
		if ( nMinOut > 0 )
		{
			REQUIRE( drv.pacer.Generating() );
			CHECK( nOut >= nMinOut );
		}
	}
}

TEST_CASE( "pacing: 212 fps on 280 Hz, target = refresh, Low latency never presents fewer than the game", "[pacing]" )
{
	// Steady and +-1 ms: >= 97% of the 280 Hz ticks carry an output (the rest repeat).
	CheckHighRatio( 212.0, Priority::LowLatency, 0, 0.0, 540 );
	CheckHighRatio( 212.0, Priority::LowLatency, 0, 1.0, 540 );
	// +-2 ms is worse than the game's own interval spread: >= 90% of the ticks.
	// (487 measured since the interval clamp dropped to 1 ms: at +-2 ms on a 4.7 ms interval the
	// 8-sample median briefly dips under the stop ratio, which the old 4 ms floor used to hide.)
	CheckHighRatio( 212.0, Priority::LowLatency, 0, 2.0, 470 );
	// A 2 ms paint lead hides frames that land in it: the count can dip a little under
	// the game's (measured 421 / 424 and 416 / 424), never more than 3%.
	CheckHighRatio( 212.0, Priority::LowLatency, 2 * MS, 1.0, 0, 0.97 );
	CheckHighRatio( 212.0, Priority::LowLatency, 2 * MS, 2.0, 0, 0.97 );
}

TEST_CASE( "pacing: 234 fps on 280 Hz, target = refresh, Smoothness never presents fewer than the game", "[pacing]" )
{
	CheckHighRatio( 234.0, Priority::Smoothness, 0, 0.5, 0 );
	CheckHighRatio( 234.0, Priority::Smoothness, 2 * MS, 0.5, 0 );
	CheckHighRatio( 234.0, Priority::Smoothness, 0, 0.0, 0 );
	CheckHighRatio( 234.0, Priority::Smoothness, 0, 1.5, 0 );
	CheckHighRatio( 234.0, Priority::Smoothness, 2 * MS, 1.5, 0 );
}

TEST_CASE( "pacing: Smoothness keeps the output rate and never presents fewer than the game whatever the paint's lead", "[pacing]" )
{
	// Smoothness's D >= one interval already clears any lead, so the paint-lead logic
	// that Low latency used to carry is not needed there.
	CheckHighRatio( 212.0, Priority::Smoothness, 4 * MS, 1.0, 540 );
	CheckHighRatio( 212.0, Priority::Smoothness, 4 * MS, 2.0, 0 );        // outputs 537 (>= the game's); "still generating at the end" is no longer guaranteed at +-2 ms, see above
	CheckHighRatio( 120.0, Priority::Smoothness, 4 * MS, 1.0, 540 );
}

// ---------------------------------------------------------------------------
//  Motion-blur window (trailing shutter: the window ends at the shown instant)
// ---------------------------------------------------------------------------

namespace
{
	constexpr double kOut = 8.0e6;     // output interval: 8 ms (125 Hz)
	constexpr double kGame = 32.0e6;   // game interval: 32 ms (a pair spans 4 outputs, M = 4)
	constexpr float kEps = 1e-6f;

	bool Near( float a, float b ) { return std::fabs( a - b ) < kEps; }
}

TEST_CASE( "pacing: blur window, blur only is [1 - s, 1]", "[pacing][blur]" )
{
	// No interpolation: the real frame is shown (t = 1), output interval = game interval.
	{
		const BlurRelative r = BlurRelative::ShownFrame;
		const BlurSpan h = BlurWindow( 1.0, kGame, kGame, 0.5, r );
		CHECK( Near( h.t0, 0.5f ) );
		CHECK( Near( h.t1, 1.0f ) );
		const BlurSpan f = BlurWindow( 1.0, kGame, kGame, 1.0, r );
		CHECK( Near( f.t0, 0.0f ) );
		CHECK( Near( f.t1, 1.0f ) );
	}
	// The film look at t = 1 is the same window.
	const BlurSpan g = BlurWindow( 1.0, kGame, kGame, 0.5, BlurRelative::GameFrame );
	CHECK( Near( g.t0, 0.5f ) );
	CHECK( Near( g.t1, 1.0f ) );
}

TEST_CASE( "pacing: blur window with interpolation is [(m - s)/M, m/M], s = 1 tiles the pair", "[pacing][blur]" )
{
	const int M = 4;
	for ( double s : { 0.25, 0.5, 1.0 } )
		for ( int m = 1; m <= M; m++ )
		{
			const BlurSpan w = BlurWindow( double( m ) / M, kOut, kGame, s, BlurRelative::ShownFrame );
			INFO( "s " << s << " m " << m );
			CHECK( Near( w.t0, float( ( m - s ) / M ) ) );
			CHECK( Near( w.t1, float( double( m ) / M ) ) );
		}
	// s = 1: neighbouring windows meet end to start (the pair is tiled exactly).
	for ( int m = 1; m < M; m++ )
		CHECK( Near( BlurWindow( double( m ) / M, kOut, kGame, 1.0, BlurRelative::ShownFrame ).t1,
			BlurWindow( double( m + 1 ) / M, kOut, kGame, 1.0, BlurRelative::ShownFrame ).t0 ) );
	// the window ends at the shown instant, never past it (no future frame needed)
	CHECK( Near( BlurWindow( 0.3, kOut, kGame, 1.0, BlurRelative::ShownFrame ).t1, 0.3f ) );
}

TEST_CASE( "pacing: blur window, shown-frame length follows the output interval", "[pacing][blur]" )
{
	// Shown-frame: the same amount is a longer stretch of the pair at a lower output rate.
	const BlurSpan a = BlurWindow( 0.8, kOut, kGame, 1.0, BlurRelative::ShownFrame );
	const BlurSpan b = BlurWindow( 0.8, 2.0 * kOut, kGame, 1.0, BlurRelative::ShownFrame );
	CHECK( Near( a.t1 - a.t0, 0.25f ) );
	CHECK( Near( b.t1 - b.t0, 0.5f ) );
}

TEST_CASE( "pacing: blur window, the film look is [a, a + L] with a = clamp(t - L, 0, 1 - L)", "[pacing][blur]" )
{
	// Far enough into the pair: plain trailing.
	const BlurSpan a = BlurWindow( 0.9, kOut, kGame, 0.5, BlurRelative::GameFrame );
	CHECK( Near( a.t0, 0.4f ) );
	CHECK( Near( a.t1, 0.9f ) );
	// Near the start: the length is kept and the window is slid into the pair.
	const BlurSpan b = BlurWindow( 0.25, kOut, kGame, 0.5, BlurRelative::GameFrame );
	CHECK( Near( b.t0, 0.0f ) );
	CHECK( Near( b.t1, 0.5f ) );
	// L = 1 is the whole pair, whatever t.
	const BlurSpan c = BlurWindow( 0.4, kOut, kGame, 1.0, BlurRelative::GameFrame );
	CHECK( Near( c.t0, 0.0f ) );
	CHECK( Near( c.t1, 1.0f ) );
	// independent of the intervals
	const BlurSpan d = BlurWindow( 0.9, 0.0, 0.0, 0.5, BlurRelative::GameFrame );
	CHECK( Near( d.t0, 0.4f ) );
	CHECK( Near( d.t1, 0.9f ) );
}

TEST_CASE( "pacing: blur window is clamped to the pair", "[pacing][blur]" )
{
	// Shown-frame at a t that is not on m/M: cut at 0, not moved.
	const BlurSpan a = BlurWindow( 0.1, kOut, kGame, 1.0, BlurRelative::ShownFrame );
	CHECK( Near( a.t0, 0.0f ) );
	CHECK( Near( a.t1, 0.1f ) );
	// Out-of-range t, an oversized amount and an output interval above the game interval all stay inside [0, 1].
	for ( double t : { -1.0, 0.0, 0.5, 1.0, 2.0 } )
		for ( double amount : { 0.0, 0.25, 1.0, 5.0 } )
			for ( double out : { 0.5 * kGame, kGame, 3.0 * kGame } )
				for ( BlurRelative rel : { BlurRelative::ShownFrame, BlurRelative::GameFrame } )
				{
					const BlurSpan w = BlurWindow( t, out, kGame, amount, rel );
					INFO( "t " << t << " amount " << amount << " out " << out );
					CHECK( w.t0 >= 0.0f );
					CHECK( w.t1 <= 1.0f );
					CHECK( w.t0 <= w.t1 );
					CHECK( w.t1 <= std::max( 0.0f, std::min( float( t ), 1.0f ) ) + 1e-6f || rel == BlurRelative::GameFrame );
				}
}

TEST_CASE( "pacing: blur window, amount 0 is a point", "[pacing][blur]" )
{
	for ( BlurRelative rel : { BlurRelative::ShownFrame, BlurRelative::GameFrame } )
	{
		const BlurSpan w = BlurWindow( 0.37, kOut, kGame, 0.0, rel );
		CHECK( Near( w.t0, 0.37f ) );
		CHECK( Near( w.t1, 0.37f ) );
	}
	// Unusable intervals (no estimate yet) degrade to a point as well, never NaN.
	const BlurSpan z = BlurWindow( 1.0, 0.0, 0.0, 0.5, BlurRelative::ShownFrame );
	CHECK( Near( z.t0, 1.0f ) );
	CHECK( Near( z.t1, 1.0f ) );
	const BlurSpan n = BlurWindow( 0.5, kOut, kGame, std::nan( "" ), BlurRelative::GameFrame );
	CHECK( Near( n.t0, 0.5f ) );
	CHECK( Near( n.t1, 0.5f ) );
}

TEST_CASE( "pacing: blur window follows the pacer's decision", "[pacing][blur]" )
{
	// A 30 fps game generated at 4x: every synth's window ends at its own t, lies inside the
	// pair, and a larger shutter never gives a shorter window.
	Driver drv;
	drv.refreshHz = 125.0;
	drv.multiplier = 4;
	WarmUp( drv, 32.0 );
	REQUIRE( drv.pacer.Generating() );
	int nSynth = 0;
	for ( const Painted &p : drv.log )
	{
		if ( p.d.t >= 1.0f )
			continue;
		nSynth++;
		const BlurSpan narrow = BlurWindow( p.d.t, drv.pacer.OutputMs() * 1e6, drv.pacer.IntervalMs() * 1e6, 0.25, BlurRelative::ShownFrame );
		const BlurSpan wide = BlurWindow( p.d.t, drv.pacer.OutputMs() * 1e6, drv.pacer.IntervalMs() * 1e6, 0.75, BlurRelative::ShownFrame );
		CHECK( Near( wide.t1, p.d.t ) );
		CHECK( Near( narrow.t1, p.d.t ) );
		CHECK( wide.t0 >= 0.0f );
		CHECK( wide.t1 - wide.t0 >= narrow.t1 - narrow.t0 );
	}
	CHECK( nSynth > 0 );
}

// ---------------------------------------------------------------------------
//  The three features on one timeline: frame generation (which instants get an
//  output), the lag-spike buffer (how far behind), motion blur (the window each
//  output shows). Each works alone, every combination works.
// ---------------------------------------------------------------------------

namespace
{
	// An output that went through the renderer's one call (recordSynthBlur(prev, curr, out, t0, t1)).
	bool IsRendered( const Pacer::Decision &d ) { return d.prevId != 0; }

	int CountNew( const Driver &drv, Ns t0, Ns t1 )
	{
		return drv.CountOutputs( t0, t1 );
	}
}

TEST_CASE( "pacing: FG off, no blur, no delay is a pure pass-through", "[pacing][combine]" )
{
	Driver drv;
	drv.frameGen = false;
	WarmUp( drv, 32.0 );
	const Pacer::Decision d = drv.log.back().d;
	CHECK( d.inert );
	CHECK_FALSE( drv.pacer.Planning() );
	CHECK( drv.Status().reason == Reason::Off );
	CHECK( d.t0 == 1.0f );
	CHECK( d.t1 == 1.0f );
	CHECK( d.prevId == 0 );
}

TEST_CASE( "pacing: FG off + blur: one output per real frame, window [1 - s, 1]", "[pacing][combine]" )
{
	Driver drv;
	drv.refreshHz = 125.0;
	drv.frameGen = false;
	drv.blur = true;
	drv.blurAmount = 0.5;
	WarmUp( drv, 32.0, 1000 * MS );
	const Ns tStart = drv.now;
	const std::vector<Ns> arrivals = Driver::Periodic( tStart + 3 * MS, 32.0, tStart + 2000 * MS );
	drv.RunArrivals( arrivals, tStart + 2000 * MS );
	REQUIRE( drv.pacer.Planning() );
	CHECK_FALSE( drv.pacer.Generating() );
	CHECK( drv.pacer.RealRate() );
	// exactly one output per real frame
	const int nOut = CountNew( drv, tStart + 100 * MS, tStart + 1900 * MS );
	int nArr = 0;
	for ( Ns a : arrivals )
		nArr += ( a >= tStart + 100 * MS && a < tStart + 1900 * MS ) ? 1 : 0;
	INFO( "outputs " << nOut << " arrivals " << nArr );
	CHECK( std::abs( nOut - nArr ) <= 1 );        // the window edges can split a frame from its paint
	int nChecked = 0;
	for ( const Painted &p : drv.log )
	{
		if ( p.t < tStart + 100 * MS || !p.d.newOutput )
			continue;
		nChecked++;
		CHECK( IsRendered( p.d ) );
		CHECK( p.d.t == 1.0f );
		CHECK( Near( p.d.t0, 0.5f ) );
		CHECK( Near( p.d.t1, 1.0f ) );
		CHECK( p.d.currId == p.d.showId );     // the real frame itself, blurred over the part before it
	}
	CHECK( nChecked > 40 );
	const Pacer::Report r = drv.Status();
	CHECK( r.reason == Reason::Off );
	CHECK_FALSE( r.fgActive );
	CHECK( r.blurActive );
	CHECK( r.blurSamples == 4 );
	CHECK( r.blurWindowMs > 15.0f );           // 0.5 of a 32 ms interval = 16 ms
	CHECK( r.blurWindowMs < 17.0f );
	CHECK( r.bufferDelayMs == 0.0f );
}

TEST_CASE( "pacing: FG 2x + blur: each output has its own window, ending at its instant, the windows tile", "[pacing][combine]" )
{
	Driver drv;
	drv.refreshHz = 125.0;
	drv.multiplier = 2;
	drv.blur = true;
	drv.blurAmount = 1.0;                      // 360 degrees: tiles
	// Real frames land exactly on a tick (offset 0), so the output instants sit on the
	// m/M grid and the windows meet end to start. Off the grid (a late tick, jitter)
	// they overlap or leave a small gap, the way the cadence does.
	WarmUp( drv, 32.0, 1500 * MS, 0 );
	REQUIRE( drv.pacer.Generating() );
	int nSynth = 0, nReal = 0;
	for ( size_t i = 0; i < drv.log.size(); i++ )
	{
		const Painted &p = drv.log[ i ];
		if ( p.t < 800 * MS || !p.d.newOutput )
			continue;
		CHECK( IsRendered( p.d ) );
		CHECK( p.d.t0 <= p.d.t1 );
		CHECK( Near( p.d.t1, p.d.t ) );
		CHECK( p.d.t1 - p.d.t0 > 0.45f );      // one output interval = half the pair
		CHECK( p.d.t1 - p.d.t0 < 0.55f );
		( p.d.t < 1.0f ? nSynth : nReal )++;
		// the previous output of the same pair ended where this one starts
		for ( size_t k = i; k-- > 0; )
		{
			const Painted &q = drv.log[ k ];
			if ( !q.d.newOutput )
				continue;
			if ( q.d.currId == p.d.currId && q.d.t < p.d.t )
				CHECK( std::fabs( q.d.t1 - p.d.t0 ) < 0.02f );
			break;
		}
	}
	CHECK( nSynth > 10 );
	CHECK( nReal > 10 );
	const Pacer::Report r = drv.Status();
	CHECK( r.fgActive );
	CHECK( r.blurActive );
	CHECK( r.activeN == 2 );
}

TEST_CASE( "pacing: blur at the refresh: the game reaches the output rate, blur still applies per real frame", "[pacing][combine]" )
{
	Driver drv;
	drv.refreshHz = 60.0;
	drv.multiplier = 2;
	drv.blur = true;
	drv.blurAmount = 0.5;
	WarmUp( drv, 10.0, 1500 * MS );             // 100 fps on 60 Hz: nothing to generate
	CHECK_FALSE( drv.pacer.Generating() );
	REQUIRE( drv.pacer.RealRate() );
	const Pacer::Report r = drv.Status();
	CHECK( r.reason == Reason::GameTooFast );
	CHECK_FALSE( r.fgActive );
	CHECK( r.blurActive );
	const Pacer::Decision d = drv.log.back().d;
	CHECK_FALSE( d.inert );
	// and with blur off the same game is a plain pass-through again
	Driver off;
	off.refreshHz = 60.0;
	off.multiplier = 2;
	WarmUp( off, 10.0, 1500 * MS );
	CHECK( off.log.back().d.inert );
	CHECK( off.Status().reason == Reason::GameTooFast );
}

namespace
{
	// 16 ms real frames with `nMissing` of them missing after the 50th.
	std::vector<Ns> GapArrivals( int nMissing, Ns tEnd )
	{
		std::vector<Ns> v;
		for ( int k = 0; ; k++ )
		{
			if ( k > 50 && k <= 50 + nMissing )
				continue;
			const Ns t = 3 * MS + Ns( k ) * 16 * MS;
			if ( t >= tEnd )
				break;
			v.push_back( t );
		}
		return v;
	}

	// The longest stretch between two consecutive new outputs in [t0, t1), ms.
	double LongestOutputGapMs( const Driver &drv, Ns t0, Ns t1 )
	{
		double flMax = 0.0;
		Ns tPrev = 0;
		for ( const Painted &p : drv.log )
		{
			if ( p.t < t0 || p.t >= t1 || !p.d.newOutput )
				continue;
			if ( tPrev != 0 )
				flMax = std::max( flMax, double( p.t - tPrev ) / double( MS ) );
			tPrev = p.t;
		}
		return flMax;
	}

	int CountRenderedSynths( const Driver &drv, Ns t0, Ns t1 )
	{
		int n = 0;
		for ( const Painted &p : drv.log )
			n += ( p.t >= t0 && p.t < t1 && p.d.newOutput && p.d.prevId != 0 && p.d.t < 1.0f ) ? 1 : 0;
		return n;
	}
}

TEST_CASE( "pacing: FG off + a 50 ms buffer fills a two-frame gap with synths and no hitch", "[pacing][combine]" )
{
	const Ns tEnd = 2000 * MS;
	// The gap sits at real frame 51..52, i.e. 3 + 51 * 16 = 819 .. 851 ms.
	Driver buffered;
	buffered.refreshHz = 125.0;
	buffered.frameGen = false;
	buffered.extraDelayNs = 50 * MS;
	buffered.RunArrivals( GapArrivals( 2, tEnd ), tEnd );
	CHECK( buffered.pacer.HistoryDepth() > kHistory );
	const double flGapBuffered = LongestOutputGapMs( buffered, 400 * MS, 1900 * MS );
	const int nFilled = CountRenderedSynths( buffered, 800 * MS, 1200 * MS );
	INFO( "longest gap between outputs with the buffer: " << flGapBuffered << " ms, synths " << nFilled );
	CHECK( flGapBuffered <= 24.0 );             // 16 ms frames on 8 ms ticks: no hitch
	CHECK( nFilled >= 2 );                      // the missing frames were synthesised

	// without the buffer the same stream hitches: nothing can be filled
	Driver plain;
	plain.refreshHz = 125.0;
	plain.frameGen = false;
	plain.blur = true;                          // keeps the planner active
	plain.blurAmount = 0.5;
	plain.RunArrivals( GapArrivals( 2, tEnd ), tEnd );
	CHECK( LongestOutputGapMs( plain, 400 * MS, 1900 * MS ) > 40.0 );
	CHECK( CountRenderedSynths( plain, 800 * MS, 1200 * MS ) == 0 );
	// the delay shows in the status: D is the buffer alone, the buffer is its own line
	const Pacer::Report r = buffered.Status();
	CHECK( Near( r.delayMs, 50.0f ) );
	CHECK( Near( r.bufferDelayMs, 50.0f ) );
	CHECK_FALSE( r.fgActive );
}

TEST_CASE( "pacing: FG off + buffer: real frames are shown at their own time plus D", "[pacing][combine]" )
{
	Driver drv;
	drv.refreshHz = 125.0;
	drv.frameGen = false;
	drv.extraDelayNs = 40 * MS;
	const Ns tEnd = 1500 * MS;
	drv.RunArrivals( Driver::Periodic( 3 * MS, 16.0, tEnd ), tEnd );
	int n = 0;
	for ( const Painted &p : drv.log )
	{
		if ( p.t < 800 * MS || !p.d.newOutput )
			continue;
		// the arrival time of the frame shown: its id is the arrival index + 1
		const Ns tArr = 3 * MS + Ns( ( p.d.showId - 1 ) * 16 * MS );
		INFO( "shown at " << p.t / MS << " ms, frame " << p.d.showId << " arrived " << tArr / MS << " ms" );
		CHECK( p.t >= tArr + 40 * MS );
		CHECK( p.t < tArr + 40 * MS + 9 * MS );        // within one 8 ms tick of arrival + D
		n++;
	}
	CHECK( n > 30 );
}

TEST_CASE( "pacing: FG + buffer: delayed, so the gap is interpolated", "[pacing][combine]" )
{
	const Ns tEnd = 2000 * MS;
	Driver drv;
	drv.refreshHz = 125.0;
	drv.multiplier = 2;
	drv.extraDelayNs = 50 * MS;
	drv.RunArrivals( GapArrivals( 2, tEnd ), tEnd );
	REQUIRE( drv.pacer.Generating() );
	const Pacer::Report r = drv.Status();
	CHECK( r.fgActive );
	CHECK( r.bufferDelayMs == 50.0f );
	CHECK( r.delayMs >= 50.0f );
	CHECK( LongestOutputGapMs( drv, 400 * MS, 1900 * MS ) <= 24.0 );
	// FG alone, the same: the extra delay is the only difference in the status
	Driver alone;
	alone.refreshHz = 125.0;
	alone.multiplier = 2;
	alone.RunArrivals( GapArrivals( 0, tEnd ), tEnd );
	CHECK( alone.Status().bufferDelayMs == 0.0f );
	CHECK( alone.Status().delayMs < 50.0f );
}

TEST_CASE( "pacing: all three on at once", "[pacing][combine]" )
{
	const Ns tEnd = 2500 * MS;
	Driver drv;
	drv.refreshHz = 125.0;
	drv.multiplier = 2;
	drv.extraDelayNs = 50 * MS;
	drv.blur = true;
	drv.blurSamples = 4;
	drv.blurAmount = 0.5;
	drv.RunArrivals( GapArrivals( 2, tEnd ), tEnd );
	REQUIRE( drv.pacer.Generating() );
	int n = 0;
	for ( const Painted &p : drv.log )
	{
		if ( p.t < 600 * MS || !p.d.newOutput )
			continue;
		n++;
		INFO( "paint " << p.t / MS << " ms" );
		CHECK( IsRendered( p.d ) );
		CHECK( p.d.prevId < p.d.currId );
		CHECK( p.d.t0 >= 0.0f );
		CHECK( p.d.t0 <= p.d.t1 );
		CHECK( p.d.t1 <= 1.0f );
	}
	CHECK( n > 60 );
	CHECK( LongestOutputGapMs( drv, 400 * MS, 2400 * MS ) <= 24.0 );
	const Pacer::Report r = drv.Status();
	CHECK( r.fgActive );
	CHECK( r.blurActive );
	CHECK( r.blurSamples == 4 );
	CHECK( r.blurWindowMs > 0.0f );
	CHECK( r.bufferDelayMs == 50.0f );
	CHECK( r.activeN == 2 );
}

TEST_CASE( "pacing: the cost guard budgets the blur samples too", "[pacing][combine]" )
{
	auto Make = []( int nSamples )
	{
		Driver drv;
		drv.refreshHz = 125.0;
		drv.frameGen = false;
		drv.blur = true;
		drv.blurSamples = nSamples;
		drv.estimateMs = 0.3f;
		drv.synthMs = 0.1f;
		drv.costSeq = 1;
		drv.Run( 1000 * MS, 5.0 );              // 200 fps: 25% of 5 ms = 1.25 ms
		return drv;
	};
	// 2 samples: 0.3 + 0.2 fits; 16 samples: 0.3 + 1.6 does not
	Driver few = Make( 2 );
	CHECK( few.pacer.Planning() );
	CHECK_FALSE( few.pacer.CostBlocked() );
	Driver many = Make( 16 );
	CHECK( many.pacer.CostBlocked() );
	CHECK_FALSE( many.pacer.Planning() );
	CHECK( many.Status().reason == Reason::CostGuard );
	CHECK( many.log.back().d.inert );
	// with the guard off the samples are only measured
	Driver off;
	off.refreshHz = 125.0;
	off.frameGen = false;
	off.blur = true;
	off.blurSamples = 16;
	off.estimateMs = 0.3f;
	off.synthMs = 0.1f;
	off.costGuard = false;
	off.Run( 1000 * MS, 5.0 );
	CHECK( off.pacer.Planning() );
}

TEST_CASE( "pacing: a reset resets all three together", "[pacing][combine]" )
{
	Driver drv;
	drv.refreshHz = 125.0;
	drv.frameGen = false;
	drv.blur = true;
	drv.extraDelayNs = 30 * MS;
	WarmUp( drv, 16.0, 1000 * MS );
	REQUIRE( drv.pacer.Planning() );
	drv.focusKey = 9;                            // another window
	const Pacer::Decision d = drv.ArriveAndPaint( drv.now + 16 * MS );
	CHECK( d.reset );
	CHECK( drv.pacer.HistoryCount() <= 1 );
	CHECK_FALSE( drv.pacer.EstimateValid() );
	// the sequence resumes after the warm-up with the same settings
	drv.Run( 1000 * MS, 16.0 );
	CHECK( drv.pacer.Planning() );
	CHECK( drv.Status().bufferDelayMs == 30.0f );
}

TEST_CASE( "pacing: switching frame generation on and off live keeps blur and the buffer", "[pacing][combine]" )
{
	Driver drv;
	drv.refreshHz = 125.0;
	drv.blur = true;
	drv.multiplier = 2;
	WarmUp( drv, 32.0, 1000 * MS );
	CHECK( drv.pacer.Generating() );
	drv.frameGen = false;
	drv.Run( 1000 * MS, 32.0 );
	CHECK_FALSE( drv.pacer.Generating() );
	CHECK( drv.pacer.RealRate() );
	CHECK( drv.Status().blurActive );
	drv.frameGen = true;
	drv.Run( 1000 * MS, 32.0 );
	CHECK( drv.pacer.Generating() );
}

// ---------------------------------------------------------------------------
//  The lag-spike buffer (LagBufferSettings): detection, judgement, recency, ramp,
//  test mode, history depth, and the combinations
// ---------------------------------------------------------------------------

namespace
{
	struct SpikeAt
	{
		double sec;       // game time of the first missing frame
		int    missing;   // real frames missing (the grid carries on after them)
	};

	// Real frames on a grid of intervalMs from 3 ms, with `missing` frames dropped at each spike.
	std::vector<Ns> SpikeArrivals( double intervalMs, Ns tEnd, const std::vector<SpikeAt> &spikes )
	{
		std::vector<Ns> v;
		for ( uint64_t k = 0;; k++ )
		{
			const Ns t = 3 * MS + Ns( double( k ) * intervalMs * double( MS ) );
			if ( t >= tEnd )
				break;
			bool bMissing = false;
			for ( const SpikeAt &sp : spikes )
			{
				const uint64_t kAt = uint64_t( sp.sec * 1000.0 / intervalMs + 0.5 );
				if ( k >= kAt && k < kAt + uint64_t( sp.missing ) )
					bMissing = true;
			}
			if ( !bMissing )
				v.push_back( t );
		}
		return v;
	}

	// Real frames on a grid of intervalMs from 3 ms; at each gap {sec, gapMs} the frame after the
	// gap comes gapMs after the one before it (a late frame, not missing ones), and the grid
	// carries on from it. For a stall of an exact length.
	std::vector<Ns> StallArrivals( double intervalMs, double endSec, const std::vector<std::pair<double, double>> &gaps )
	{
		std::vector<Ns> v;
		double t = 3.0;
		size_t g = 0;
		while ( t < endSec * 1000.0 )
		{
			v.push_back( Ns( t * 1e6 ) );
			double step = intervalMs;
			if ( g < gaps.size() && t + intervalMs > gaps[ g ].first * 1000.0 )
				step = gaps[ g++ ].second;
			t += step;
		}
		return v;
	}

	Driver LagDriver( double flMaxMs = 50.0, double flLookbackSec = 300.0 )
	{
		Driver drv;
		drv.refreshHz = 240.0;
		drv.frameGen = false;
		drv.lag.enabled = true;
		drv.lag.maxBufferMs = flMaxMs;
		drv.lag.lookbackSec = flLookbackSec;
		return drv;
	}

	// Real-rate outputs with a delay: every output is one real frame or a fill. Longest
	// stretch between outputs in [t0, t1), ms.
	double LongestGap( const Driver &drv, double sec0, double sec1 )
	{
		return LongestOutputGapMs( drv, Ns( sec0 * 1e9 ), Ns( sec1 * 1e9 ) );
	}
}

TEST_CASE( "lag buffer: off is inert and counts nothing", "[lagbuffer]" )
{
	Driver drv = LagDriver();
	drv.lag.enabled = false;
	const Ns tEnd = 30 * 1000 * MS;
	drv.RunArrivals( SpikeArrivals( 1000.0 / 60.0, tEnd, { { 10.0, 2 }, { 20.0, 2 } } ), tEnd );
	CHECK( drv.pacer.BufferMs() == 0.0 );
	CHECK( drv.pacer.HistoryDepth() == kHistory );
	const Pacer::Report r = drv.Status();
	CHECK( r.bufferDelayMs == 0.0f );
	CHECK( r.bufferTargetMs == 0.0f );
	CHECK( r.spikesInWindow == 0 );
	CHECK( r.historyFrames == kHistory );
	CHECK( r.reason == Reason::Off );
	for ( const Painted &p : drv.log )
		CHECK( p.d.inert );                  // a pass-through, as without the module
}

TEST_CASE( "lag buffer: a two-frame spike grows the buffer to spike + margin and the next one is filled", "[lagbuffer]" )
{
	Driver drv = LagDriver();
	const double flI = 1000.0 / 60.0;
	const Ns tEnd = 30 * 1000 * MS;
	drv.RunArrivals( SpikeArrivals( flI, tEnd, { { 10.0, 2 }, { 20.0, 2 } } ), tEnd );
	// the first spike freezes (nothing known yet): one long gap
	CHECK( LongestGap( drv, 9.5, 12.0 ) > 1.5 * flI );
	// ... and sized the buffer: spike 33.3 ms x 1.1 + 1 ms = 37.7 ms (the exact value the
	// second spike, in the same place, then re-confirms)
	INFO( "buffer " << drv.pacer.BufferMs() << " ms" );
	CHECK( drv.pacer.BufferMs() > 35.0 );
	CHECK( drv.pacer.BufferMs() < 40.0 );
	// the second one is covered: no output gap above 1.5 intervals, the gap filled by synths
	INFO( "longest gap over the second spike " << LongestGap( drv, 19.5, 22.0 ) << " ms" );
	CHECK( LongestGap( drv, 19.5, 22.0 ) <= 1.5 * flI );
	CHECK( CountRenderedSynths( drv, Ns( 19.9e9 ), Ns( 21.0e9 ) ) >= 2 );
	// the report
	const Pacer::Report r = drv.Status();
	CHECK( r.bufferDelayMs > 35.0f );
	CHECK( r.bufferTargetMs > 35.0f );
	CHECK( r.spikesInWindow == 2 );
	CHECK( r.outliersIgnored == 0 );
	CHECK( std::fabs( r.lastSpikeMs - 33.3f ) < 1.5f );
	CHECK( r.delayMs == r.bufferDelayMs );
}

TEST_CASE( "lag buffer: ordinary jitter is not a spike", "[lagbuffer]" )
{
	Driver drv = LagDriver();
	std::vector<Ns> v;
	Ns t = 3 * MS;
	for ( int k = 0; k < 1500; k++ )
	{
		v.push_back( t );
		t += Ns( ( 16.667 + ( k % 3 == 0 ? 4.0 : k % 3 == 1 ? -3.0 : 1.0 ) ) * double( MS ) );   // 60 fps +-4 ms
	}
	drv.RunArrivals( v, v.back() + MS );
	CHECK( drv.Status().spikesInWindow == 0 );
	CHECK( drv.pacer.BufferMs() == 0.0 );
}

TEST_CASE( "lag buffer: a compile-style drop is an outlier and builds no buffer", "[lagbuffer]" )
{
	const double flI = 1000.0 / 60.0;
	// 6 frames missing: a 100 ms spike (> the 50 ms max, under the stall reset)
	{
		Driver drv = LagDriver( 50.0 );
		const Ns tEnd = 20 * 1000 * MS;
		drv.RunArrivals( SpikeArrivals( flI, tEnd, { { 10.0, 6 } } ), tEnd );
		const Pacer::Report r = drv.Status();
		CHECK( r.outliersIgnored == 1 );
		CHECK( r.spikesInWindow == 0 );
		CHECK( drv.pacer.BufferMs() == 0.0 );
		CHECK( r.bufferTargetMs == 0.0f );
		CHECK( r.lastSpikeMs > 80.0f );              // logged all the same
	}
	// 12 frames missing: 200 ms, a stall that resets the history (the default 100 ms
	// reset lifts to max + 50 ms with the buffer on, so 200 ms is a reset at max 50)
	{
		Driver drv = LagDriver( 50.0 );
		const Ns tEnd = 20 * 1000 * MS;
		drv.RunArrivals( SpikeArrivals( flI, tEnd, { { 10.0, 12 } } ), tEnd );
		const Pacer::Report r = drv.Status();
		CHECK( r.outliersIgnored == 1 );
		CHECK( drv.pacer.BufferMs() == 0.0 );
	}
	// the same 100 ms drop with a 250 ms max is a spike like any other, and is sized for
	{
		Driver drv = LagDriver( 250.0 );
		const Ns tEnd = 20 * 1000 * MS;
		drv.RunArrivals( SpikeArrivals( flI, tEnd, { { 10.0, 6 } } ), tEnd );
		CHECK( drv.Status().spikesInWindow == 1 );
		CHECK( drv.pacer.BufferMs() > 100.0 );
	}
}

TEST_CASE( "lag buffer: spikes every 150 s keep the buffer steady inside a 300 s window", "[lagbuffer]" )
{
	Driver drv = LagDriver( 50.0, 300.0 );
	const double flI = 1000.0 / 60.0;
	const Ns tEnd = 700 * 1000 * MS;
	drv.RunArrivals( SpikeArrivals( flI, tEnd, { { 30, 2 }, { 180, 2 }, { 330, 2 }, { 480, 2 }, { 630, 2 } } ), tEnd );
	double flMin = 1e9, flMax = 0.0;
	for ( const Painted &p : drv.log )
	{
		if ( p.t < Ns( 200e9 ) )
			continue;                                  // the second spike's ramp is done by 200 s
		flMin = std::min( flMin, p.bufMs );
		flMax = std::max( flMax, p.bufMs );
	}
	INFO( "buffer range after 200 s: " << flMin << " .. " << flMax << " ms" );
	CHECK( flMin > 15.0 );                         // never back to zero between spikes
	CHECK( flMax < 40.0 );
	// ... the recurrent ones are covered: no gap above 1.5 intervals over the last spike
	CHECK( LongestGap( drv, 629.5, 632.0 ) <= 1.5 * flI );
}

TEST_CASE( "lag buffer: a single spike decays to nothing after the look-back window", "[lagbuffer]" )
{
	Driver drv = LagDriver( 50.0, 60.0 );
	const double flI = 1000.0 / 60.0;
	drv.RunArrivals( SpikeArrivals( flI, 15 * 1000 * MS, { { 10.0, 2 } } ), 15 * 1000 * MS );
	CHECK( drv.pacer.BufferMs() > 35.0 );
	CHECK( drv.pacer.HistoryDepth() >= 5 );
	// inside the window it only fades (to 30 % at the very end), it is not dropped
	drv.Run( 53 * 1000 * MS, flI );
	CHECK( drv.pacer.BufferMs() > 5.0 );
	// past it (10 s + 60 s) the target is 0, and a 2 s wait + a ramp at 3 % later so is the buffer
	drv.Run( 12 * 1000 * MS, flI );
	INFO( "buffer " << drv.pacer.BufferMs() << " ms, depth " << drv.pacer.HistoryDepth() );
	CHECK( drv.pacer.BufferMs() == 0.0 );
	CHECK( drv.pacer.HistoryDepth() == kHistory );
	CHECK( drv.Status().spikesInWindow == 0 );
}

TEST_CASE( "lag buffer: a stall grows the delay at once, everything else by the ramp, the content time never goes back", "[lagbuffer]" )
{
	Driver drv = LagDriver();
	const double flI = 1000.0 / 60.0;
	const Ns tEnd = 25 * 1000 * MS;
	drv.RunArrivals( SpikeArrivals( flI, tEnd, { { 10.0, 2 } } ), tEnd );
	double flMaxRate = 0.0;
	double flJump = 0.0;
	int nJumps = 0, nGrowth = 0;
	for ( size_t i = 1; i < drv.log.size(); i++ )
	{
		const Painted &a = drv.log[ i - 1 ], &b = drv.log[ i ];
		if ( b.t <= a.t )
			continue;
		const double flRate = ( b.bufMs - a.bufMs ) / ( double( b.t - a.t ) / 1e6 );
		if ( flRate > kLagRampEps * 1.001 )
		{
			nJumps++;                               // growth beyond the ramp: only the stall credit
			flJump = b.bufMs - a.bufMs;
			CHECK( b.t > Ns( 10e9 ) );
			CHECK( b.t < Ns( 10.2e9 ) );            // on the paint that follows the late frame
		}
		else
		{
			flMaxRate = std::max( flMaxRate, std::fabs( flRate ) );
		}
		if ( b.bufMs > a.bufMs )
			nGrowth++;
		if ( a.tau != 0 && b.tau != 0 )
			CHECK( b.tau >= a.tau );                // never backwards
	}
	INFO( "the jump " << flJump << " ms, the fastest other change " << flMaxRate * 100.0 << " % of real time" );
	// Why: the display stood still for the stall, so the buffer is credited what it was short
	// by (the spike's size, 33.3 ms from nothing) on the very paint, and the ramp only
	// finishes the margin (spike x 1.1 + 1 ms = 37.7 ms) at 3 % of real time.
	CHECK( nJumps == 1 );
	CHECK( flJump > 33.0 );
	CHECK( flJump < 36.0 );
	CHECK( flMaxRate <= kLagRampEps * 1.001 );
	CHECK( nGrowth > 20 );                         // the margin did ramp, a little per paint
	CHECK( drv.pacer.BufferMs() > 37.0 );
	CHECK( drv.pacer.BufferMs() < 38.5 );
	// no output jump after the stall (the first spike's own freeze is the stall itself)
	CHECK( LongestGap( drv, 11.5, 24.0 ) <= 1.5 * flI );
	// shown frames never go back
	uint64_t ulShown = 0;
	for ( const Painted &p : drv.log )
	{
		if ( !p.d.newOutput )
			continue;
		const uint64_t ulHi = p.d.prevId != 0 ? p.d.currId : p.d.showId;
		CHECK( ulHi >= ulShown );
		ulShown = std::max( ulShown, ulHi );
	}
}

TEST_CASE( "lag buffer: growth with no stall is still the 3 % ramp (test mode, a raised max)", "[lagbuffer]" )
{
	const double flI = 1000.0 / 60.0;
	// ForceMax from nothing: no spike, so no credit: never faster than 3 %, tau monotonic
	{
		Driver drv = LagDriver();
		drv.lag.testMode = LagTestMode::ForceMax;
		drv.Run( 5 * 1000 * MS, flI );
		double flMaxRate = 0.0;
		int nGrowth = 0;
		for ( size_t i = 1; i < drv.log.size(); i++ )
		{
			const Painted &a = drv.log[ i - 1 ], &b = drv.log[ i ];
			if ( b.t <= a.t )
				continue;
			flMaxRate = std::max( flMaxRate, ( b.bufMs - a.bufMs ) / ( double( b.t - a.t ) / 1e6 ) );
			nGrowth += b.bufMs > a.bufMs ? 1 : 0;
			if ( a.tau != 0 && b.tau != 0 )
				CHECK( b.tau >= a.tau );
		}
		INFO( "fastest growth " << flMaxRate * 100.0 << " %" );
		CHECK( flMaxRate <= kLagRampEps * 1.001 );
		CHECK( nGrowth > 100 );
		CHECK( drv.pacer.BufferMs() == 50.0 );
		CHECK( LongestGap( drv, 0.5, 5.0 ) <= 1.5 * flI );
	}
	// a raised max after a spike sized the buffer at the old cap: the rest of the way is the ramp
	{
		Driver drv = LagDriver( 20.0 );
		drv.RunArrivals( SpikeArrivals( flI, 8 * 1000 * MS, { { 3.0, 3 } } ), 8 * 1000 * MS );   // 50 ms: above the 20 ms max, an outlier
		CHECK( drv.pacer.BufferMs() == 0.0 );
		drv.lag.maxBufferMs = 100.0;
		drv.lag.testMode = LagTestMode::ForceMax;
		drv.Run( 100 * MS, flI );
		CHECK( drv.pacer.BufferMs() < 5.0 );                // not a jump to the new cap
	}
}

TEST_CASE( "lag buffer: test mode pins the buffer at its max or at zero, through the ramp", "[lagbuffer]" )
{
	Driver drv = LagDriver( 250.0 );
	const double flI = 1000.0 / 60.0;
	drv.lag.testMode = LagTestMode::ForceMax;
	drv.Run( 4 * 1000 * MS, flI );
	// 3 % of real time: 4 s reach about 120 ms, not 250
	CHECK( drv.pacer.BufferMs() > 100.0 );
	CHECK( drv.pacer.BufferMs() < 130.0 );
	CHECK( drv.pacer.BufferTargetMs() == 250.0 );
	drv.Run( 6 * 1000 * MS, flI );
	CHECK( drv.pacer.BufferMs() == 250.0 );
	// 250 ms at 60 fps: 15 frames + the margin, about 17
	INFO( "depth " << drv.pacer.HistoryDepth() );
	CHECK( drv.pacer.HistoryDepth() >= 16 );
	CHECK( drv.pacer.HistoryDepth() <= 18 );
	CHECK( drv.Status().historyFrames == drv.pacer.HistoryDepth() );
	CHECK( drv.Status().bufferDelayMs == 250.0f );
	// with no spike at all the buffer is still the user's to feel; the real stream is
	// still delayed by exactly that: output = arrival + 250 ms
	const Painted &last = drv.log.back();
	CHECK( last.delayMs == 250.0 );

	// ForceMin: back to zero by the same ramp, and the history follows after its wait
	drv.lag.testMode = LagTestMode::ForceMin;
	drv.Run( 2 * 1000 * MS, flI );
	CHECK( drv.pacer.BufferMs() < 250.0 );          // not at once
	CHECK( drv.pacer.BufferMs() > 100.0 );
	drv.Run( 14 * 1000 * MS, flI );
	CHECK( drv.pacer.BufferMs() == 0.0 );
	CHECK( drv.pacer.HistoryDepth() == kHistory );
	// and a test mode with the buffer off does nothing
	Driver off = LagDriver( 250.0 );
	off.lag.enabled = false;
	off.lag.testMode = LagTestMode::ForceMax;
	off.Run( 5 * 1000 * MS, flI );
	CHECK( off.pacer.BufferMs() == 0.0 );
}

TEST_CASE( "lag buffer: the history depth follows the buffer, ahead of the ramp", "[lagbuffer]" )
{
	Driver drv = LagDriver();
	const double flI = 1000.0 / 60.0;
	const Ns tEnd = 70 * 1000 * MS;
	drv.lag.lookbackSec = 20.0;
	drv.RunArrivals( SpikeArrivals( flI, tEnd, { { 10.0, 2 } } ), tEnd );
	int nMax = 0;
	for ( const Painted &p : drv.log )
	{
		nMax = std::max( nMax, p.depth );
		if ( p.t < Ns( 5e9 ) )
			CHECK( p.depth == kHistory );             // nothing before the first spike
		// it never lags behind the buffer it has to cover
		CHECK( p.depth >= int( std::ceil( p.bufMs / flI - 0.05 ) ) + kHistoryMargin );
		// and a synth never reaches for a frame outside the kept window
		if ( p.d.newOutput && p.d.prevId != 0 )
			CHECK( p.d.prevId + uint64_t( p.depth ) > p.d.newestId );
	}
	INFO( "peak depth " << nMax );
	CHECK( nMax >= 5 );
	CHECK( nMax <= 6 );
	CHECK( drv.pacer.HistoryDepth() == kHistory );     // decayed back after the window + the wait
	// the shrink waited: the depth did not drop while the buffer was still above what it needs
	for ( const Painted &p : drv.log )
		if ( p.t > Ns( 12e9 ) && p.t < Ns( 16e9 ) )
			CHECK( p.depth >= 5 );
	// the ceiling covers 250 ms up to 250 fps (above that the buffer's cap shrinks, see "[highfps]")
	CHECK( kHistoryMax >= int( std::ceil( kLagMaxBufferMs / 4.0 ) ) + kHistoryMargin );
}

TEST_CASE( "lag buffer: the buffer never exceeds what the history can hold at a fast game rate", "[lagbuffer]" )
{
	Driver drv = LagDriver( 250.0 );
	drv.lag.testMode = LagTestMode::ForceMax;
	drv.Run( 12 * 1000 * MS, 5.0 );                  // 200 fps: 62 frames x 5 ms = 310 ms > 250
	CHECK( drv.pacer.BufferMs() == 250.0 );
	CHECK( drv.pacer.HistoryDepth() <= kHistoryMax );
	CHECK( drv.pacer.HistoryDepth() >= 50 );
	Driver fast = LagDriver( 250.0 );
	fast.lag.testMode = LagTestMode::ForceMax;
	fast.Run( 12 * 1000 * MS, 4.0 );                 // 250 fps
	CHECK( fast.pacer.BufferMs() <= 250.0 );
	CHECK( fast.pacer.HistoryDepth() <= kHistoryMax );
	CHECK( fast.pacer.BufferMs() >= 240.0 );
}

TEST_CASE( "lag buffer: with frame generation 2x the gap is filled too", "[lagbuffer][combine]" )
{
	const double flI = 1000.0 / 60.0;
	const Ns tEnd = 30 * 1000 * MS;
	Driver drv = LagDriver();
	drv.frameGen = true;
	drv.multiplier = 2;
	drv.RunArrivals( SpikeArrivals( flI, tEnd, { { 10.0, 2 }, { 20.0, 2 } } ), tEnd );
	REQUIRE( drv.pacer.Generating() );
	CHECK( drv.pacer.BufferMs() > 35.0 );
	INFO( "longest gap " << LongestGap( drv, 19.5, 22.0 ) << " ms" );
	CHECK( LongestGap( drv, 19.5, 22.0 ) <= 1.5 * flI );
	// the same stream with FG alone freezes at the second spike
	Driver plain;
	plain.refreshHz = 240.0;
	plain.multiplier = 2;
	plain.RunArrivals( SpikeArrivals( flI, tEnd, { { 10.0, 2 }, { 20.0, 2 } } ), tEnd );
	CHECK( LongestGap( plain, 19.5, 22.0 ) > 1.5 * flI );
	// FG's cadence is untouched by the buffer: the same number of outputs a second once settled
	const double flRateBuf = double( drv.CountOutputs( 24 * 1000 * MS, 28 * 1000 * MS ) ) / 4.0;
	const double flRatePlain = double( plain.CountOutputs( 24 * 1000 * MS, 28 * 1000 * MS ) ) / 4.0;
	CHECK( std::fabs( flRateBuf - flRatePlain ) < 3.0 );
	CHECK( drv.Status().fgActive );
	CHECK( drv.Status().activeN == 2 );
	// target-fps mode as well
	Driver tgt = LagDriver();
	tgt.frameGen = true;
	tgt.mode = Mode::Target;
	tgt.targetFps = 120;
	tgt.RunArrivals( SpikeArrivals( flI, tEnd, { { 10.0, 2 }, { 20.0, 2 } } ), tEnd );
	CHECK( LongestGap( tgt, 19.5, 22.0 ) <= 1.5 * flI );
}

TEST_CASE( "lag buffer: with motion blur, and with all three", "[lagbuffer][combine]" )
{
	const double flI = 1000.0 / 60.0;
	const Ns tEnd = 30 * 1000 * MS;
	for ( int nFg = 0; nFg < 2; nFg++ )
	{
		Driver drv = LagDriver();
		drv.frameGen = nFg == 1;
		drv.multiplier = 2;
		drv.blur = true;
		drv.blurSamples = 4;
		drv.blurAmount = 0.5;
		drv.RunArrivals( SpikeArrivals( flI, tEnd, { { 10.0, 2 }, { 20.0, 2 } } ), tEnd );
		INFO( ( nFg ? "all three" : "buffer + blur" ) << ": longest gap " << LongestGap( drv, 19.5, 22.0 ) << " ms" );
		CHECK( LongestGap( drv, 19.5, 22.0 ) <= 1.5 * flI );
		const Pacer::Report r = drv.Status();
		CHECK( r.blurActive );
		CHECK( r.fgActive == ( nFg == 1 ) );
		CHECK( r.bufferDelayMs > 35.0f );
		int n = 0;
		for ( const Painted &p : drv.log )
		{
			if ( p.t < Ns( 15e9 ) || !p.d.newOutput )
				continue;
			n++;
			CHECK( p.d.t0 >= 0.0f );
			CHECK( p.d.t0 <= p.d.t1 );
			CHECK( p.d.t1 <= 1.0f );
			if ( p.d.prevId != 0 )
				CHECK( p.d.prevId < p.d.currId );
		}
		CHECK( n > 100 );
		CHECK( CountRenderedSynths( drv, Ns( 19.9e9 ), Ns( 21e9 ) ) >= 2 );
	}
}

TEST_CASE( "lag buffer: a new stream forgets the spikes, a stall does not", "[lagbuffer]" )
{
	Driver drv = LagDriver();
	const double flI = 1000.0 / 60.0;
	drv.RunArrivals( SpikeArrivals( flI, 15 * 1000 * MS, { { 10.0, 2 } } ), 15 * 1000 * MS );
	CHECK( drv.pacer.BufferMs() > 30.0 );
	drv.focusKey = 7;                                 // another window
	drv.ArriveAndPaint( drv.now + 16 * MS );
	CHECK( drv.pacer.BufferMs() == 0.0 );
	CHECK( drv.Status().spikesInWindow == 0 );
}

// ---------------------------------------------------------------------------
//  The stall credit: a spike the buffer was too small for grows it at once; a smaller
//  hitch is bridged; a lower detection threshold; the limits that stay
// ---------------------------------------------------------------------------
//
// The user: "it should still detect lags and based on that determine the buffer size, but
// otherwise it should make sure that for example we have a buffer of 50 and usually we get
// a frame every 10 milliseconds, but then there isn't a frame for 14 milliseconds, then it
// should just generate the frames for that as soon as it can based on the next frame, if
// the lag spike was bigger than the buffer, then it obviously can't do that. But then it
// should increase the buffer and then it should start working again". Measured before
// the credit (100 fps, max 50, 60 ms stalls at 3, 3.5, 4 and 5 s from no buffer): the
// output froze for 58, 42, 29 and 0 ms -- the 3 % ramp had not caught up -- so the buffer
// "waited". The first stall's freeze is unavoidable (nothing was known); the rest are not.

namespace
{
	// The first paint at or after tSec.
	const Painted *PaintAtOrAfter( const Driver &drv, double tSec )
	{
		for ( const Painted &p : drv.log )
			if ( p.t >= Ns( tSec * 1e9 ) )
				return &p;
		return nullptr;
	}

	// The paint before the first one at or after tSec.
	const Painted *PaintBefore( const Driver &drv, double tSec )
	{
		const Painted *pPrev = nullptr;
		for ( const Painted &p : drv.log )
		{
			if ( p.t >= Ns( tSec * 1e9 ) )
				return pPrev;
			pPrev = &p;
		}
		return nullptr;
	}
}

TEST_CASE( "lag buffer: a hiccup shorter than the buffer is bridged by generated frames, t about 0.71", "[lagbuffer][stall]" )
{
	// 100 fps, a 50 ms buffer (test mode, settled), then one frame 14 ms after the last
	Driver drv = LagDriver();
	drv.lag.testMode = LagTestMode::ForceMax;
	drv.RunArrivals( StallArrivals( 10.0, 8.0, { { 6.0, 14.0 } } ), 8000 * MS );
	REQUIRE( drv.pacer.BufferMs() == 50.0 );
	// the 4 ms hiccup is detected, and (4 ms < D) earns no credit
	CHECK( drv.Status().spikesInWindow == 1 );
	CHECK( drv.pacer.BufferMs() == 50.0 );
	INFO( "longest gap " << LongestGap( drv, 5.95, 6.2 ) << " ms" );
	CHECK( LongestGap( drv, 5.95, 6.2 ) <= 1.5 * 10.0 );
	// a synth sits at 10/14 of the pair that brackets the hiccup, and then the real frame
	bool bT71 = false;
	for ( const Painted &p : drv.log )
		if ( p.t > Ns( 5.99e9 ) && p.t < Ns( 6.2e9 ) && p.d.newOutput && p.d.prevId != 0 && std::fabs( p.d.t - 0.714f ) < 0.03f )
			bT71 = true;
	CHECK( bT71 );
	// the content time and the shown frames stay monotonic
	int64_t tauPrev = 0;
	uint64_t ulShown = 0;
	for ( const Painted &p : drv.log )
	{
		if ( p.t < Ns( 5.0e9 ) )
			continue;
		if ( p.tau != 0 )
		{
			CHECK( p.tau >= tauPrev );
			tauPrev = p.tau;
		}
		if ( p.d.newOutput )
		{
			const uint64_t ulHi = p.d.prevId != 0 ? p.d.currId : p.d.showId;
			CHECK( ulHi >= ulShown );
			ulShown = std::max( ulShown, ulHi );
		}
	}
}

TEST_CASE( "lag buffer: a spike larger than the buffer stalls once, grows it at once, and the repeat is covered", "[lagbuffer][stall]" )
{
	// a 40 ms interval at 2 s (size 30) sizes the buffer at 34 ms; a 60 ms one at 5 s (size 50)
	// then finds D = 34: it stalls for about (size - D) + one interval, and the buffer is at 50
	// (the max) on the very paint after it, not 1.7 s later
	Driver drv = LagDriver();
	drv.RunArrivals( StallArrivals( 10.0, 8.0, { { 2.0, 40.0 }, { 5.0, 60.0 }, { 5.5, 60.0 } } ), 8000 * MS );
	const Painted *pBefore = PaintBefore( drv, 5.0 );
	REQUIRE( pBefore );
	INFO( "D before the spike " << pBefore->bufMs << " ms" );
	CHECK( pBefore->bufMs > 32.0 );
	CHECK( pBefore->bufMs < 35.0 );
	const double flStall = LongestGap( drv, 4.99, 5.2 );
	INFO( "stall " << flStall << " ms" );
	CHECK( flStall > 15.0 );                       // it did stall: nothing was known to cover 50 ms
	CHECK( flStall < 31.0 );                       // for (60 - 34) = 26 ms, not the 60 ms of the gap
	// the paint after the late frame: D is already at the spike's size or more
	const Painted *pAfter = PaintAtOrAfter( drv, 5.06 );
	REQUIRE( pAfter );
	CHECK( pAfter->bufMs >= 49.9 );
	CHECK( pAfter->bufMs <= 50.0 );
	// the repeat 0.5 s later: no gap above 1.5 intervals (the 240 Hz tick rounds 10 ms to 12.5)
	INFO( "repeat " << LongestGap( drv, 5.49, 5.7 ) << " ms" );
	CHECK( LongestGap( drv, 5.49, 5.7 ) <= 1.5 * 10.0 );
	CHECK( CountRenderedSynths( drv, Ns( 5.55e9 ), Ns( 5.65e9 ) ) >= 2 );
}

TEST_CASE( "lag buffer: 60 ms stalls at 3, 3.5, 4 and 5 s from nothing: only the first freezes", "[lagbuffer][stall]" )
{
	Driver drv = LagDriver();
	drv.RunArrivals( StallArrivals( 10.0, 7.0, { { 3.0, 60.0 }, { 3.5, 60.0 }, { 4.0, 60.0 }, { 5.0, 60.0 } } ), 7000 * MS );
	// before: 58 / 42 / 29 / 0 ms of freeze (the buffer ramped at 3 % and was never there in time)
	CHECK( LongestGap( drv, 2.99, 3.3 ) > 40.0 );
	for ( double flAt : { 3.5, 4.0, 5.0 } )
	{
		INFO( "stall at " << flAt << " s: longest gap " << LongestGap( drv, flAt - 0.01, flAt + 0.2 ) << " ms" );
		CHECK( LongestGap( drv, flAt - 0.01, flAt + 0.2 ) <= 1.5 * 10.0 );
	}
	CHECK( drv.pacer.BufferMs() == 50.0 );
}

TEST_CASE( "lag buffer: a 14 ms gap at 100 fps is a spike (1.3x) and is bridged next time; ordinary jitter and a fast game's noise are not", "[lagbuffer][stall][threshold]" )
{
	Driver drv = LagDriver();
	drv.RunArrivals( StallArrivals( 10.0, 8.0, { { 2.0, 14.0 }, { 3.0, 14.0 }, { 4.0, 14.0 }, { 5.0, 14.0 }, { 6.0, 14.0 } } ), 8000 * MS );
	const Pacer::Report r = drv.Status();
	CHECK( r.spikesInWindow == 5 );
	CHECK( r.outliersIgnored == 0 );
	INFO( "buffer " << drv.pacer.BufferMs() );
	CHECK( drv.pacer.BufferMs() > 5.0 );           // 4 ms x 1.1 + 1 ms = 5.4
	CHECK( drv.pacer.BufferMs() < 6.0 );
	CHECK( LongestGap( drv, 5.99, 6.1 ) <= 1.5 * 10.0 );
	// the first one's buffer came at once (a 4 ms stall, credited 4 ms)
	const Painted *pFirst = PaintAtOrAfter( drv, 2.02 );
	REQUIRE( pFirst );
	CHECK( pFirst->bufMs >= 3.9 );

	// no jitter floor (the user's choice): the jitter that is NOT a spike is excluded by the
	// ratio and the minimum excess alone. 60 fps +-4 ms peaks at 1.24x ("ordinary jitter ..."
	// above); a 240 fps game's 4.2 ms interval with a 5.6 ms one is 1.34x but only 1.4 ms over
	{
		Driver fast = LagDriver();
		std::vector<Ns> v;
		Ns t = 3 * MS;
		for ( int k = 0; k < 6000; k++ )
		{
			v.push_back( t );
			t += Ns( ( k % 10 == 5 ? 5.6 : 4.17 ) * double( MS ) );
		}
		fast.RunArrivals( v, v.back() + MS );
		CHECK( fast.Status().spikesInWindow == 0 );
		CHECK( fast.pacer.BufferMs() == 0.0 );
	}
	// ... while 2 ms over IS one: the same game with a 6.4 ms interval
	{
		Driver fast = LagDriver();
		std::vector<Ns> v;
		Ns t = 3 * MS;
		for ( int k = 0; k < 6000; k++ )
		{
			v.push_back( t );
			t += Ns( ( k % 100 == 50 ? 6.4 : 4.17 ) * double( MS ) );
		}
		fast.RunArrivals( v, v.back() + MS );
		CHECK( fast.Status().spikesInWindow >= 1 );
	}
}

TEST_CASE( "lag buffer: a spike above the max is still an outlier: no credit, no buffer, counted", "[lagbuffer][stall]" )
{
	Driver drv = LagDriver( 50.0 );
	// a 40 ms interval sizes D at ~34; a 70 ms one (size 60 > 50) must not move it
	drv.RunArrivals( StallArrivals( 10.0, 8.0, { { 2.0, 40.0 }, { 5.0, 70.0 } } ), 8000 * MS );
	const Painted *pBefore = PaintBefore( drv, 5.0 );
	REQUIRE( pBefore );
	const Painted *pAfter = PaintAtOrAfter( drv, 5.1 );
	REQUIRE( pAfter );
	INFO( "before " << pBefore->bufMs << " after " << pAfter->bufMs );
	CHECK( pAfter->bufMs <= pBefore->bufMs + 0.1 );
	for ( const Painted &p : drv.log )
		if ( p.t > Ns( 5.0e9 ) && p.t < Ns( 5.2e9 ) )
			CHECK( p.bufMs <= pBefore->bufMs + 0.1 );
	const Pacer::Report r = drv.Status();
	CHECK( r.outliersIgnored == 1 );
	CHECK( r.spikesInWindow == 1 );
	// from nothing too: D stays 0
	Driver none = LagDriver( 50.0 );
	none.RunArrivals( StallArrivals( 10.0, 6.0, { { 3.0, 70.0 } } ), 6000 * MS );
	CHECK( none.pacer.BufferMs() == 0.0 );
	CHECK( none.Status().outliersIgnored == 1 );
}

TEST_CASE( "lag buffer: a spike inside the look-back that has faded is credited again, up to its size", "[lagbuffer][stall]" )
{
	// a 40 ms interval (size 30) at 10 s in a 300 s window: by 170 s the weight has fallen to
	// 0.6 and the buffer shrank with it; the same stall at 180 s finds D too small and grows it at once
	Driver drv = LagDriver( 50.0, 300.0 );
	drv.RunArrivals( StallArrivals( 10.0, 190.0, { { 10.0, 40.0 }, { 180.0, 40.0 } } ), 190000 * MS );
	const Painted *pBefore = PaintBefore( drv, 180.0 );
	REQUIRE( pBefore );
	INFO( "D before the repeat " << pBefore->bufMs );
	CHECK( pBefore->bufMs < 28.0 );                // decayed below the spike's size
	CHECK( pBefore->bufMs > 10.0 );                // but inside the window it is never dropped
	const Painted *pAfter = PaintAtOrAfter( drv, 180.05 );
	REQUIRE( pAfter );
	INFO( "D after " << pAfter->bufMs );
	CHECK( pAfter->bufMs >= 29.9 );                // back to the spike's size at once
	CHECK( pAfter->bufMs <= 35.0 );
}

TEST_CASE( "lag buffer: with frame generation the credit counts the whole delay and never over-grows", "[lagbuffer][stall][combine]" )
{
	const double flI = 1000.0 / 60.0;
	for ( int nMode = 0; nMode < 3; nMode++ )
	{
		Driver drv = LagDriver();
		drv.frameGen = true;
		drv.multiplier = 2;
		if ( nMode == 1 )
		{
			drv.mode = Mode::Target;
			drv.targetFps = 120;
		}
		if ( nMode == 2 )
			drv.priority = Priority::Smoothness;
		// a 50 ms interval at 3 s: size 50 - 16.7 = 33.3
		drv.RunArrivals( StallArrivals( flI, 6.0, { { 3.0, 50.0 } } ), 6000 * MS );
		const double flSize = 50.0 - flI;
		const Painted *pBefore = PaintBefore( drv, 3.0 );
		REQUIRE( pBefore );
		// on the first paint that grows it, the buffer grows by (size - the whole delay), plus at most
		// one ramp step (3 % of a capped 100 ms paint): the credit counts the base delay too, so it
		// does not put a whole size on top of a delay that already covers part of it
		const Painted *pGrown = nullptr;
		for ( const Painted &p : drv.log )
			if ( p.t > Ns( 3.0e9 ) && p.bufMs > pBefore->bufMs + 0.01 )
			{
				pGrown = &p;
				break;
			}
		REQUIRE( pGrown );
		const double flGrowth = pGrown->bufMs - pBefore->bufMs;
		INFO( "mode " << nMode << ": D before " << pBefore->delayMs << " (buffer " << pBefore->bufMs << "), buffer after " << pGrown->bufMs
			<< ", growth " << flGrowth << ", size " << flSize );
		CHECK( flGrowth <= flSize - pBefore->delayMs + kLagRampEps * 100.0 + 0.1 );
		CHECK( flGrowth >= 0.5 * ( flSize - pBefore->delayMs ) );
		// the history follows it
		for ( const Painted &p : drv.log )
			if ( p.t > Ns( 2.9e9 ) )
				CHECK( p.depth >= int( std::ceil( p.bufMs / flI - 0.05 ) ) + kHistoryMargin );
		// settled, the buffer is the target the spike always sized (x 1.1 + 1 ms), not more
		const Painted *pLate = PaintAtOrAfter( drv, 4.0 );
		REQUIRE( pLate );
		INFO( "settled buffer " << pLate->bufMs );
		CHECK( pLate->bufMs <= flSize * ( 1.0 + kLagMarginFrac ) + kLagMarginMs + 0.01 );
		CHECK( pLate->bufMs >= flSize * ( 1.0 + kLagMarginFrac ) + kLagMarginMs - 0.5 );
		CHECK( drv.pacer.Generating() );
	}
}

TEST_CASE( "lag buffer: switching it off drops the buffer at once even with a stall credit pending", "[lagbuffer][stall][instantoff]" )
{
	Driver drv = LagDriver();
	drv.RunArrivals( StallArrivals( 10.0, 5.0, { { 2.0, 40.0 } } ), 5000 * MS );
	REQUIRE( drv.pacer.BufferMs() > 30.0 );
	// the late frame arrives (the credit is noted) and the buffer is off on the paint that follows
	drv.lag.enabled = false;
	drv.ArriveAndPaint( drv.now + 80 * MS );
	CHECK( drv.log.back().bufMs == 0.0 );
	CHECK( drv.log.back().delayMs == 0.0 );
	drv.Run( 300 * MS, 10.0 );
	CHECK( drv.pacer.BufferMs() == 0.0 );
	// switched back on: from zero, by the ramp -- the discarded credit does not come back
	drv.lag.enabled = true;
	drv.Run( 100 * MS, 10.0 );
	INFO( "buffer " << drv.pacer.BufferMs() );
	CHECK( drv.pacer.BufferMs() < 5.0 );
}

TEST_CASE( "lag buffer: a stall at 400 fps is credited only up to what the history can hold", "[lagbuffer][stall][highfps]" )
{
	// 2.5 ms interval, max 250: the history caps the buffer at (65 - 3 - 2) x 2.5 = 150 ms. A 200 ms
	// stall (size 197.5, within the max, above the cap) jumps the buffer to the cap and no further.
	Driver drv = LagDriver( 250.0 );
	drv.RunArrivals( StallArrivals( 2.5, 8.0, { { 4.0, 200.0 } } ), 8000 * MS );
	double flMax = 0.0;
	int nDepthMax = 0;
	for ( const Painted &p : drv.log )
	{
		flMax = std::max( flMax, p.bufMs );
		nDepthMax = std::max( nDepthMax, p.depth );
	}
	const double flCap = double( kHistoryMax - kHistory - kHistoryMargin ) * 2.5;
	INFO( "peak buffer " << flMax << " cap " << flCap << ", peak depth " << nDepthMax );
	CHECK( flMax <= flCap + 1e-6 );
	CHECK( flMax >= flCap - 1.0 );
	CHECK( nDepthMax <= kHistoryMax );
	const Painted *pAfter = PaintAtOrAfter( drv, 4.3 );
	REQUIRE( pAfter );
	CHECK( pAfter->bufMs >= flCap - 1.0 );
}

// ======================================================================
//  Video / timestamps mode (Pacer::OnFrame)
// ======================================================================

namespace
{
	// A host's video loop: frames with PTS (ns) announced a little ahead of the clock, one
	// paint per present tick at V = tick time. Media clock: V = offset + pts / rate.
	struct VideoDriver
	{
		Pacer pacer;
		double refreshHz = 144.0;
		double fps = 24.0;             // pts_k = llround( k * 1e9 / fps )
		double rate = 1.0;
		Mode mode = Mode::Target;      // target 0 = every tick of the display
		int multiplier = 2;
		int targetFps = 0;
		bool frameGen = true;
		bool blur = false;
		double blurAmount = 0.5;
		LagBufferSettings lag;
		Ns extraDelayNs = 0;
		int64_t offsetNs = 1'000'000'000;
		int lookahead = 2;             // frames kept announced ahead of the one on screen
		int withholdFrom = -1;         // frame indexes in [withholdFrom, withholdTo) are not announced before releaseAtNs
		int withholdTo = -1;
		int64_t releaseAtNs = 0;
		FrameFormat fmt{ 1920, 1080, 87 };

		int nextFrame = 0;
		uint64_t tick = 0;
		struct Out { int64_t V; int64_t tau; Pacer::Decision d; };
		std::vector<Out> log;

		int64_t Pts( int k ) const { return int64_t( std::llround( double( k ) * 1e9 / fps ) ); }
		int64_t PtsOfId( uint64_t id ) const { return Pts( int( id ) - 1 ); }   // id = k + 1
		int64_t TickNs( uint64_t i ) const { return int64_t( double( i ) * 1e9 / refreshHz ); }

		Pacer::Inputs Inputs( int64_t V ) const
		{
			Pacer::Inputs in;
			in.now = Ns( V );
			in.presentNs = Ns( V );
			in.mode = mode;
			in.multiplier = multiplier;
			in.targetFps = targetFps;
			in.refreshHz = refreshHz;
			in.frameGen = frameGen;
			in.blur = blur;
			in.blurSamples = 4;
			in.blurAmount = blurAmount;
			in.lagBuffer = lag;
			in.extraDelayNs = extraDelayNs;
			in.mediaOffsetNs = offsetNs;
			in.playbackRate = rate;
			return in;
		}

		void Announce( int64_t tau )
		{
			while ( pacer.WantsFrame() && Pts( nextFrame ) <= tau + int64_t( lookahead ) * int64_t( 1e9 / fps ) )
			{
				if ( nextFrame >= withholdFrom && nextFrame < withholdTo && TickNs( tick ) < releaseAtNs )
					break;
				pacer.OnFrame( uint64_t( nextFrame ) + 1, Ns( Pts( nextFrame ) ), 1, fmt );
				nextFrame++;
			}
		}

		bool skippable = false;        // the host lets a tick with nothing new be dropped

		Pacer::Decision Tick()
		{
			const int64_t V = TickNs( tick );
			const int64_t tau = int64_t( std::llround( double( V - offsetNs ) * rate ) );
			Announce( tau );
			const Pacer::Decision d = pacer.OnPaint( Inputs( V ), nextFrame ? uint64_t( nextFrame ) : 0, true, skippable );
			log.push_back( { V, tau, d } );
			tick++;
			return d;
		}

		// Ticks from the current one until V reaches tEndNs.
		void RunTo( int64_t tEndNs ) { while ( TickNs( tick ) < tEndNs ) Tick(); }

		// The content time an output SHOWS, in PTS ns: a real frame's own PTS or the
		// interpolated instant of its pair (the end of the blur window when blurred).
		double Shown( const Pacer::Decision &d ) const
		{
			if ( d.prevId == 0 || d.t1 >= 1.0f )
				return double( PtsOfId( d.prevId == 0 ? d.showId : d.currId ) );
			const double a = double( PtsOfId( d.prevId ) ), b = double( PtsOfId( d.currId ) );
			return a + double( d.t1 ) * ( b - a );
		}
	};
}

TEST_CASE( "video: 24 fps on 144 Hz, every tick is an output at its exact content time", "[video]" )
{
	VideoDriver v;
	v.RunTo( v.offsetNs + 3'000'000'000 );
	int nChecked = 0, nSynth = 0, nReal = 0;
	for ( const auto &o : v.log )
	{
		if ( o.tau < v.Pts( 2 ) )
			continue;                                    // warm-up: the first frames
		const int k = int( std::floor( double( o.tau ) / ( 1e9 / 24.0 ) + 1e-9 ) );
		const double span = double( v.Pts( k + 1 ) - v.Pts( k ) );
		const double tExp = double( o.tau - v.Pts( k ) ) / span;
		CHECK( o.d.newOutput );
		CHECK_FALSE( o.d.inert );
		CHECK_FALSE( o.d.skip );
		if ( o.d.prevId != 0 )
		{
			CHECK( std::fabs( double( o.d.t ) - tExp ) < 1e-6 );
			CHECK( o.d.prevId == uint64_t( k ) + 1 );
			CHECK( o.d.currId == uint64_t( k ) + 2 );
			nSynth++;
		}
		else
		{
			// On a real frame: t ~ 0 of pair k or ~1 of pair k-1.
			CHECK( std::min( tExp, 1.0 - tExp ) < 1e-4 + 1e-9 );
			nReal++;
		}
		nChecked++;
	}
	CHECK( nChecked > 250 );
	CHECK( nSynth == nChecked - nReal );
	CHECK( nReal >= 10 );                                // one real frame per 6 ticks
	CHECK( v.pacer.Generating() );
	const Pacer::Report r = v.pacer.TakeStatus( v.Inputs( v.TickNs( v.tick ) ), Ns( v.TickNs( v.tick ) ) );
	CHECK( r.reason == Reason::Normal );
	CHECK( r.delayMs == 0.0f );
	CHECK( std::fabs( r.gameFps - 24.0f ) < 0.01f );
}

TEST_CASE( "video: 24 fps film, Fixed 2x keeps a steady 3-tick cadence on 144 Hz", "[video]" )
{
	VideoDriver v;
	v.mode = Mode::Fixed;
	v.multiplier = 2;
	v.RunTo( v.offsetNs + 3'000'000'000 );
	// 48 outputs a second out of 144 ticks: one new output every 3 ticks.
	int nNew = 0;
	int64_t lastNewTick = -1;
	for ( size_t i = 0; i < v.log.size(); i++ )
	{
		if ( v.log[ i ].tau < v.Pts( 4 ) )
			continue;
		if ( v.log[ i ].d.newOutput )
		{
			if ( lastNewTick >= 0 )
				CHECK( int64_t( i ) - lastNewTick == 3 );
			lastNewTick = int64_t( i );
			nNew++;
		}
	}
	CHECK( nNew > 100 );
}

TEST_CASE( "video: 23.976 fps on 120 Hz over 60 s does not drift", "[video]" )
{
	VideoDriver v;
	v.refreshHz = 120.0;
	v.fps = 24000.0 / 1001.0;
	v.RunTo( v.offsetNs + 60'000'000'000 );
	REQUIRE( v.log.size() >= 7200 );
	double flMaxErr = 0.0;
	int nExactReal = 0, nOnTick = 0;
	for ( const auto &o : v.log )
	{
		if ( o.tau < v.Pts( 2 ) )
			continue;
		CHECK( o.d.newOutput || o.d.prevId == 0 );
		const int k = int( std::floor( double( o.tau ) / ( 1e9 / v.fps ) ) );
		const int kk = o.tau >= v.Pts( k + 1 ) ? k + 1 : ( o.tau < v.Pts( k ) ? k - 1 : k );
		const double span = double( v.Pts( kk + 1 ) - v.Pts( kk ) );
		const double flErr = std::fabs( v.Shown( o.d ) - double( o.tau ) ) / span;
		flMaxErr = std::max( flMaxErr, flErr );
		// A PTS that falls on a tick (every 200th frame: 1001 ticks) is shown as itself.
		if ( std::llabs( o.tau - v.Pts( kk ) ) < 2 || std::llabs( o.tau - v.Pts( kk + 1 ) ) < 2 )
		{
			nOnTick++;
			if ( o.d.prevId == 0 )
				nExactReal++;
		}
	}
	CHECK( flMaxErr <= 1e-4 );
	CHECK( nOnTick >= 5 );                               // 60 s / (200 frames = 8.34 s) = 7
	CHECK( nExactReal == nOnTick );
	// The content time of the very last output is where the media clock says, not a drift away.
	const auto &last = v.log.back();
	CHECK( std::fabs( v.Shown( last.d ) - double( last.tau ) ) < 0.01 * 1e9 / v.fps );
}

TEST_CASE( "video: a late decoder holds t = 1 (WaitingForDecoder) and resumes on the exact t", "[video]" )
{
	VideoDriver v;
	v.withholdFrom = 10;
	v.withholdTo = 11;
	// Frame 10 (pts 416.7 ms) is released at tau = pts_9 + 0.5 span, i.e. while frame 9 is on screen.
	v.releaseAtNs = v.offsetNs + v.Pts( 9 ) + int64_t( 0.5 * ( v.Pts( 10 ) - v.Pts( 9 ) ) );
	bool bSawWait = false;
	double flPrev = 0.0;
	for ( int i = 0; i < 400 && v.TickNs( v.tick ) < v.offsetNs + v.Pts( 14 ); i++ )
	{
		const Pacer::Decision d = v.Tick();
		const auto &o = v.log.back();
		if ( o.tau < v.Pts( 3 ) )
			continue;
		const double flShown = v.Shown( d );
		CHECK( flShown >= flPrev - 1.0 );                // the content never goes backwards
		flPrev = flShown;
		if ( v.pacer.WaitingForDecoder() )
		{
			bSawWait = true;
			CHECK( o.tau > v.Pts( 9 ) );
			CHECK( d.prevId == 0 );
			CHECK( d.t == 1.0f );
			CHECK( d.showId == 10 );                     // frame 9, id 10
			CHECK( v.pacer.TakeStatus( v.Inputs( o.V ), Ns( o.V ) ).reason == Reason::WaitingForDecoder );
		}
		else if ( o.tau > v.Pts( 9 ) && d.prevId != 0 )
		{
			int k = 9;
			while ( v.Pts( k + 1 ) <= o.tau )
				k++;
			CHECK( d.prevId == uint64_t( k ) + 1 );
			const double tExp = double( o.tau - v.Pts( k ) ) / double( v.Pts( k + 1 ) - v.Pts( k ) );
			CHECK( std::fabs( double( d.t ) - tExp ) < 1e-6 );
		}
	}
	CHECK( bSawWait );
	CHECK_FALSE( v.pacer.WaitingForDecoder() );
}

TEST_CASE( "video: Discontinuity is a seek, the first frame is shown directly", "[video]" )
{
	VideoDriver v;
	v.RunTo( v.offsetNs + 1'000'000'000 );
	REQUIRE( v.pacer.Generating() );
	// Seek to 10 s: a new timeline.
	v.pacer.Discontinuity();
	CHECK( v.pacer.PendingFrames() == 0 );
	const int kSeek = 240;                               // 10 s of film
	v.nextFrame = kSeek;
	v.offsetNs = v.TickNs( v.tick ) - v.Pts( kSeek ) - 10'000'000;   // pts_240 is due 10 ms from now
	v.lookahead = 0;                                     // only the first frame is decoded yet
	const Pacer::Decision d = v.Tick();
	CHECK( d.reset );
	CHECK( d.prevId == 0 );
	CHECK( d.t == 1.0f );
	CHECK( d.showId == uint64_t( kSeek ) + 1 );          // before its time: shown directly
	CHECK_FALSE( d.inert );
	v.lookahead = 2;
	v.RunTo( v.TickNs( v.tick ) + 1'000'000'000 );
	int nSynth = 0;
	for ( const auto &o : v.log )
		if ( o.d.prevId != 0 && o.tau > v.Pts( kSeek ) && o.d.prevId > uint64_t( kSeek ) )
		{
			const int k = int( o.d.prevId ) - 1;
			CHECK( std::fabs( double( o.d.t ) - double( o.tau - v.Pts( k ) ) / double( v.Pts( k + 1 ) - v.Pts( k ) ) ) < 1e-6 );
			nSynth++;
		}
	CHECK( nSynth > 100 );
}

TEST_CASE( "video: a PTS that goes backwards is a seek, an equal one a duplicate", "[video]" )
{
	Pacer p;
	FrameFormat f{ 64, 64, 1 };
	p.OnFrame( 1, 1000 * MS, 1, f );
	p.OnFrame( 2, 1041 * MS, 1, f );
	CHECK( p.PendingFrames() == 2 );
	CHECK( p.OnFrame( 2, 1041 * MS, 1, f ) );            // duplicate, ignored
	CHECK( p.PendingFrames() == 2 );
	p.OnFrame( 9, 500 * MS, 1, f );                      // backwards: forgets, keeps this one
	CHECK( p.PendingFrames() == 1 );
	CHECK( p.Timestamps() );
}

TEST_CASE( "video: the announced-ahead queue is bounded", "[video]" )
{
	Pacer p;
	FrameFormat f{ 64, 64, 1 };
	int nAccepted = 0;
	for ( int i = 0; i < 20; i++ )
		nAccepted += p.OnFrame( uint64_t( i ) + 1, Ns( i ) * 41 * MS, 1, f ) ? 1 : 0;
	CHECK( nAccepted == kVideoLookahead );
	CHECK_FALSE( p.WantsFrame() );
}

TEST_CASE( "video: blur's window comes from the PTS spacing", "[video][blur]" )
{
	VideoDriver v;
	v.blur = true;
	v.blurAmount = 0.5;                                  // ShownFrame: s * o / g = 0.5 / 6 of the pair
	v.RunTo( v.offsetNs + 2'000'000'000 );
	int nChecked = 0;
	for ( const auto &o : v.log )
	{
		if ( o.tau < v.Pts( 4 ) || o.d.prevId == 0 )
			continue;
		const double w = 0.5 / 6.0;
		CHECK( std::fabs( double( o.d.t1 ) - double( o.d.t ) ) < 1e-6 );
		CHECK( std::fabs( double( o.d.t0 ) - std::max( 0.0, double( o.d.t ) - w ) ) < 2e-4 );
		nChecked++;
	}
	CHECK( nChecked > 100 );
	CHECK( v.pacer.TakeStatus( v.Inputs( v.TickNs( v.tick ) ), Ns( v.TickNs( v.tick ) ) ).blurActive );
}

TEST_CASE( "video: the lag buffer, the extra delay and the priorities are inert", "[video]" )
{
	VideoDriver a, b;
	b.lag.enabled = true;
	b.lag.testMode = LagTestMode::ForceMax;
	b.extraDelayNs = 50 * MS;
	a.RunTo( a.offsetNs + 2'000'000'000 );
	b.RunTo( b.offsetNs + 2'000'000'000 );
	REQUIRE( a.log.size() == b.log.size() );
	for ( size_t i = 0; i < a.log.size(); i++ )
	{
		CHECK( a.log[ i ].d.t == b.log[ i ].d.t );
		CHECK( a.log[ i ].d.prevId == b.log[ i ].d.prevId );
		CHECK( a.log[ i ].d.showId == b.log[ i ].d.showId );
	}
	CHECK( b.pacer.BufferMs() == 0.0 );
	CHECK( b.pacer.DelayMs() == 0.0 );
	CHECK( b.pacer.HistoryDepth() == kHistory );
	const Pacer::Report r = b.pacer.TakeStatus( b.Inputs( b.TickNs( b.tick ) ), Ns( b.TickNs( b.tick ) ) );
	CHECK( r.bufferDelayMs == 0.0f );
	CHECK( r.bufferTargetMs == 0.0f );
}

TEST_CASE( "video: frame generation off schedules real frames at their PTS, one new output per frame", "[video]" )
{
	VideoDriver v;
	v.frameGen = false;
	v.RunTo( v.offsetNs + 2'000'000'000 );
	int nNew = 0;
	for ( const auto &o : v.log )
	{
		if ( o.tau < v.Pts( 3 ) )
			continue;
		CHECK( o.d.prevId == 0 );
		CHECK_FALSE( o.d.inert );
		// The frame whose time has come, not the newest announced one.
		const int k = int( std::floor( double( o.tau + 4000 ) / ( 1e9 / 24.0 ) ) );   // within the 1e-4 snap
		CHECK( std::llabs( v.PtsOfId( o.d.showId ) - v.Pts( k ) ) < 2 );
		nNew += o.d.newOutput ? 1 : 0;
	}
	CHECK( nNew >= 40 );
	CHECK( nNew <= 50 );                                 // ~24 a second for ~1.9 s
}

TEST_CASE( "video: content at the output rate is scheduled, not passed through", "[video]" )
{
	VideoDriver v;
	v.fps = 60.0;
	v.refreshHz = 60.0;
	v.RunTo( v.offsetNs + 1'000'000'000 );
	for ( const auto &o : v.log )
	{
		if ( o.tau < v.Pts( 4 ) )
			continue;
		CHECK( o.d.prevId == 0 );
		CHECK_FALSE( o.d.inert );
		CHECK( std::llabs( v.PtsOfId( o.d.showId ) - o.tau ) < int64_t( 1e9 / 60 ) );
	}
	CHECK( v.pacer.TakeStatus( v.Inputs( v.TickNs( v.tick ) ), Ns( v.TickNs( v.tick ) ) ).reason == Reason::GameTooFast );
}

TEST_CASE( "video: 2x playback rate maps the media clock", "[video]" )
{
	VideoDriver v;
	v.rate = 2.0;
	v.RunTo( v.offsetNs + 2'000'000'000 );
	int n = 0;
	for ( const auto &o : v.log )
	{
		if ( o.tau < v.Pts( 3 ) || o.d.prevId == 0 )
			continue;
		const int k = int( o.d.prevId ) - 1;
		CHECK( std::fabs( double( o.d.t ) - double( o.tau - v.Pts( k ) ) / double( v.Pts( k + 1 ) - v.Pts( k ) ) ) < 1e-6 );
		n++;
	}
	CHECK( n > 100 );
	CHECK( std::fabs( v.pacer.TakeStatus( v.Inputs( v.TickNs( v.tick ) ), Ns( v.TickNs( v.tick ) ) ).gameFps - 24.0f ) < 0.01f );
}

TEST_CASE( "video: arrival and timestamp sources do not mix silently", "[video]" )
{
	Pacer p;
	FrameFormat f{ 64, 64, 1 };
	p.OnFrame( 1, 10 * MS, 1, f );
	CHECK( p.Timestamps() );
	p.OnArrival( 2, 20 * MS, 1, f );
	CHECK_FALSE( p.Timestamps() );
	CHECK( p.PendingFrames() == 0 );
}

// ---- real-rate skips and the cold-start history depth ----

namespace
{
	struct OutKey
	{
		Ns t; uint64_t show, prev, curr; float t0, t1; int64_t tau;
		bool operator==( const OutKey &o ) const
		{
			return t == o.t && show == o.show && prev == o.prev && curr == o.curr && t0 == o.t0 && t1 == o.t1 && tau == o.tau;
		}
	};

	std::vector<OutKey> OutputsOf( const Driver &drv )
	{
		std::vector<OutKey> v;
		for ( const Painted &p : drv.log )
			if ( p.d.newOutput )
				v.push_back( { p.t, p.d.showId, p.d.prevId, p.d.currId, p.d.t0, p.d.t1, p.tau } );
		return v;
	}

	int PaintsOf( const Driver &drv, Ns t0, Ns t1 )
	{
		int n = 0;
		for ( const Painted &p : drv.log )
			if ( p.t >= t0 && p.t < t1 && !p.d.skip )
				n++;
		return n;
	}

	// A 30 fps game on a 120 Hz tick, FG off, with a delay and/or blur.
	Driver RealRateDriver( bool bSkip, double flBufMs, bool bBlur )
	{
		Driver drv;
		drv.refreshHz = 120.0;
		drv.frameGen = false;
		drv.skipVblanks = bSkip;
		drv.blur = bBlur;
		if ( flBufMs > 0.0 )
		{
			drv.lag.enabled = true;
			drv.lag.maxBufferMs = flBufMs;
			drv.lag.testMode = LagTestMode::ForceMax;
		}
		return drv;
	}
}

TEST_CASE( "real-rate: ticks with nothing new are skippable, outputs unchanged", "[realrate][skip]" )
{
	const double flI = 1000.0 / 30.0;
	Driver a = RealRateDriver( false, 50.0, false );
	Driver b = RealRateDriver( true, 50.0, false );
	a.Run( 20 * 1000 * MS, flI );
	b.Run( 20 * 1000 * MS, flI );
	REQUIRE( b.pacer.BufferMs() == 50.0 );
	// identical output sequence (what is shown, when, and its content time)
	CHECK( OutputsOf( a ) == OutputsOf( b ) );
	CHECK( OutputsOf( a ).size() > 500 );
	// settled: a paint for each new frame (its ring copy) plus one for each output, so
	// at most ~2 x 30 a second; it used to be one per 120 Hz tick
	const Ns t0 = 10 * 1000 * MS, t1 = 20 * 1000 * MS;
	const int nOld = PaintsOf( a, t0, t1 );
	const int nNew = PaintsOf( b, t0, t1 );
	int nArrivals = 0;
	for ( const Painted &p : b.log )
		if ( p.t >= t0 && p.t < t1 && p.d.newestId != 0 && !p.d.skip && p.d.newOutput == false )
			nArrivals++;
	INFO( "paints old " << nOld << " new " << nNew );
	CHECK( nOld >= 10 * 120 - 2 );
	CHECK( nNew >= 10 * 30 );
	CHECK( nNew <= b.CountOutputs( t0, t1 ) + 10 * 31 + 2 );
	CHECK( nNew <= 10 * 2 * 31 + 2 );
	CHECK( nArrivals <= 10 * 31 + 2 );                // non-output paints: only the ring copies
	CHECK( b.CountOutputs( 10 * 1000 * MS, 20 * 1000 * MS ) == a.CountOutputs( 10 * 1000 * MS, 20 * 1000 * MS ) );
	// a skip never carries a reset or a new output
	for ( const Painted &p : b.log )
		if ( p.d.skip )
			CHECK( ( !p.d.newOutput && !p.d.reset ) );
}

TEST_CASE( "real-rate: without a skippable host nothing is ever skipped", "[realrate][skip]" )
{
	Driver a = RealRateDriver( false, 50.0, false );
	a.Run( 5 * 1000 * MS, 1000.0 / 30.0 );
	for ( const Painted &p : a.log )
		CHECK_FALSE( p.d.skip );
}

TEST_CASE( "real-rate: blur only skips the same way and loses no output", "[realrate][skip][blur]" )
{
	const double flI = 1000.0 / 30.0;
	Driver a = RealRateDriver( false, 0.0, true );
	Driver b = RealRateDriver( true, 0.0, true );
	a.Run( 10 * 1000 * MS, flI );
	b.Run( 10 * 1000 * MS, flI );
	CHECK( OutputsOf( a ) == OutputsOf( b ) );
	CHECK( OutputsOf( b ).size() > 250 );
	CHECK( PaintsOf( b, 5 * 1000 * MS, 10 * 1000 * MS ) <= 5 * 31 + 2 );
}

TEST_CASE( "real-rate: a late frame after a gap still gets its output when skipping", "[realrate][skip]" )
{
	const double flI = 1000.0 / 30.0;
	Driver a = RealRateDriver( false, 50.0, false );
	Driver b = RealRateDriver( true, 50.0, false );
	std::vector<Ns> arr = Driver::Periodic( 3 * MS, flI, 6 * 1000 * MS );
	std::vector<Ns> kept;
	for ( Ns t : arr )
		if ( t < 3000 * MS || t > 3120 * MS )
			kept.push_back( t );               // a 120 ms hole: a lag spike
	a.RunArrivals( kept, 6 * 1000 * MS );
	b.RunArrivals( kept, 6 * 1000 * MS );
	CHECK( OutputsOf( a ) == OutputsOf( b ) );
}

TEST_CASE( "video: a repeated real frame is skippable, outputs unchanged", "[video][skip]" )
{
	VideoDriver a, b;
	a.fps = b.fps = 24.0;
	a.frameGen = b.frameGen = false;
	b.skippable = true;
	a.RunTo( a.offsetNs + 3'000'000'000 );
	b.RunTo( b.offsetNs + 3'000'000'000 );
	REQUIRE( a.log.size() == b.log.size() );
	int nSkip = 0, nOutA = 0, nOutB = 0;
	for ( size_t i = 0; i < a.log.size(); i++ )
	{
		nOutA += a.log[ i ].d.newOutput;
		nOutB += b.log[ i ].d.newOutput;
		nSkip += b.log[ i ].d.skip;
		CHECK( a.log[ i ].d.newOutput == b.log[ i ].d.newOutput );
		if ( a.log[ i ].d.newOutput )
		{
			CHECK( a.log[ i ].d.showId == b.log[ i ].d.showId );
			CHECK( a.log[ i ].d.t == b.log[ i ].d.t );
		}
		CHECK_FALSE( ( b.log[ i ].d.skip && b.log[ i ].d.newOutput ) );
	}
	CHECK( nOutA == nOutB );
	CHECK( nSkip > 100 );     // 144 Hz ticks of a 24 fps file
}

TEST_CASE( "lag buffer: a forced max buffer does not balloon the history before the interval is known", "[lagbuffer][depth]" )
{
	for ( double flFps : { 30.0, 60.0, 120.0 } )
	{
		const double flI = 1000.0 / flFps;
		Driver drv = LagDriver( 100.0 );
		drv.lag.testMode = LagTestMode::ForceMax;
		drv.Run( 10 * 1000 * MS, flI );
		const int nCeil = int( std::ceil( 100.0 / flI ) ) + kHistoryMargin + 2;
		int nMax = 0;
		for ( const Painted &p : drv.log )
			nMax = std::max( nMax, p.depth );
		INFO( "fps " << flFps << " peak depth " << nMax << " limit " << nCeil );
		CHECK( nMax <= nCeil );
		if ( flFps == 30.0 )
			CHECK( nMax <= 9 );
		// the first paints are at the default depth, and the target is not sized from a guess
		CHECK( drv.log.front().depth == kHistory );
		CHECK( drv.log.front().bufMs == 0.0 );
	}
}

TEST_CASE( "lag buffer: switching it off drops the buffer at once", "[lagbuffer][instantoff]" )
{
	for ( bool bFG : { false, true } )
	{
		Driver drv = LagDriver();
		drv.frameGen = bFG;
		drv.lag.testMode = LagTestMode::ForceMax;
		drv.Run( 10 * 1000 * MS, 1000.0 / 60.0 );
		REQUIRE( drv.pacer.BufferMs() > 49.0 );
		REQUIRE( drv.Status().bufferDelayMs > 49.0f );

		// what the base delay settles to with the buffer off
		Driver ref = LagDriver();
		ref.frameGen = bFG;
		ref.lag.enabled = false;
		ref.Run( 10 * 1000 * MS, 1000.0 / 60.0 );
		const double flBase = ref.pacer.DelayMs();

		drv.lag.enabled = false;
		const size_t n0 = drv.log.size();
		drv.Run( 2 * 1000 * MS, 1000.0 / 60.0 );
		REQUIRE( drv.log.size() > n0 + 10 );

		// the very next paint: buffer gone, delay is the base delay
		const Painted &p = drv.log[ n0 ];
		CHECK( p.bufMs == 0.0 );
		CHECK( Near( p.delayMs, flBase ) );
		CHECK( drv.Status().bufferDelayMs == 0.0f );
		CHECK( drv.Status().bufferTargetMs == 0.0f );

		// tau and the shown frame never go backwards, across the switch or after it
		int64_t tauPrev = n0 > 0 ? drv.log[ n0 - 1 ].tau : 0;
		uint64_t idPrev = 0;
		for ( size_t i = n0 > 0 ? n0 - 1 : 0; i < drv.log.size(); i++ )
		{
			const Painted &q = drv.log[ i ];
			if ( q.tau != 0 )          // 0: an inert pass-through shows the newest frame, no content time
			{
				CHECK( q.tau >= tauPrev );
				tauPrev = q.tau;
			}
			if ( q.d.newOutput )
			{
				CHECK( q.d.showId >= idPrev );
				idPrev = q.d.showId;
			}
			if ( i >= n0 )
				CHECK( q.bufMs == 0.0 );
		}
	}
}

TEST_CASE( "lag buffer: re-enabling after an instant off starts from zero and grows normally", "[lagbuffer][instantoff]" )
{
	Driver drv = LagDriver();
	drv.lag.testMode = LagTestMode::ForceMax;
	drv.Run( 10 * 1000 * MS, 1000.0 / 60.0 );
	REQUIRE( drv.pacer.BufferMs() > 49.0 );
	drv.lag.enabled = false;
	drv.Run( 500 * MS, 1000.0 / 60.0 );
	REQUIRE( drv.pacer.BufferMs() == 0.0 );
	drv.lag.enabled = true;
	drv.Run( 200 * MS, 1000.0 / 60.0 );
	// the ramp (3 %): ~6 ms after 200 ms, nowhere near an instant jump
	CHECK( drv.pacer.BufferMs() > 1.0 );
	CHECK( drv.pacer.BufferMs() < 10.0 );
	drv.Run( 5 * 1000 * MS, 1000.0 / 60.0 );
	CHECK( Near( drv.pacer.BufferMs(), 50.0 ) );
}

TEST_CASE( "lag buffer: with the buffer enabled the target falling to zero still shrinks by the ramp", "[lagbuffer][instantoff]" )
{
	Driver drv = LagDriver();
	drv.lag.testMode = LagTestMode::ForceMax;
	drv.Run( 10 * 1000 * MS, 1000.0 / 60.0 );
	drv.lag.testMode = LagTestMode::ForceMin;
	drv.Run( 100 * MS, 1000.0 / 60.0 );
	CHECK( drv.pacer.BufferMs() > 40.0 );   // 3 % of 100 ms = 3 ms, not a jump
	drv.Run( 3 * 1000 * MS, 1000.0 / 60.0 );
	CHECK( drv.pacer.BufferMs() == 0.0 );
}

// ---------------------------------------------------------------------------
//  High game rates: the interval clamp is 1 ms (1000 fps), not 4 ms (250 fps)
// ---------------------------------------------------------------------------

namespace
{
	// 280 Hz display, Target = the refresh, Pause at refresh rate on.
	Driver HighFpsDriver()
	{
		Driver drv;
		drv.refreshHz = 280.0;
		drv.mode = Mode::Target;
		drv.targetFps = 280;
		drv.pauseAtRefresh = true;
		drv.canExceed = false;
		return drv;
	}
}

TEST_CASE( "high fps: a 340 and a 450 fps game on 280 Hz with Pause at refresh on passes through", "[highfps]" )
{
	for ( double flFps : { 340.0, 450.0 } )
	{
		Driver drv = HighFpsDriver();
		drv.Run( 2000 * MS, 1000.0 / flFps );
		const Pacer::Report s = drv.Status();
		CHECK( std::fabs( s.gameFps - float( flFps ) ) < 0.03f * float( flFps ) );
		CHECK( s.delayMs == 0.0f );
		CHECK( s.reason == Reason::GameTooFast );
		CHECK( drv.pacer.DelayMs() == 0.0 );
		int nChecked = 0;
		for ( const Painted &p : drv.log )
			if ( p.t > 1000 * MS )
			{
				nChecked++;
				CHECK( p.d.inert );
								CHECK( p.delayMs == 0.0 );
			}
		CHECK( nChecked > 100 );
	}
}

TEST_CASE( "high fps: a 260 fps game on 280 Hz generates, and the hysteresis holds from both sides", "[highfps]" )
{
	// 260 fps: ratio 1.077, between the stop (1.02) and start (1.10) ratios.
	Driver a = HighFpsDriver();
	a.Run( 1500 * MS, 1000.0 / 260.0 );
	CHECK( std::fabs( a.Status().gameFps - 260.0f ) < 8.0f );
	// First decision uses the stop ratio alone: 1.077 > 1.02 -> generating.
	CHECK( a.pacer.Generating() );

	// ... a game that was passing through (340 fps) and drops to 260 stays passing: 1.077 < 1.10.
	Driver b = HighFpsDriver();
	b.Run( 1500 * MS, 1000.0 / 340.0 );
	REQUIRE_FALSE( b.pacer.Generating() );
	b.Run( 1500 * MS, 1000.0 / 260.0 );
	CHECK_FALSE( b.pacer.Generating() );
	// ... and starts only once the game is below ~254 fps (ratio > 1.10).
	b.Run( 1500 * MS, 1000.0 / 250.0 );
	CHECK( b.pacer.Generating() );

	// From the other side: generating at 240, rising to 270 (ratio 1.037 > 1.02) keeps generating;
	// 280 (1.0) stops it.
	Driver c = HighFpsDriver();
	c.Run( 1500 * MS, 1000.0 / 240.0 );
	REQUIRE( c.pacer.Generating() );
	c.Run( 1500 * MS, 1000.0 / 270.0 );
	CHECK( c.pacer.Generating() );
	c.Run( 1500 * MS, 1000.0 / 290.0 );
	CHECK_FALSE( c.pacer.Generating() );
}

TEST_CASE( "high fps: fixed 2x with Pause off generates a 400 fps game at 800 fps", "[highfps]" )
{
	Driver drv;
	drv.refreshHz = 280.0;
	drv.mode = Mode::Fixed;
	drv.multiplier = 2;
	drv.pauseAtRefresh = false;
	drv.canExceed = true;
	drv.Run( 2000 * MS, 1000.0 / 400.0 );
	CHECK( drv.pacer.Generating() );
	const Pacer::Report s = drv.Status();
	CHECK( std::fabs( s.gameFps - 400.0f ) < 12.0f );
	CHECK( s.reason == Reason::Normal );
	CHECK( std::fabs( drv.pacer.OutputMs() - 1.25 ) < 0.05 );
	CHECK( drv.pacer.ExtraTimer() );       // 1.25 ms < the 3.57 ms tick: the host's own timer paints
}

TEST_CASE( "high fps: the lag buffer at 400 fps is capped by the history, not exploded", "[highfps][lagbuffer]" )
{
	Driver drv = LagDriver( 250.0 );
	drv.lag.testMode = LagTestMode::ForceMax;
	drv.Run( 12 * 1000 * MS, 2.5 );
	CHECK( drv.pacer.HistoryDepth() <= kHistoryMax );
	CHECK( drv.pacer.BufferMs() <= double( kHistoryMax - kHistory - kHistoryMargin ) * 2.5 + 1e-6 );   // 150 ms
	CHECK( drv.pacer.BufferMs() >= 145.0 );
	for ( const Painted &p : drv.log )
		CHECK( p.depth <= kHistoryMax );

	// 1000 fps (the clamp): 60 frames x 1 ms.
	Driver top = LagDriver( 250.0 );
	top.lag.testMode = LagTestMode::ForceMax;
	top.Run( 8 * 1000 * MS, 1.0 );
	CHECK( top.pacer.HistoryDepth() <= kHistoryMax );
	CHECK( top.pacer.BufferMs() <= 60.0 + 1e-6 );
	CHECK( top.pacer.BufferMs() >= 55.0 );
}

// ---------------------------------------------------------------------------
//  The generate / pass decision: a steady rate and a dwell (no flapping near the target)
// ---------------------------------------------------------------------------
//
// The user's report: the HUD's "Both" readout kept switching between "game > output" and the
// lone game number "for no apparent reason" -- 246 flips in 246 s at a 280 Hz target with a
// game wandering 210..330 fps. The decision was made on the pacing's own 8-frame median.

namespace
{
	// 280 Hz, Target = the display's refresh (target 0), Smoothness, Pause at refresh OFF on a
	// backend that can exceed it: the user's profile.
	Driver FlapDriver()
	{
		Driver drv;
		drv.refreshHz = 280.0;
		drv.mode = Mode::Target;
		drv.targetFps = 0;
		drv.priority = Priority::Smoothness;
		drv.pauseAtRefresh = false;
		drv.canExceed = true;
		return drv;
	}

	// A deterministic, portable jitter source (uniform, scaled to the requested sigma).
	struct Jitter
	{
		uint64_t s = 0x9E3779B97F4A7C15ull;
		double Next()   // uniform in [-1, 1)
		{
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			return double( s >> 11 ) / double( 1ull << 52 ) - 1.0;
		}
		double Sigma( double flSigma ) { return Next() * flSigma * 1.7320508075688772; }   // uniform +-sqrt3 sigma: stddev sigma
	};

	// Arrivals at fpsAt(t) with relative interval jitter `sigma`; a gap (hitch) of hitchMs extra
	// at the first arrival at or after hitchAt (-1 = none).
	template <class F>
	std::vector<Ns> Arrivals( Ns t0, Ns t1, F fpsAt, double sigma, uint64_t seed = 1, Ns hitchAt = ~Ns( 0 ), double hitchMs = 0.0 )
	{
		Jitter j;
		j.s ^= seed * 0xD1B54A32D192ED03ull;
		std::vector<Ns> v;
		Ns t = t0;
		bool bHitched = false;
		while ( t < t1 )
		{
			v.push_back( t );
			double flIv = 1000.0 / fpsAt( double( t - t0 ) / 1e9 ) * ( 1.0 + j.Sigma( sigma ) );
			flIv = std::max( flIv, 1.0 );
			if ( !bHitched && t >= hitchAt )
			{
				flIv += hitchMs;
				bHitched = true;
			}
			t += Ns( flIv * 1e6 );
		}
		return v;
	}

	struct Flips
	{
		int n = 0;                    // changes of Generating() after tSkip
		double minGapMs = 1e9;        // the shortest time between two of them
		std::vector<double> atS;      // when
	};

	Flips CountFlips( const Driver &drv, Ns tSkip )
	{
		Flips f;
		bool bPrev = false, bFirst = true;
		Ns tPrev = 0;
		for ( const Painted &p : drv.log )
		{
			if ( p.t < tSkip )
			{
				bPrev = p.gen;
				bFirst = false;
				continue;
			}
			if ( !bFirst && p.gen != bPrev )
			{
				f.n++;
				f.atS.push_back( double( p.t ) / 1e9 );
				if ( tPrev != 0 )
					f.minGapMs = std::min( f.minGapMs, double( p.t - tPrev ) / 1e6 );
				tPrev = p.t;
			}
			bPrev = p.gen;
			bFirst = false;
		}
		return f;
	}

	// The first change of Generating() at or after t (seconds), or -1.
	double FirstFlipAfter( const Driver &drv, double flS )
	{
		bool bPrev = false;
		bool bHave = false;
		for ( const Painted &p : drv.log )
		{
			if ( double( p.t ) / 1e9 >= flS && bHave && p.gen != bPrev )
				return double( p.t ) / 1e9;
			bPrev = p.gen;
			bHave = true;
		}
		return -1.0;
	}
}

TEST_CASE( "steady rate: a hitch does not move it, a real step pulls it all the way", "[steady]" )
{
	SteadyRate r;
	CHECK_FALSE( r.Valid() );
	for ( int i = 0; i < 200; i++ )
		r.AddInterval( 3.7 );
	CHECK( r.Valid() );
	CHECK( std::fabs( r.IntervalMs() - 3.7 ) < 1e-9 );
	r.AddInterval( 50.0 );                         // one hitch among 3.7 ms frames
	CHECK( std::fabs( r.IntervalMs() / 3.7 - 1.0 ) < 0.015 );
	for ( int i = 0; i < 100; i++ )
		r.AddInterval( 3.7 );
	CHECK( std::fabs( r.IntervalMs() / 3.7 - 1.0 ) < 0.005 );      // and it is gone again

	// 300 -> 150 fps: within ~0.1 s the measure is above the 1.10 start ratio at 280 Hz (3.93 ms).
	SteadyRate s;
	for ( int i = 0; i < 200; i++ )
		s.AddInterval( 1000.0 / 300.0 );
	double flT = 0.0;
	while ( s.IntervalMs() < 3.93 && flT < 2000.0 )
	{
		s.AddInterval( 1000.0 / 150.0 );
		flT += 1000.0 / 150.0;
	}
	CHECK( flT < 200.0 );

	// Its mean is the true rate (frame-weighted): 10% / 25% uniform-ish jitter around 3.7 ms.
	for ( double flSigma : { 0.10, 0.25 } )
	{
		SteadyRate m;
		Jitter j;
		for ( int i = 0; i < 4000; i++ )
			m.AddInterval( 3.7 * ( 1.0 + j.Sigma( flSigma ) ) );
		CHECK( std::fabs( m.IntervalMs() / 3.7 - 1.0 ) < 0.03 );
	}

	// Young history = the median of what there is (no wild first mean).
	SteadyRate y;
	for ( double v : { 3.7, 3.7, 3.7, 40.0 } )
		y.AddInterval( v );
	CHECK( std::fabs( y.IntervalMs() - 3.7 ) < 1e-9 );
}

TEST_CASE( "decision: a game hovering near the target does not flap (Target auto, 280 Hz)", "[steady][flap]" )
{
	struct Case { double fps, sigma, drift; };
	const Case cases[] = {
		{ 255, 0.10, 0 }, { 262, 0.10, 0 }, { 270, 0.10, 0 }, { 275, 0.10, 0 },
		{ 255, 0.25, 0 }, { 262, 0.25, 0 }, { 270, 0.25, 0 }, { 275, 0.25, 0 },
		{ 262, 0.10, 0.15 }, { 270, 0.10, 0.15 },
	};
	for ( const Case &c : cases )
	{
		for ( uint64_t seed = 1; seed <= 3; seed++ )
		{
			Driver drv = FlapDriver();
			const Ns tEnd = 60 * 1000 * MS;
			// a slow +-15% drift: a period of ~58 s
			auto fpsAt = [&]( double t ) { return c.fps * ( 1.0 + c.drift * std::sin( t * c.fps * 0.0004 ) ); };
			drv.RunArrivals( Arrivals( 3 * MS, tEnd, fpsAt, c.sigma, seed ), tEnd );
			const Flips f = CountFlips( drv, 2000 * MS );
			INFO( "fps " << c.fps << " sigma " << c.sigma << " drift " << c.drift << " seed " << seed << " flips " << f.n );
			CHECK( f.n <= 3 );                                    // per minute (the run is one)
			CHECK( f.minGapMs >= kStartDwellMs - 1.0 );           // never a flip-flop inside the dwell
		}
	}
}

TEST_CASE( "decision: Target 400 and Fixed 3x near their own edges do not flap either", "[steady][flap]" )
{
	// An explicit target flaps the same way near the target (a 380 fps game, target 400).
	Driver t = FlapDriver();
	t.targetFps = 400;
	const Ns tEnd = 60 * 1000 * MS;
	t.RunArrivals( Arrivals( 3 * MS, tEnd, []( double ) { return 380.0; }, 0.10 ), tEnd );
	CHECK( CountFlips( t, 2000 * MS ).n <= 3 );
	// Fixed 3x with Pause at refresh ON: r = min(3, game / refresh) -> it hovers at 270 fps too.
	Driver f = FlapDriver();
	f.mode = Mode::Fixed;
	f.multiplier = 3;
	f.pauseAtRefresh = true;
	f.canExceed = false;
	f.RunArrivals( Arrivals( 3 * MS, tEnd, []( double ) { return 270.0; }, 0.10 ), tEnd );
	CHECK( CountFlips( f, 2000 * MS ).n <= 3 );
	// Fixed 3x with Pause off, and an explicit 400 target at 270 fps: generating throughout, as before.
	Driver g = FlapDriver();
	g.mode = Mode::Fixed;
	g.multiplier = 3;
	g.RunArrivals( Arrivals( 3 * MS, 10 * 1000 * MS, []( double ) { return 270.0; }, 0.10 ), 10 * 1000 * MS );
	CHECK( CountFlips( g, 500 * MS ).n == 0 );
	CHECK( g.pacer.Generating() );
	Driver h = FlapDriver();
	h.targetFps = 400;
	h.RunArrivals( Arrivals( 3 * MS, 10 * 1000 * MS, []( double ) { return 270.0; }, 0.10 ), 10 * 1000 * MS );
	CHECK( CountFlips( h, 500 * MS ).n == 0 );
	CHECK( h.pacer.Generating() );
	// ... and a game above the target (450 fps vs 400) passes through.
	Driver k = FlapDriver();
	k.targetFps = 400;
	k.RunArrivals( Arrivals( 3 * MS, 10 * 1000 * MS, []( double ) { return 450.0; }, 0.10 ), 10 * 1000 * MS );
	CHECK_FALSE( k.pacer.Generating() );
}

TEST_CASE( "decision: a game that really drops starts generating within 0.5 s, one that really rises stops within 1.5 s", "[steady][flap]" )
{
	for ( uint64_t seed = 1; seed <= 3; seed++ )
	{
		// 300 -> 150 fps at t = 5 s (the arrivals' own clock starts at 3 ms).
		Driver d = FlapDriver();
		const Ns tEnd = 12 * 1000 * MS;
		d.RunArrivals( Arrivals( 3 * MS, tEnd, []( double t ) { return t < 5.0 ? 300.0 : 150.0; }, 0.10, seed ), tEnd );
		REQUIRE_FALSE( d.log[ d.log.size() / 4 ].gen );
		const double flOn = FirstFlipAfter( d, 5.0 );
		INFO( "seed " << seed << " starts at " << flOn );
		CHECK( flOn > 5.0 );
		CHECK( flOn - 5.0 < 0.5 );
		CHECK( flOn - 5.0 >= 0.3 * 0.5 );          // not instantly: the dwell and the steady rate
		CHECK( d.pacer.Generating() );

		// 150 -> 320 fps at t = 5 s.
		Driver u = FlapDriver();
		u.RunArrivals( Arrivals( 3 * MS, tEnd, []( double t ) { return t < 5.0 ? 150.0 : 320.0; }, 0.10, seed ), tEnd );
		REQUIRE( u.log[ u.log.size() / 4 ].gen );
		const double flOff = FirstFlipAfter( u, 5.0 );
		INFO( "seed " << seed << " stops at " << flOff );
		CHECK( flOff > 5.0 );
		CHECK( flOff - 5.0 < 1.5 );
		CHECK( flOff - 5.0 >= kStopDwellMs / 1000.0 );   // the dwell: never sooner than 1 s
		CHECK_FALSE( u.pacer.Generating() );
	}
}

TEST_CASE( "decision: a single hitch does not flip the state", "[steady][flap]" )
{
	// 50 ms is below kGapResetNs (a longer stall is the gap path and drops the history).
	for ( double flFps : { 200.0, 262.0, 300.0 } )
	{
		for ( double flHitchMs : { 50.0, 90.0 } )
		{
			Driver d = FlapDriver();
			const Ns tEnd = 20 * 1000 * MS;
			d.RunArrivals( Arrivals( 3 * MS, tEnd, [&]( double ) { return flFps; }, 0.10, 1, 10 * 1000 * MS, flHitchMs ), tEnd );
			INFO( "fps " << flFps << " hitch " << flHitchMs << " ms" );
			CHECK( CountFlips( d, 3000 * MS ).n == 0 );
		}
	}
}

TEST_CASE( "decision: the dwell is time, not frames: a steady change lands after 0.3 s to start, 1 s to stop", "[steady][flap]" )
{
	// Exactly periodic frames, so only the dwell (and the rate measure's short lag) is measured.
	Driver d = FlapDriver();
	d.Run( 4 * 1000 * MS, 1000.0 / 300.0 );          // 0.93 -> passing through
	REQUIRE_FALSE( d.pacer.Generating() );
	const Ns tStep = d.now;
	d.Run( 3 * 1000 * MS, 1000.0 / 240.0 );          // 1.167 > 1.10
	const double flOn = FirstFlipAfter( d, double( tStep ) / 1e9 ) - double( tStep ) / 1e9;
	CHECK( flOn >= kStartDwellMs / 1000.0 );
	CHECK( flOn < 1.0 );          // a mild change (300 -> 240 fps): the rate measure needs ~0.45 s to cross
	const Ns tBack = d.now;
	d.Run( 4 * 1000 * MS, 1000.0 / 300.0 );
	const double flOff = FirstFlipAfter( d, double( tBack ) / 1e9 ) - double( tBack ) / 1e9;
	CHECK( flOff >= kStopDwellMs / 1000.0 );
	CHECK( flOff < 1.5 );
	CHECK_FALSE( d.pacer.Generating() );
}

TEST_CASE( "decision: the pacing still follows the 8-frame median, only the decision is steady", "[steady]" )
{
	// A 100 fps game at 4x display speed (Target 400, pause off): the output interval, the delay
	// and the cadence come from the median interval exactly as before.
	Driver d = FlapDriver();
	d.targetFps = 400;
	d.Run( 3 * 1000 * MS, 10.0 );
	CHECK( d.pacer.Generating() );
	CHECK( std::fabs( d.pacer.OutputMs() - 2.5 ) < 1e-6 );
	CHECK( std::fabs( d.pacer.IntervalMs() - 10.0 ) < 1e-6 );
	// ... and video mode keeps its immediate, PTS-exact decision (no dwell).
	Pacer p;
	const FrameFormat fmt{ 1920, 1080, 87 };
	for ( int i = 0; i < 4; i++ )
		p.OnFrame( uint64_t( i + 1 ), Ns( i ) * 41'666'667ull, 1, fmt );
	Pacer::Inputs in;
	in.now = in.presentNs = 0;
	in.mode = Mode::Target;
	in.targetFps = 0;
	in.refreshHz = 144.0;
	p.OnPaint( in, 1, true, false );
	CHECK( p.Planning() );
}

TEST_CASE( "status: generating does not flip while the multiplier hovers near 1.1x-1.3x (no multiplier gate)", "[steady][status]" )
{
	// The report: a paint-limited output (target 335, game ~177 fps: ~1.28x) kept flipping the HUD's
	// "game > output" separator because `generating` demanded >= 1.2x to turn on, >= 1.1x to stay. At
	// 280 Hz Target auto a game of 218 / 233 / 250 / 262 fps is 1.28x / 1.2x / 1.12x / 1.07x; add the
	// jitter of a real game (+-8 %) and the multiplier wanders across both old gates.
	for ( double flFps : { 218.0, 233.0, 250.0, 262.0 } )
	{
		const Ns tEnd = 40000 * MS;
		Driver live = FlapDriver();         // driven in 250 ms steps so a status can be taken between them
		int nFlips = 0, nSamples = 0, nOff = 0;
		bool bPrev = false;
		float flMin = 1e9f, flMax = 0.0f;
		const std::vector<Ns> arr = JitteredArrivals( 1000.0 / flFps, 0.08 * 1000.0 / flFps, tEnd );
		size_t i = 0;
		for ( Ns t = 0; t < tEnd; t += 250 * MS )
		{
			std::vector<Ns> chunk;
			while ( i < arr.size() && arr[ i ] < t + 250 * MS )
				chunk.push_back( arr[ i++ ] );
			live.RunArrivals( chunk, t + 250 * MS );
			const Pacer::Report r = live.Status();
			if ( t < 5000 * MS )
			{
				bPrev = r.generating;
				continue;
			}
			nSamples++;
			nOff += r.generating ? 0 : 1;
			nFlips += r.generating != bPrev ? 1 : 0;
			bPrev = r.generating;
			if ( r.steadyMultiplier > 0.0f )
			{
				flMin = std::min( flMin, r.steadyMultiplier );
				flMax = std::max( flMax, r.steadyMultiplier );
			}
		}
		INFO( "game " << flFps << " fps: steady multiplier " << flMin << " .. " << flMax << ", flips " << nFlips << ", off in " << nOff << " of " << nSamples << " samples" );
		CHECK( nSamples > 100 );
		CHECK( nFlips == 0 );
		CHECK( nOff == 0 );
	}
}

TEST_CASE( "status: generating is the steady signal, activeN the instantaneous one", "[steady][status]" )
{
	// Clearly generating: Fixed 2x of a 60 fps game on 125 Hz is 2x.
	Driver a;
	a.mode = Mode::Fixed;
	a.multiplier = 2;
	a.priority = Priority::Smoothness;
	Pacer::Report r;
	for ( int i = 0; i < 6; i++ )
	{
		a.Run( 250 * MS, 1000.0 / 60.0 );
		r = a.Status();
	}
	CHECK( r.fgActive );
	CHECK( r.generating );
	CHECK( r.steadyMultiplier > 1.8f );
	CHECK( r.steadyMultiplier < 2.2f );

	// Passing through (a 200 fps game, 125 Hz): never.
	Driver b;
	b.mode = Mode::Target;
	b.targetFps = 0;
	b.priority = Priority::Smoothness;
	for ( int i = 0; i < 6; i++ )
	{
		b.Run( 250 * MS, 1000.0 / 200.0 );
		r = b.Status();
	}
	CHECK_FALSE( r.fgActive );
	CHECK_FALSE( r.generating );

	// The pacer says Generate (a 262 fps game on 280 Hz: ratio 1.07, inside the hold band) and the
	// effective multiplier is only ~1.07, but frames ARE being generated: `generating` is true (no
	// multiplier gate: the user, "If its generating any frames, it should show the '>'. Otherwise
	// that differentiation is useless."); steadyMultiplier stays the informational ratio.
	Driver c = FlapDriver();
	c.Run( 1000 * MS, 1000.0 / 262.0 );
	for ( int i = 0; i < 6; i++ )
	{
		c.Run( 250 * MS, 1000.0 / 262.0 );
		r = c.Status();
	}
	CHECK( r.fgActive );
	CHECK( r.activeN >= 2 );
	CHECK( r.steadyMultiplier < 1.2f );
	CHECK( r.steadyMultiplier > 1.0f );
	CHECK( r.generating );

	// A game that stops needing generation clears it as soon as the pacer stops (fgActive false),
	// and a long run of real frames never turns it on again.
	Driver e = FlapDriver();
	for ( int i = 0; i < 8; i++ )
	{
		e.Run( 250 * MS, 1000.0 / 200.0 );
		r = e.Status();
	}
	REQUIRE( r.generating );
	for ( int i = 0; i < 16; i++ )
	{
		e.Run( 250 * MS, 1000.0 / 330.0 );
		r = e.Status();
	}
	CHECK_FALSE( r.fgActive );
	CHECK_FALSE( r.generating );
}

// ---------------------------------------------------------------------------
//  The extra timer's enter / leave band around o == tick
// ---------------------------------------------------------------------------

namespace
{
	// 100 Hz (10 ms tick), a 30 fps game, Target mode, the output timer allowed.
	// Feeds a few seconds at the given target and returns ExtraTimer() afterwards.
	bool ExtraAtTarget( Driver &drv, int nTarget )
	{
		drv.targetFps = nTarget;
		drv.Run( 1500 * MS, 1000.0 / 30.0 );
		return drv.pacer.ExtraTimer();
	}

	Driver ExtraBandDriver()
	{
		Driver drv;
		drv.refreshHz = 100.0;
		drv.mode = Mode::Target;
		drv.pauseAtRefresh = false;
		drv.canExceed = true;
		return drv;
	}
}

TEST_CASE( "pacing: the extra timer leaves when the target drops back to the display refresh", "[pacing]" )
{
	Driver drv = ExtraBandDriver();
	CHECK( ExtraAtTarget( drv, 200 ) );          // o = 5 ms < 0.98 tick: entered
	CHECK( std::fabs( drv.pacer.OutputMs() - 5.0 ) < 1e-6 );
	CHECK_FALSE( ExtraAtTarget( drv, 0 ) );      // display refresh: o == tick, must leave
	CHECK( std::fabs( drv.pacer.OutputMs() - 10.0 ) < 1e-6 );
	CHECK_FALSE( ExtraAtTarget( drv, 100 ) );    // an explicit target of the refresh too
}

TEST_CASE( "pacing: o == tick from the start never enters the extra timer", "[pacing]" )
{
	Driver drv = ExtraBandDriver();
	CHECK_FALSE( ExtraAtTarget( drv, 0 ) );
	CHECK( std::fabs( drv.pacer.OutputMs() - 10.0 ) < 1e-6 );
}

TEST_CASE( "pacing: the extra timer's band does not flap for o a hair under a tick", "[pacing]" )
{
	// 101 fps -> o = 9.90 ms = 0.990 tick: inside the band (0.98 .. 0.995).
	Driver fresh = ExtraBandDriver();
	CHECK_FALSE( ExtraAtTarget( fresh, 101 ) );  // not entered from below the band's top edge
	// 103 fps -> 0.971 tick: enters; then 101 stays in; then the refresh leaves.
	Driver drv = ExtraBandDriver();
	CHECK( ExtraAtTarget( drv, 103 ) );
	CHECK( ExtraAtTarget( drv, 101 ) );
	CHECK_FALSE( ExtraAtTarget( drv, 100 ) );
	CHECK_FALSE( ExtraAtTarget( drv, 101 ) );    // and a hair under does not re-enter
}
