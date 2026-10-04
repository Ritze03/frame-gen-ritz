#pragma once
// minitest.h -- a tiny stand-in for the part of Catch2 that gpu/pacingtest.cpp uses,
// so the library's CPU-only tests need no dependency. TEST_CASE( name, tags ),
// REQUIRE / REQUIRE_FALSE (abort the test case), CHECK / CHECK_FALSE (record and
// carry on), INFO( stream expression ) (printed with any failure in its scope).
// Include from exactly one translation unit: it defines main().
//   pacingtest              run everything
//   pacingtest <substring>  run the cases whose name or tag contains it
//   pacingtest --list       list the cases
// Exit status 0 = all passed.

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace minitest
{
	struct Case
	{
		const char *name;
		const char *tags;
		void ( *fn )();
	};

	inline std::vector<Case> &Cases()
	{
		static std::vector<Case> v;
		return v;
	}

	struct Registrar
	{
		Registrar( const char *name, const char *tags, void ( *fn )() ) { Cases().push_back( { name, tags, fn } ); }
	};

	struct Abort {};   // thrown by a failed REQUIRE

	inline std::vector<const std::string *> &Infos()
	{
		static std::vector<const std::string *> v;
		return v;
	}

	// INFO's scope: pushed on construction, popped at the end of the enclosing block.
	struct Info
	{
		std::string msg;
		explicit Info( std::string m ) : msg( std::move( m ) ) { Infos().push_back( &msg ); }
		~Info() { Infos().pop_back(); }
		Info( const Info & ) = delete;
		Info &operator=( const Info & ) = delete;
	};

	inline int &Failures()
	{
		static int n = 0;
		return n;
	}

	inline int &Checks()
	{
		static int n = 0;
		return n;
	}

	inline void Fail( const char *file, int line, const char *macro, const char *expr )
	{
		Failures()++;
		fprintf( stderr, "  FAILED %s( %s )\n    at %s:%d\n", macro, expr, file, line );
		for ( const std::string *p : Infos() )
			fprintf( stderr, "    with message: %s\n", p->c_str() );
	}

	inline void Check( bool bOk, const char *file, int line, const char *macro, const char *expr, bool bRequire )
	{
		Checks()++;
		if ( bOk )
			return;
		Fail( file, line, macro, expr );
		if ( bRequire )
			throw Abort();
	}
}

#define MT_CAT2( a, b ) a##b
#define MT_CAT( a, b ) MT_CAT2( a, b )

#define TEST_CASE( name, tags ) \
	static void MT_CAT( mt_case_, __LINE__ )(); \
	static const minitest::Registrar MT_CAT( mt_reg_, __LINE__ )( name, tags, &MT_CAT( mt_case_, __LINE__ ) ); \
	static void MT_CAT( mt_case_, __LINE__ )()

#define CHECK( ... )         minitest::Check( bool( __VA_ARGS__ ), __FILE__, __LINE__, "CHECK", #__VA_ARGS__, false )
#define CHECK_FALSE( ... )   minitest::Check( !bool( __VA_ARGS__ ), __FILE__, __LINE__, "CHECK_FALSE", #__VA_ARGS__, false )
#define REQUIRE( ... )       minitest::Check( bool( __VA_ARGS__ ), __FILE__, __LINE__, "REQUIRE", #__VA_ARGS__, true )
#define REQUIRE_FALSE( ... ) minitest::Check( !bool( __VA_ARGS__ ), __FILE__, __LINE__, "REQUIRE_FALSE", #__VA_ARGS__, true )
#define INFO( ... ) \
	const minitest::Info MT_CAT( mt_info_, __LINE__ )( ( [&]() { std::ostringstream mt_os; mt_os << __VA_ARGS__; return mt_os.str(); } )() )

int main( int argc, char **argv )
{
	const char *pszFilter = argc > 1 ? argv[ 1 ] : nullptr;
	if ( pszFilter && !strcmp( pszFilter, "--list" ) )
	{
		for ( const minitest::Case &c : minitest::Cases() )
			printf( "%s %s\n", c.name, c.tags );
		return 0;
	}
	int nRun = 0, nFailed = 0;
	for ( const minitest::Case &c : minitest::Cases() )
	{
		if ( pszFilter && !strstr( c.name, pszFilter ) && !strstr( c.tags, pszFilter ) )
			continue;
		nRun++;
		const int nBefore = minitest::Failures();
		try
		{
			c.fn();
		}
		catch ( const minitest::Abort & )
		{
		}
		const bool bOk = minitest::Failures() == nBefore;
		nFailed += bOk ? 0 : 1;
		printf( "%s  %s\n", bOk ? "ok  " : "FAIL", c.name );
	}
	printf( "\n%d test cases, %d failed; %d assertions, %d failed\n", nRun, nFailed, minitest::Checks(), minitest::Failures() );
	return nFailed ? 1 : 0;
}
