//-*- Mode: C++ -*-

//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCADEF_H
#define ALIHLTTPCCADEF_H


/**
 * Definitions needed for AliHLTTPCCATracker
 *
 */

//#define HLTCA_STANDALONE // compilation w/o root
#define HLTCA_INTERNAL_PERFORMANCE

#ifdef __CUDACC__
#define HLTCA_GPUCODE
#endif

#ifdef WIN32
#ifndef R__WIN32
#define R__WIN32
#endif
#endif

#if defined(R__WIN32)
#ifdef INTEL_RUNTIME
#pragma warning(disable : 1786)
#pragma warning(disable : 1478)
#pragma warning(disable : 161)
#endif

#ifdef VSNET_RUNTIME
#pragma warning(disable : 4616)
#pragma warning(disable : 4996)
#pragma warning(disable : 1684)
#endif
#endif

#if defined(HLTCA_STANDALONE) || defined(HLTCA_GPUCODE)

// class TObject{};

#define ClassDef(name,id)
#define ClassImp(name)

typedef unsigned char  UChar_t;     //Unsigned Character 1 byte (unsigned char)
#ifdef R__B64    // Note: Long_t and ULong_t are currently not portable types
typedef int            Seek_t;      //File pointer (int)
typedef long           Long_t;      //Signed long integer 8 bytes (long)
typedef unsigned long  ULong_t;     //Unsigned long integer 8 bytes (unsigned long)
#else
typedef int            Seek_t;      //File pointer (int)
typedef long           Long_t;      //Signed long integer 4 bytes (long)
typedef unsigned long  ULong_t;     //Unsigned long integer 4 bytes (unsigned long)
#endif
typedef float          Float16_t;   //Float 4 bytes written with a truncated mantissa
typedef double         Double32_t;  //Double 8 bytes in memory, written as a 4 bytes float
typedef char           Text_t;      //General string (char)
typedef unsigned char  Byte_t;      //Byte (8 bits) (unsigned char)
typedef short          Version_t;   //Class version identifier (short)
typedef const char     Option_t;    //Option string (const char)
typedef int            Ssiz_t;      //String size (int)
typedef float          Real_t;      //TVector and TMatrix element type (float)
#if defined(R__WIN32) && !defined(__CINT__)
typedef __int64          Long64_t;  //Portable signed long integer 8 bytes
typedef unsigned __int64 ULong64_t; //Portable unsigned long integer 8 bytes
#else
typedef long long          Long64_t; //Portable signed long integer 8 bytes
typedef unsigned long long ULong64_t;//Portable unsigned long integer 8 bytes
#endif
typedef double         Axis_t;      //Axis values type (double)
typedef double         Stat_t;      //Statistics type (double)
typedef short          Font_t;      //Font number (short)
typedef short          Style_t;     //Style number (short)
typedef short          Marker_t;    //Marker number (short)
typedef short          Width_t;     //Line width (short)
typedef short          Color_t;     //Color number (short)
typedef short          SCoord_t;    //Screen coordinates (short)
typedef double         Coord_t;     //Pad world coordinates (double)
typedef float          Angle_t;     //Graphics angle (float)
typedef float          Size_t;      //Attribute size (float)

#else

#include "Rtypes.h"
#include "AliHLTDataTypes.h"

namespace AliHLTTPCCADefinitions
{
  extern const AliHLTComponentDataType fgkTrackletsDataType;
}

#endif

#ifdef HLTCA_GPUCODE
#define ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP 5
#define ALIHLTTPCCANEIGHBOURS_FINDER_MAX_FGRIDCONTENTUPDOWN 700
#define ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS 3500
#define ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM 656					//Max amount of hits in a row that can be stored in shared memory, make sure this is divisible by 4
#else
#define ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP 20
#define ALIHLTTPCCANEIGHBOURS_FINDER_MAX_FGRIDCONTENTUPDOWN 7000
#define ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS 10000
#define ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM 5000
#endif

#ifdef HLTCA_GPUCODE

#define GPUd() __device__
#define GPUhd() __host__ __device__
#define GPUh() __host__ inline
#define GPUg() __global__

#define GPUshared() __shared__
#define GPUsync() __syncthreads()

__constant__ float4 gAliHLTTPCCATracker[30000/sizeof( float4 )];

#else

#define GPUd()
#define GPUhd()
#define GPUg()
#define GPUh()
#define GPUshared()
#define GPUsync()

struct float2 { float x; float y; };
struct uchar2 { unsigned char x; unsigned char y; };
struct ushort2 { unsigned short x; unsigned short y; };
struct uint1 { unsigned int x; };
struct uint4 { unsigned int x, y, z, w; };

#ifdef R__WIN32
#include <float.h>

inline bool finite(float x)
{
	return(x <= FLT_MAX);
}
#endif

#endif

/*
 * Helper for compile-time verification of correct API usage
 */

#ifndef HLTCA_GPUCODE
namespace
{
  template<bool> struct HLTTPCCA_STATIC_ASSERT_FAILURE;
  template<> struct HLTTPCCA_STATIC_ASSERT_FAILURE<true> {};
}

#define HLTTPCCA_STATIC_ASSERT_CONCAT_HELPER(a, b) a##b
#define HLTTPCCA_STATIC_ASSERT_CONCAT(a, b) HLTTPCCA_STATIC_ASSERT_CONCAT_HELPER(a, b)
#define STATIC_ASSERT(cond, msg) \
  typedef HLTTPCCA_STATIC_ASSERT_FAILURE<cond> HLTTPCCA_STATIC_ASSERT_CONCAT(_STATIC_ASSERTION_FAILED_##msg, __LINE__); \
  HLTTPCCA_STATIC_ASSERT_CONCAT(_STATIC_ASSERTION_FAILED_##msg, __LINE__) Error_##msg; \
  (void) Error_##msg
#else
#define STATIC_ASSERT(a, b)
#endif

namespace
{
  template<typename T1>
  void UNUSED_PARAM1( const T1 & ) {}
  template<typename T1, typename T2>
  void UNUSED_PARAM2( const T1 &, const T2 & ) {}
  template<typename T1, typename T2, typename T3>
  void UNUSED_PARAM3( const T1 &, const T2 &, const T3 & ) {}
  template<typename T1, typename T2, typename T3, typename T4>
  void UNUSED_PARAM4( const T1 &, const T2 &, const T3 &, const T4 & ) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5>
  void UNUSED_PARAM5( const T1 &, const T2 &, const T3 &, const T4 &, const T5 & ) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
  void UNUSED_PARAM6( const T1 &, const T2 &, const T3 &, const T4 &, const T5 &, const T6 & ) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
  void UNUSED_PARAM7( const T1 &, const T2 &, const T3 &, const T4 &, const T5 &, const T6 &, const T7 & ) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
  void UNUSED_PARAM8( const T1 &, const T2 &, const T3 &, const T4 &, const T5 &, const T6 &, const T7 &, const T8 & ) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
  void UNUSED_PARAM9( const T1 &, const T2 &, const T3 &, const T4 &, const T5 &, const T6 &, const T7 &, const T8 &, const T9 & ) {}
}

#endif
