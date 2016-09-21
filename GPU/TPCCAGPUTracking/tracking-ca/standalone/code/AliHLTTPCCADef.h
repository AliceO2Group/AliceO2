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

#if defined(__CUDACC__) || defined(__OPENCL__)
#define HLTCA_GPUCODE
#endif //__CUDACC__

#ifdef WIN32
#ifndef R__WIN32
#define R__WIN32
#endif //!R__WIN32
#endif //WIN32

#ifdef R__WIN32
#ifdef INTEL_RUNTIME
#pragma warning(disable : 1786)
#pragma warning(disable : 1478)
#pragma warning(disable : 161)
#pragma warning(disable : 94)
#pragma warning(disable : 1229)
#endif //INTEL_RUNTIME

#ifdef VSNET_RUNTIME
#pragma warning(disable : 4616)
#pragma warning(disable : 4996)
#pragma warning(disable : 1684)
#endif //VSNET_RUNTIME
#endif //R__WIN32

#if defined(HLTCA_STANDALONE) || (defined(HLTCA_GPUCODE) && defined(__OPENCL__) && !defined(HLTCA_HOSTCODE))

// class TObject{};

#ifdef ClassDef
#undef ClassDef
#endif //ClassDef

#ifdef ClassTmp
#undef ClassTmp
#endif //ClassTmp

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
#endif //R__B64
typedef float          Float16_t;   //Float 4 bytes written with a truncated mantissa
typedef double         Double32_t;  //Double 8 bytes in memory, written as a 4 bytes float
typedef char           Text_t;      //General string (char)
typedef unsigned char  Byte_t;      //Byte (8 bits) (unsigned char)
typedef short          Version_t;   //Class version identifier (short)
typedef const char     Option_t;    //Option string (const char)
typedef int            Ssiz_t;      //String size (int)
typedef float          Real_t;      //TVector and TMatrix element type (float)
#if defined(__OPENCL__) && !defined(HLTCA_HOSTCODE)
typedef long Long64_t;
typedef unsigned long ULong64_t;
#else
#if defined(R__WIN32) && !defined(__CINT__)
typedef __int64          Long64_t;  //Portable signed long integer 8 bytes
typedef unsigned __int64 ULong64_t; //Portable unsigned long integer 8 bytes
#else
typedef long long          Long64_t; //Portable signed long integer 8 bytes
typedef unsigned long long ULong64_t;//Portable unsigned long integer 8 bytes
#endif
#endif //R__WIN32 && !__CINT__
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

#define TRACKER_KEEP_TEMPDATA

#else

#include "Rtypes.h"
#include "AliHLTDataTypes.h"

#include "AliHLTTPCCADefinitions.h"

#endif //HLTCA_STANDALONE

#define EXTERN_ROW_HITS
#define TRACKLET_SELECTOR_MIN_HITS(QPT) (QPT > 10 ? 10 : (QPT > 5 ? 15 : 29)) //Minimum hits should depend on Pt, lot Pt tracks can have few hits. 29 Hits default, 15 for < 200 mev, 10 for < 100 mev

#define GLOBAL_TRACKING_RANGE 45					//Number of rows from the upped/lower limit to search for global track candidates in for
#define GLOBAL_TRACKING_Y_RANGE_UPPER_LEFT 0.85		//Inner portion of y-range in slice that is not used in searching for global track candidates
#define GLOBAL_TRACKING_Y_RANGE_LOWER_LEFT 0.85
#define GLOBAL_TRACKING_Y_RANGE_UPPER_RIGHT 0.85
#define GLOBAL_TRACKING_Y_RANGE_LOWER_RIGHT 0.85
//#define GLOBAL_TRACKING_ONLY_UNASSIGNED_HITS		//Only use unassigned clusters in the global tracking step
//#define GLOBAL_TRACKING_EXTRAPOLATE_ONLY			//Do not update the track parameters with new hits from global tracing
#define GLOBAL_TRACKING_MIN_ROWS 10					//Min num of rows an additional global track must span over
#define GLOBAL_TRACKING_MIN_HITS 8					//Min num of hits for an additional global track
#ifdef HLTCA_STANDALONE
#define GLOBAL_TRACKING_MAINTAIN_TRACKLETS			//Maintain tracklets for standalone OpenGL event display
#endif

#define REPRODUCIBLE_CLUSTER_SORTING

#ifdef HLTCA_GPUCODE
#define ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP 6
#define ALIHLTTPCCANEIGHBOURS_FINDER_MAX_FGRIDCONTENTUPDOWN 1000
#define ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS 3500
#define ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM 1536					//Max amount of hits in a row that can be stored in shared memory, make sure this is divisible by ROW ALIGNMENT
#else
#define ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP 20
#define ALIHLTTPCCANEIGHBOURS_FINDER_MAX_FGRIDCONTENTUPDOWN 7000
#define ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS 10000
#define ALIHLTTPCCATRACKLET_CONSTRUCTOR_TEMP_MEM 15000
#endif //HLTCA_GPUCODE

#ifdef HLTCA_GPUCODE
#ifdef __OPENCL__
#ifdef HLTCA_HOSTCODE //HLTCA_GPUCODE & __OPENCL & HLTCA_HOSTCODE

#define GPUdi() //TRIGGER_ERROR_NO_DEVICE_CODE
#define GPUhdi() inline
#define GPUd() //TRIGGER_ERROR_NO_DEVICE_CODE
#define GPUi() inline
#define GPUhd() 
#define GPUh() 
#define GPUg() TRIGGER_ERROR_NO_DEVICE_CODE
#define GPUshared() 
#define GPUsync() 

struct float4 { float x, y, z, w; };
struct float2 { float x; float y; };
struct uchar2 { unsigned char x, y; };
struct short2 { short x, y; };
struct ushort2 { unsigned short x, y; };
struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct int4 { int x, y, z, w; };
struct uint1 { unsigned int x; };
struct uint2 { unsigned int x, y; };
struct uint3 { unsigned int x, y, z; };
struct uint4 { unsigned int x, y, z, w; };
struct uint16 { unsigned int x[16]; };

#else //HLTCA_HOSTCODE : HLTCA_GPUCODE & __OPENCL__ & !HLTCA_HOSTCODE

#define GPUdi() inline
#define GPUhdi() inline
#define GPUd() 
#define GPUi() inline
#define GPUhd() 
#define GPUh() TRIGGER_ERROR_NO_HOST_CODE
#define GPUg() __kernel
#define GPUshared() __local
#define GPUsync() barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)

#endif //HLTCA_HOSTCODE

#else //__OPENCL__ : HLTCA_GPUCODE & !__OPENCL__

#define GPUdi() __device__ inline
#define GPUhdi() __host__ __device__ inline
#define GPUd() __device__
#define GPUi() inline
#define GPUhd() __host__ __device__
#define GPUh() __host__ inline
#define GPUg() __global__
#define GPUshared() __shared__
#define GPUsync() __syncthreads()

#endif //__OPENCL
#else //HLTCA_GPUCODE : !HLTCA_GPUCODE

#define GPUdi()
#define GPUhdi()
#define GPUd()
#define GPUi()
#define GPUhd()
#define GPUg()
#define GPUh()
#define GPUshared()
#define GPUsync()

struct float4 { float x, y, z, w; };
struct float2 { float x; float y; };
struct uchar2 { unsigned char x, y; };
struct short2 { short x, y; };
struct ushort2 { unsigned short x, y; };
struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct int4 { int x, y, z, w; };
struct uint1 { unsigned int x; };
struct uint2 { unsigned int x, y; };
struct uint3 { unsigned int x, y, z; };
struct uint4 { unsigned int x, y, z, w; };
struct uint16 { unsigned int x[16]; };

#ifdef R__WIN32
#include <float.h>

inline bool finite(float x)
{
	return(x <= FLT_MAX);
}
#endif //R__WIN32

#endif //HLTCA_GPUCODE

#if defined(__OPENCL__) && !defined(HLTCA_HOSTCODE)
#define GPUsharedref() GPUshared()
#define GPUglobalref() __global
//#define GPUconstant() __constant //Replace __constant by __global (possibly add const __restrict where possible later!)
#define GPUconstant() GPUglobalref()
#else
#define GPUconstant()
#define GPUsharedref()
#define GPUglobalref()
#endif

enum LocalOrGlobal { Mem_Local, Mem_Global, Mem_Constant, Mem_Plain };
#if defined(__OPENCL__) && !defined(HLTCA_HOSTCODE)
template<LocalOrGlobal, typename L, typename G, typename C, typename P> struct MakeTypeHelper;
template<typename L, typename G, typename C, typename P> struct MakeTypeHelper<Mem_Local, L, G, C, P> { typedef L type; };
template<typename L, typename G, typename C, typename P> struct MakeTypeHelper<Mem_Global, L, G, C, P> { typedef G type; };
template<typename L, typename G, typename C, typename P> struct MakeTypeHelper<Mem_Constant, L, G, C, P> { typedef C type; };
template<typename L, typename G, typename C, typename P> struct MakeTypeHelper<Mem_Plain, L, G, C, P> { typedef P type; };
#define MakeType(base_type) typename MakeTypeHelper<LG, GPUshared() base_type, GPUglobalref() base_type, GPUconstant() base_type, base_type>::type 
#define MEM_CLASS_PRE() template<LocalOrGlobal LG>
#define MEM_LG(type) type<LG>
#define MEM_CLASS_PRE2() template<LocalOrGlobal LG2>
#define MEM_LG2(type) type<LG2>
#define MEM_CLASS_PRE12() template<LocalOrGlobal LG> template<LocalOrGlobal LG2>
#define MEM_CLASS_PRE23() template<LocalOrGlobal LG2, LocalOrGlobal LG3>
#define MEM_LG3(type) type<LG3>
#define MEM_CLASS_PRE234() template<LocalOrGlobal LG2, LocalOrGlobal LG3, LocalOrGlobal LG4>
#define MEM_LG4(type) type<LG4>
#define MEM_GLOBAL(type) type<Mem_Global>
#define MEM_LOCAL(type) type<Mem_Local>
#define MEM_CONSTANT(type) type<Mem_Global>
#define MEM_PLAIN(type) type<Mem_Plain>
#define MEM_TEMPLATE() template <typename T>
#define MEM_TYPE(type) T
#define MEM_TEMPLATE2() template <typename T, typename T2>
#define MEM_TYPE2(type) T2
#define MEM_TEMPLATE3() template <typename T, typename T2, typename T3>
#define MEM_TYPE3(type) T3
#define MEM_TEMPLATE4() template <typename T, typename T2, typename T3, typename T4>
#define MEM_TYPE4(type) T4
//#define MEM_CONSTANT() <Mem_Constant> //Use __global for time being instead of __constant, see above
#else
#define MakeType(base_type) base_type
#define MEM_CLASS_PRE()
#define MEM_LG(type) type
#define MEM_CLASS_PRE2()
#define MEM_LG2(type) type
#define MEM_CLASS_PRE12()
#define MEM_CLASS_PRE23()
#define MEM_LG3(type) type
#define MEM_CLASS_PRE234()
#define MEM_LG4(type) type
#define MEM_GLOBAL(type) type
#define MEM_LOCAL(type) type
#define MEM_CONSTANT(type) type
#define MEM_PLAIN(type) type
#define MEM_TEMPLATE()
#define MEM_TYPE(type) type
#define MEM_TEMPLATE2()
#define MEM_TYPE2(type) type
#define MEM_TEMPLATE3()
#define MEM_TYPE3(type) type
#define MEM_TEMPLATE4()
#define MEM_TYPE4(type) type
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
#endif //!HLTCA_GPUCODE

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

#define UNROLL2(var, code) code;var++;code;var++;
#define UNROLL4(var, code) UNROLL2(var, code) UNROLL2(var, code)
#define UNROLL8(var, code) UNROLL4(var, code) UNROLL4(var, code)
#define UNROLL16(var, code) UNROLL8(var, code) UNROLL8(var, code)
#define UNROLL32(var, code) UNROLL16(var, code) UNROLL16(var, code)

#endif //ALIHLTTPCCADEF_H
