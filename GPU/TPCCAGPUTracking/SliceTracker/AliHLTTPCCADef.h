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

#if defined(__CUDACC__) || defined(__OPENCL__)
  #define HLTCA_GPUCODE //Executed on GPU
#endif

#if defined(WIN32) && defined(INTEL_RUNTIME)
  #pragma warning(disable : 1786)
  #pragma warning(disable : 1478)
  #pragma warning(disable : 161)
  #pragma warning(disable : 94)
  #pragma warning(disable : 1229)
#endif

#if defined(WIN32) && defined(VSNET_RUNTIME)
  #pragma warning(disable : 4616)
  #pragma warning(disable : 4996)
  #pragma warning(disable : 1684)
#endif

#if defined(HLTCA_STANDALONE) || (defined(HLTCA_GPUCODE) && defined(__OPENCL__) && !defined(HLTCA_HOSTCODE)) || !defined(HLTCA_BUILD_ALIROOT_LIB) 

  #if !defined(ROOT_Rtypes) && !defined(__CLING__)
    #define ClassDef(name,id)
    #define ClassImp(name)
  #endif

  #define TRACKER_KEEP_TEMPDATA
#endif //HLTCA_STANDALONE

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

#include "AliHLTTPCCASettings.h"
#define CALINK_INVAL ((calink) -1)
struct cahit2{cahit x, y;};

#ifdef HLTCA_FULL_CLUSTERDATA
  #define HLTCA_EVDUMP_FILE "event_full"
#else
  #define HLTCA_EVDUMP_FILE "event"
#endif

#ifdef GPUseStatError
  #define GMPropagatePadRowTime
#endif

#ifdef GMPropagatePadRowTime //Needs full clusterdata
  #define HLTCA_FULL_CLUSTERDATA
#endif

#if defined(HLTCA_STANDALONE) | defined(HLTCA_GPUCODE) //No support for Full Field Propagator or Statistical errors
  #ifdef GMPropagatorUseFullField
    #undef GMPropagatorUseFullField
  #endif
  #ifdef GPUseStatError
    #undef GPUseStatError
  #endif
#endif

#ifdef EXTERN_ROW_HITS
  #define GETRowHit(iRow) tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr]
  #define SETRowHit(iRow, val) tracker.TrackletRowHits()[iRow * s.fNTracklets + r.fItr] = val
#else
  #define GETRowHit(iRow) tracklet.RowHit(iRow)
  #define SETRowHit(iRow, val) tracklet.SetRowHit(iRow, val)
#endif

#ifdef HLTCA_GPUCODE
  #define MAKESharedRef(vartype, varname, varglobal, varshared) const GPUsharedref() MEM_LOCAL(vartype) &varname = varshared;
#else
  #define MAKESharedRef(vartype, varname, varglobal, varshared) const GPUglobalref() MEM_GLOBAL(vartype) &varname = varglobal;
#endif

#ifdef HLTCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
  #define TEXTUREFetchCons(type, texture, address, entry) tex1Dfetch(texture, ((char*) address - tracker.Data().GPUTextureBase()) / sizeof(type) + entry);
#else
  #define TEXTUREFetchCons(type, texture, address, entry) address[entry];
#endif

#endif //ALIHLTTPCCADEF_H

#ifdef HLTCA_CADEBUG
  #ifdef CADEBUG
    #undef CADEBUG
  #endif
  #ifdef HLTCA_CADEBUG_ENABLED
    #undef HLTCA_CADEBUG_ENABLED
  #endif
  #if HLTCA_CADEBUG == 1 && !defined(HLTCA_GPUCODE)
    #define CADEBUG(...) __VA_ARGS__
    #define HLTCA_CADEBUG_ENABLED
  #endif
  #undef HLTCA_CADEBUG
#endif

#ifndef CADEBUG
  #define CADEBUG(...)
#endif
