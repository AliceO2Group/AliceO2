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

#if defined(HLTCA_STANDALONE) || defined(HLTCA_GPUCODE)

// class TObject{};

#define ClassDef(name,id)
#define ClassImp(name)

typedef char           Char_t;      //Signed Character 1 byte (char)
typedef unsigned char  UChar_t;     //Unsigned Character 1 byte (unsigned char)
typedef short          Short_t;     //Signed Short integer 2 bytes (short)
typedef unsigned short UShort_t;    //Unsigned Short integer 2 bytes (unsigned short)
#ifdef R__INT16
typedef long           Int_t;       //Signed integer 4 bytes
typedef unsigned long  UInt_t;      //Unsigned integer 4 bytes
#else
typedef int            Int_t;       //Signed integer 4 bytes (int)
typedef unsigned int   UInt_t;      //Unsigned integer 4 bytes (unsigned int)
#endif
#ifdef R__B64    // Note: Long_t and ULong_t are currently not portable types
typedef int            Seek_t;      //File pointer (int)
typedef long           Long_t;      //Signed long integer 8 bytes (long)
typedef unsigned long  ULong_t;     //Unsigned long integer 8 bytes (unsigned long)
#else
typedef int            Seek_t;      //File pointer (int)
typedef long           Long_t;      //Signed long integer 4 bytes (long)
typedef unsigned long  ULong_t;     //Unsigned long integer 4 bytes (unsigned long)
#endif
typedef float          Float_t;     //Float 4 bytes (float)
typedef float          Float16_t;   //Float 4 bytes written with a truncated mantissa
typedef double         Double_t;    //Double 8 bytes
typedef double         Double32_t;  //Double 8 bytes in memory, written as a 4 bytes float
typedef char           Text_t;      //General string (char)
typedef bool           Bool_t;      //Boolean (0=false, 1=true) (bool)
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

#define GPUd() __device__ 
#define GPUhd() __host__ __device__ 
#define GPUh() __host__ inline 
#define GPUg() __global__ 

#define GPUshared() __shared__ 
#define GPUsync() __syncthreads()

__constant__ float4 gAliHLTTPCCATracker[30000/sizeof(float4)];

#else

#define GPUd() 
#define GPUhd()
#define GPUg()
#define GPUh() 
#define GPUshared() 
#define GPUsync() 

struct float2{ float x; float y; };
struct uchar2{ unsigned char x; unsigned char y; };
struct ushort2{ unsigned short x; unsigned short y; };
struct uint1{ unsigned int x; };
struct uint4{ unsigned int x,y,z,w; };

#endif


#endif
