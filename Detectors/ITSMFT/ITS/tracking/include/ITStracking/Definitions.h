// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Definitions.h
/// \brief

#ifndef TRACKINGITS_DEFINITIONS_H_
#define TRACKINGITS_DEFINITIONS_H_

// #define CA_DEBUG
// #define VTX_DEBUG

template <typename T>
void discardResult(const T&)
{
}

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#endif

#ifdef CA_DEBUG
#define CA_DEBUGGER(x) x
#else
#define CA_DEBUGGER(x) \
  do {                 \
  } while (0)
#endif

#if defined(__CUDA_ARCH__) // ????
#define TRACKINGITSU_GPU_DEVICE
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define MATH_CEIL ceil

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstddef>
#endif
#include "../GPU/ITStrackingGPU/Array.h"

template <typename T, size_t Size>
using GPUArray = o2::its::gpu::Array<T, Size>;

#ifdef __CUDACC__
#define GPU_ARCH "CUDA"

typedef cudaStream_t GPUStream;
inline int getGPUCores(const int major, const int minor)
{
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] =
    {
      {0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48},  // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
      {0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
      {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
      {0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
      {0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
      {0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
      {0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
      {0x60, 64},  // Pascal Generation (SM 6.0) GP100 class
      {0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
      {0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
      {0x70, 64},  // Volta Generation (SM 7.0) GV100 class
      {0x72, 64},  // Volta Generation (SM 7.2) GV10B class
      {0x75, 64},  // Turing Generation (SM 7.5) TU1xx class
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run properly
  return nGpuArchCoresPerSM[index - 1].Cores;
}
inline int getGPUMaxThreadsPerComputingUnit()
{
  return 8;
}

#else // __HIPCC__
#define GPU_ARCH "HIP"
typedef hipStream_t GPUStream;
inline int getGPUCores(const int major, const int minor)
{
  // Hardcoded result for AMD RADEON WX 9100, to be decided if and how determine this paramter
  return 4096;
}

inline int getGPUMaxThreadsPerComputingUnit()
{
  return 8;
}
#endif

#else
#define MATH_CEIL std::ceil
#ifndef __VECTOR_TYPES_H__
#include "GPUCommonDef.h"
#endif
#ifndef __OPENCL__
#include <cstddef>
template <typename T, size_t Size>
using GPUArray = std::array<T, Size>;
#else
#include "../GPU/ITStrackingGPU/Array.h"
template <typename T, size_t Size>
using GPUArray = o2::its::gpu::Array<T, Size>;
#endif

typedef struct _dummyStream {
} GPUStream;
#endif

#endif
