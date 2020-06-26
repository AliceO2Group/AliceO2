// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Definitions.h
/// \brief
///

#ifndef TRACKINGEC0__INCLUDE_CADEFINITIONS_H_
#define TRACKINGEC0__INCLUDE_CADEFINITIONS_H_

// #define _ALLOW_DEBUG_TREES_ITS_ // to allow debug (vertexer only)

#ifndef __OPENCL__
#include <array>
#endif

//#define CA_DEBUG

#ifdef CA_DEBUG
#define CA_DEBUGGER(x) x
#else
#define CA_DEBUGGER(x) \
  do {                 \
  } while (0)
#ifndef NDEBUG
#define NDEBUG 1
#endif
#endif

#if defined(CUDA_ENABLED)
#define TRACKINGEC0__GPU_MODE true
#else
#define TRACKINGEC0__GPU_MODE false
#endif

#if defined(__CUDACC__)
#define TRACKINGEC0__GPU_COMPILING
#endif

#if defined(__CUDA_ARCH__)
#define TRACKINGEC0__GPU_DEVICE
#endif

#if defined(__CUDACC__)

#define GPU_HOST __host__
#define GPU_DEVICE __device__
#define GPU_HOST_DEVICE __host__ __device__
#define GPU_GLOBAL __global__
#define GPU_SHARED __shared__
#define GPU_SYNC __syncthreads()

#define MATH_CEIL ceil

#include <cstddef>
#include "EC0trackingCUDA/Array.h"

template <typename T, std::size_t Size>
using GPUArray = o2::ecl::GPU::Array<T, Size>;

typedef cudaStream_t GPUStream;

#else

#define GPU_HOST
#define GPU_DEVICE
#define GPU_HOST_DEVICE
#define GPU_GLOBAL
#define GPU_SHARED
#define GPU_SYNC

#define MATH_CEIL std::ceil

#ifndef __VECTOR_TYPES_H__

#include "GPUCommonDef.h"

#endif

#ifndef __OPENCL__
#include <cstddef>
template <typename T, size_t Size>
using GPUArray = std::array<T, Size>;
#else
#include "EC0trackingCUDA/Array.h"

template <typename T, size_t Size>
using GPUArray = o2::ecl::GPU::Array<T, Size>;
#endif

typedef struct _dummyStream {
} GPUStream;

#endif

#endif /* TRACKINGEC0__INCLUDE_CADEFINITIONS_H_ */
