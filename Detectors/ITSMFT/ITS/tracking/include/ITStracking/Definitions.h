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

#ifndef TRACKINGITSU_INCLUDE_CADEFINITIONS_H_
#define TRACKINGITSU_INCLUDE_CADEFINITIONS_H_

#define _ALLOW_DEBUG_TREES_ITS_ // to allow debug (vertexer only)

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
#define TRACKINGITSU_GPU_MODE true
#else
#define TRACKINGITSU_GPU_MODE false
#endif

#if defined(__CUDACC__)
#define TRACKINGITSU_GPU_COMPILING
#endif

#if defined(__CUDA_ARCH__)
#define TRACKINGITSU_GPU_DEVICE
#endif

#if defined(__CUDACC__)

#define GPU_HOST __host__
#define GPU_DEVICE __device__
#define GPU_HOST_DEVICE __host__ __device__
#define GPU_GLOBAL __global__
#define GPU_SHARED __shared__
#define GPU_SYNC __syncthreads()

#define MATH_CEIL ceil

#include "ITStrackingCUDA/Array.h"

template <typename T, std::size_t Size>
using GPUArray = o2::its::GPU::Array<T, Size>;

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
template <typename T, size_t Size>
using GPUArray = std::array<T, Size>;
#else
#include "ITStrackingCUDA/Array.h"

template <typename T, size_t Size>
using GPUArray = o2::its::GPU::Array<T, Size>;
#endif

typedef struct _dummyStream {
} GPUStream;

#endif

#endif /* TRACKINGITSU_INCLUDE_CADEFINITIONS_H_ */
