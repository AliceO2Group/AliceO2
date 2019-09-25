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
/// \file Utils.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_GPU_UTILS_H_
#define TRACKINGITSU_INCLUDE_GPU_UTILS_H_

#include "GPUCommonDef.h"
#include "ITStrackingCUDA/Stream.h"

namespace o2
{
namespace its
{
namespace GPU
{

namespace Utils
{

namespace Host
{

#ifdef __CUDACC__
void checkCUDAError(const cudaError_t error, const char* file, const int line);
#endif

dim3 getBlockSize(const int);
dim3 getBlockSize(const int, const int);
dim3 getBlockSize(const int, const int, const int);
dim3 getBlocksGrid(const dim3&, const int);
dim3 getBlocksGrid(const dim3&, const int, const int);

void gpuMalloc(void**, const int);
void gpuFree(void*);
void gpuMemset(void*, int, int);
void gpuMemcpyHostToDevice(void*, const void*, int);
void gpuMemcpyHostToDeviceAsync(void*, const void*, int, Stream&);
void gpuMemcpyDeviceToHost(void*, const void*, int);
void gpuStartProfiler();
void gpuStopProfiler();
} // namespace Host

namespace Device
{
GPUd() int getLaneIndex();
GPUd() int shareToWarp(const int, const int);
GPUd() int gpuAtomicAdd(int*, const int);
} // namespace Device
} // namespace Utils
} // namespace GPU
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_GPU_UTILS_H_ */
