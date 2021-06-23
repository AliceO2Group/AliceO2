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
/// \file UtilsHIP.h
/// \brief
///

#ifndef O2_ITS_TRACKING_INCLUDE_UTILS_HIP_H_
#define O2_ITS_TRACKING_INCLUDE_UTILS_HIP_H_

#include "GPUCommonDef.h"
#include "ITStrackingHIP/StreamHIP.h"
#include <hip/hip_runtime_api.h>

namespace o2
{
namespace its
{
namespace gpu
{

namespace utils
{

namespace host_hip
{

#ifdef __HIPCC__
void checkHIPError(const hipError_t error, const char* file, const int line);
#endif

dim3 getBlockSize(const int);
dim3 getBlockSize(const int, const int);
dim3 getBlockSize(const int, const int, const int);
dim3 getBlocksGrid(const dim3&, const int);
dim3 getBlocksGrid(const dim3&, const int, const int);
//
void gpuMalloc(void**, const int);
void gpuFree(void*);
void gpuMemset(void*, int, int);
void gpuMemcpyHostToDevice(void*, const void*, int);
void gpuMemcpyHostToDeviceAsync(void*, const void*, int, hipStream_t&);
void gpuMemcpyDeviceToHost(void*, const void*, int);
// void gpuStartProfiler();
// void gpuStopProfiler();
} // namespace host_hip
//
namespace device_hip
{
GPUd() int getLaneIndex();
GPUd() int shareToWarp(const int, const int);
GPUd() int gpuAtomicAdd(int*, const int);
} // namespace device_hip
} // namespace utils
} // namespace gpu
} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKING_INCLUDE_UTILS_HIP_H_ */
