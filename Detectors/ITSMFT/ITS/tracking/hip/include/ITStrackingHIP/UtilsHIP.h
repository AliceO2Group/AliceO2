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

namespace Utils
{

namespace HostHIP
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
} // namespace Host
//
namespace DeviceHIP
{
GPUd() int getLaneIndex();
GPUd() int shareToWarp(const int, const int);
GPUd() int gpuAtomicAdd(int*, const int);
} // namespace Device
} // namespace Utils
} // namespace gpu
} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKING_INCLUDE_UTILS_HIP_H_ */
