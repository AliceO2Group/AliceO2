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

#ifdef __HIPCC__
#include "hip/hip_runtime.h"
#else
#include <cuda.h>
#endif

#include <iostream>

#include "DeviceInterface/GPUInterface.h"
#include "DeviceInterface/GPUInterfaceConfigParam.h"

#define gpuCheckError(x)                \
  {                                     \
    gpuAssert((x), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    if (abort) {
      throw std::runtime_error("GPU assert failed.");
    }
  }
}

namespace o2::vertexing::device
{
GPUInterface* GPUInterface::sGPUInterface = nullptr;
GPUInterface* GPUInterface::Instance()
{
  if (sGPUInterface == nullptr) {
    sGPUInterface = new GPUInterface(o2::vertexing::device::GPUInterfaceParamConfig::Instance().streamPoolSize);
  }
  return sGPUInterface;
}

void GPUInterface::register(void* addr, size_t bufferSize)
{
  gpuCheckError(cudaHostRegister(addr, bufferSize, cudaHostRegisterDefault));
}

void GPUInterface::allocAsync(void** addrPtr, size_t bufferSize, unsigned short streamId) {

}
} // namespace o2::vertexing::device