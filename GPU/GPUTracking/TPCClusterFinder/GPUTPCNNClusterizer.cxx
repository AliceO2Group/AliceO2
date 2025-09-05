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

/// \file GPUTPCNNClusterizer.cxx
/// \author Christian Sonnabend

#include "GPUReconstruction.h"
#include "ML/3rdparty/GPUORTFloat16.h"
#include "GPUTPCNNClusterizer.h"
#include "GPUSettings.h"
#include "GPULogging.h"
#include <cstdint>   // uintptr_t
#include <iomanip>   // setprecision
#include <ostream>
#include <sstream>

using namespace o2::gpu;

void GPUTPCNNClusterizer::InitializeProcessor() {}

void GPUTPCNNClusterizer::SetMaxData(const GPUTrackingInOutPointers& io) {}

void* GPUTPCNNClusterizer::setIOPointers(void* mem)
{
  // Keep track of the start address to compute how much memory we assign
  void* startMem = mem;
  if (mNnClusterizerBatchedMode > 0) {
    if (mNnInferenceInputDType == 0 && mNnClusterizerElementSize > 0) {
      computePointerWithAlignment(mem, mInputData_16, mNnClusterizerBatchedMode * mNnClusterizerElementSize);
    } else if (mNnInferenceInputDType == 1 && mNnClusterizerElementSize > 0) {
      computePointerWithAlignment(mem, mInputData_32, mNnClusterizerBatchedMode * mNnClusterizerElementSize);
    }
    computePointerWithAlignment(mem, mClusterFlags, 2 * mNnClusterizerBatchedMode);

    if (mNnInferenceOutputDType == 0 && mNnClusterizerElementSize > 0) {
      if (mNnClusterizerModelClassNumOutputNodes > 0) {
        computePointerWithAlignment(mem, mModelProbabilities_16, mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes);
      }
      if (!mNnClusterizerUseCfRegression) {
        if (mNnClusterizerModelReg1NumOutputNodes > 0) {
          computePointerWithAlignment(mem, mOutputDataReg1_16, mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes);
        }
        if (mNnClusterizerModelReg2NumOutputNodes > 0) {
          computePointerWithAlignment(mem, mOutputDataReg2_16, mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes);
        }
      }
    } else if (mNnInferenceOutputDType == 1 && mNnClusterizerElementSize > 0) {
      if (mNnClusterizerModelClassNumOutputNodes > 0) {
        computePointerWithAlignment(mem, mModelProbabilities_32, mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes);
      }
      if (!mNnClusterizerUseCfRegression) {
        if (mNnClusterizerModelReg1NumOutputNodes > 0) {
          computePointerWithAlignment(mem, mOutputDataReg1_32, mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes);
        }
        if (mNnClusterizerModelReg2NumOutputNodes > 0) {
          computePointerWithAlignment(mem, mOutputDataReg2_32, mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes);
        }
      }
    }
  }
  if (mNnClusterizerTotalClusters > 0) {
    computePointerWithAlignment(mem, mOutputDataClass, mNnClusterizerTotalClusters);
  }

  if (mNnClusterizerVerbosity > 2) {
    if (mNnClusterizerVerbosity > 3) {
      auto fmt = [](size_t bytes) {
        std::ostringstream os;
        double mb = bytes / (1024.0 * 1024.0);
        os << bytes << " bytes (" << std::fixed << std::setprecision(3) << mb << " MB)";
        return os.str();
      };

      // Safely compute sizes only if corresponding pointer was allocated (and dimensions positive)
      size_t szClusterFlags = (mClusterFlags && mNnClusterizerBatchedMode > 0) ? (size_t)2 * mNnClusterizerBatchedMode * sizeof(int8_t) : 0;
      size_t szInput16 = (mInputData_16 && mNnClusterizerBatchedMode > 0 && mNnClusterizerElementSize > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerElementSize * sizeof(OrtDataType::Float16_t) : 0;
      size_t szInput32 = (mInputData_32 && mNnClusterizerBatchedMode > 0 && mNnClusterizerElementSize > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerElementSize * sizeof(float) : 0;
      size_t szProb16 = (mModelProbabilities_16 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelClassNumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes * sizeof(OrtDataType::Float16_t) : 0;
      size_t szProb32 = (mModelProbabilities_32 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelClassNumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes * sizeof(float) : 0;
      size_t szReg1_16 = (mOutputDataReg1_16 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelReg1NumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes * sizeof(OrtDataType::Float16_t) : 0;
      size_t szReg2_16 = (mOutputDataReg2_16 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelReg2NumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes * sizeof(OrtDataType::Float16_t) : 0;
      size_t szReg1_32 = (mOutputDataReg1_32 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelReg1NumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes * sizeof(float) : 0;
      size_t szReg2_32 = (mOutputDataReg2_32 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelReg2NumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes * sizeof(float) : 0;
      size_t szOutputDataClass = (mOutputDataClass && mNnClusterizerTotalClusters > 0) ? (size_t)mNnClusterizerTotalClusters * sizeof(int32_t) : 0;

      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") Pointers set for clusterizer with memoryID " << mMemoryId << " deviceID " << mDeviceId << " and sector " << mISector;
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mOutputDataClass pointer: " << mOutputDataClass << " | " << fmt(szOutputDataClass) << " MB";
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mClusterFlags pointer: " << static_cast<const void*>(mClusterFlags) << " | " << fmt(szClusterFlags) << " MB";
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mInputData_16 pointer: " << mInputData_16 << " | " << fmt(szInput16) << " MB";
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mModelProbabilities_16 pointer: " << mModelProbabilities_16 << " | " << fmt(szProb16) << " MB";
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mOutputDataReg1_16 pointer: " << mOutputDataReg1_16 << " | " << fmt(szReg1_16) << " MB";
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mOutputDataReg2_16 pointer: " << mOutputDataReg2_16 << " | " << fmt(szReg2_16) << " MB";
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mInputData_32 pointer: " << mInputData_32 << " | " << fmt(szInput32) << " MB";
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mModelProbabilities_32 pointer: " << mModelProbabilities_32 << " | " << fmt(szProb32) << " MB";
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mOutputDataReg1_32 pointer: " << mOutputDataReg1_32 << " | " << fmt(szReg1_32) << " MB";
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mOutputDataReg2_32 pointer: " << mOutputDataReg2_32 << " | " << fmt(szReg2_32) << " MB";
    }
    // Compute allocated bytes (difference between advanced pointer and start pointer)
    size_t allocatedBytes = static_cast<size_t>(reinterpret_cast<uintptr_t>(mem) - reinterpret_cast<uintptr_t>(startMem));
    double allocatedMB = static_cast<double>(allocatedBytes) / (1024.0 * 1024.0);
    LOG(info) << std::fixed << std::setprecision(3)
              << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") Total scratch allocation in setIOPointers: " << allocatedBytes
              << " bytes (" << allocatedMB << " MB)";
  }

  return mem;
}

void GPUTPCNNClusterizer::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  int32_t memType = GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK;
  mMemoryId = mRec->RegisterMemoryAllocation(this, &GPUTPCNNClusterizer::setIOPointers, memType, "TPCNNClusterer", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::NNClusterer, (uint16_t)(mISector % mRec->GetProcessingSettings().nTPCClustererLanes)});
}
