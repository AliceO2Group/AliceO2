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
#include "GPUCommonLogger.h"

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
      auto fmt = [](size_t bytes) -> const char* {
        static char buf[64];
        double mb = (double)bytes / (1024.0 * 1024.0);
        int n = snprintf(buf, sizeof(buf), "%zu bytes (%.3f MB)", bytes, mb);
        (void)n;
        return buf;
      };

      // Element counts (number of array entries, not bytes)
      size_t elemsClusterFlags = (mClusterFlags && mNnClusterizerBatchedMode > 0) ? (size_t)2 * mNnClusterizerBatchedMode : 0;
      size_t elemsInput16 = (mInputData_16 && mNnClusterizerBatchedMode > 0 && mNnClusterizerElementSize > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerElementSize : 0;
      size_t elemsInput32 = (mInputData_32 && mNnClusterizerBatchedMode > 0 && mNnClusterizerElementSize > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerElementSize : 0;
      size_t elemsProb16 = (mModelProbabilities_16 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelClassNumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes : 0;
      size_t elemsProb32 = (mModelProbabilities_32 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelClassNumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes : 0;
      size_t elemsReg1_16 = (mOutputDataReg1_16 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelReg1NumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes : 0;
      size_t elemsReg2_16 = (mOutputDataReg2_16 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelReg2NumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes : 0;
      size_t elemsReg1_32 = (mOutputDataReg1_32 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelReg1NumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes : 0;
      size_t elemsReg2_32 = (mOutputDataReg2_32 && mNnClusterizerBatchedMode > 0 && mNnClusterizerModelReg2NumOutputNodes > 0) ? (size_t)mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes : 0;
      size_t elemsOutputDataClass = (mOutputDataClass && mNnClusterizerTotalClusters > 0) ? (size_t)mNnClusterizerTotalClusters : 0;

      // Byte sizes
      size_t szClusterFlags = elemsClusterFlags * sizeof(int8_t);
      size_t szInput16 = elemsInput16 * sizeof(OrtDataType::Float16_t);
      size_t szInput32 = elemsInput32 * sizeof(float);
      size_t szProb16 = elemsProb16 * sizeof(OrtDataType::Float16_t);
      size_t szProb32 = elemsProb32 * sizeof(float);
      size_t szReg1_16 = elemsReg1_16 * sizeof(OrtDataType::Float16_t);
      size_t szReg2_16 = elemsReg2_16 * sizeof(OrtDataType::Float16_t);
      size_t szReg1_32 = elemsReg1_32 * sizeof(float);
      size_t szReg2_32 = elemsReg2_32 * sizeof(float);
      size_t szOutputDataClass = elemsOutputDataClass * sizeof(int32_t);

      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") Pointers set for clusterizer with memoryID " << mMemoryId << " deviceID " << mDeviceId << " and sector " << mISector;
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mOutputDataClass pointer: " << mOutputDataClass
                << " | elements=" << elemsOutputDataClass << " (= mNnClusterizerTotalClusters)"
                << " | " << fmt(szOutputDataClass);
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mClusterFlags pointer: " << static_cast<const void*>(mClusterFlags)
                << " | elements=" << elemsClusterFlags << " (= 2 * mNnClusterizerBatchedMode)"
                << " | " << fmt(szClusterFlags);
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mInputData_16 pointer: " << mInputData_16
                << " | elements=" << elemsInput16 << " (= mNnClusterizerBatchedMode * mNnClusterizerElementSize)"
                << " | " << fmt(szInput16);
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mModelProbabilities_16 pointer: " << mModelProbabilities_16
                << " | elements=" << elemsProb16 << " (= mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes)"
                << " | " << fmt(szProb16);
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mOutputDataReg1_16 pointer: " << mOutputDataReg1_16
                << " | elements=" << elemsReg1_16 << " (= mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes)"
                << " | " << fmt(szReg1_16);
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mOutputDataReg2_16 pointer: " << mOutputDataReg2_16
                << " | elements=" << elemsReg2_16 << " (= mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes)"
                << " | " << fmt(szReg2_16);
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mInputData_32 pointer: " << mInputData_32
                << " | elements=" << elemsInput32 << " (= mNnClusterizerBatchedMode * mNnClusterizerElementSize)"
                << " | " << fmt(szInput32);
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mModelProbabilities_32 pointer: " << mModelProbabilities_32
                << " | elements=" << elemsProb32 << " (= mNnClusterizerBatchedMode * mNnClusterizerModelClassNumOutputNodes)"
                << " | " << fmt(szProb32);
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mOutputDataReg1_32 pointer: " << mOutputDataReg1_32
                << " | elements=" << elemsReg1_32 << " (= mNnClusterizerBatchedMode * mNnClusterizerModelReg1NumOutputNodes)"
                << " | " << fmt(szReg1_32);
      LOG(info) << "(NNCLUS, GPUTPCNNClusterizer, this=" << this << ") mOutputDataReg2_32 pointer: " << mOutputDataReg2_32
                << " | elements=" << elemsReg2_32 << " (= mNnClusterizerBatchedMode * mNnClusterizerModelReg2NumOutputNodes)"
                << " | " << fmt(szReg2_32);
    }
    // Compute allocated bytes (difference between advanced pointer and start pointer)
    size_t allocatedBytes = static_cast<size_t>(reinterpret_cast<uintptr_t>(mem) - reinterpret_cast<uintptr_t>(startMem));
    double allocatedMB = static_cast<double>(allocatedBytes) / (1024.0 * 1024.0);
    {
      char allocMsg[256];
      int nn = snprintf(allocMsg, sizeof(allocMsg),
                        "(NNCLUS, GPUTPCNNClusterizer, this=%p) Total scratch allocation in setIOPointers: %zu bytes (%.3f MB)",
                        (void*)this, (size_t)allocatedBytes, allocatedMB);
      (void)nn;
      LOG(info) << allocMsg;
    }
  }

  return mem;
}

void GPUTPCNNClusterizer::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  int32_t memType = GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_STACK;
  mMemoryId = mRec->RegisterMemoryAllocation(this, &GPUTPCNNClusterizer::setIOPointers, memType, "TPCNNClusterer", GPUMemoryReuse{GPUMemoryReuse::REUSE_1TO1, GPUMemoryReuse::NNClusterer, (uint16_t)(mISector % mRec->GetProcessingSettings().nTPCClustererLanes)});
}
