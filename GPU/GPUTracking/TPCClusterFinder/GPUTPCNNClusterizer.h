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

/// \file GPUTPCNNClusterizer.h
/// \author Christian Sonnabend

#ifndef O2_GPUTPCNNCLUSTERIZER_H
#define O2_GPUTPCNNCLUSTERIZER_H

#include "CfChargePos.h"
#include "GPUProcessor.h"

namespace o2::OrtDataType
{
struct Float16_t;
}

namespace o2::gpu
{

class GPUTPCNNClusterizer : public GPUProcessor
{
 public:
  GPUTPCNNClusterizer() = default;
  void* setIOPointers(void*);
  void RegisterMemoryAllocation();
  void InitializeProcessor();
  void SetMaxData(const GPUTrackingInOutPointers&);

  // Neural network clusterization

  int32_t mNnClusterizerSizeInputRow = 3;
  int32_t mNnClusterizerSizeInputPad = 3;
  int32_t mNnClusterizerSizeInputTime = 3;
  int32_t mNnClusterizerChargeArraySize = -1;
  int32_t mNnClusterizerElementSize = -1;
  int8_t mNnClusterizerAddIndexData = 1;
  int8_t mNnClusterizerUseClassification = 1;
  float mNnClassThreshold = 0.01;
  int8_t mNnSigmoidTrafoClassThreshold = 1;
  int8_t mNnClusterizerSetDeconvolutionFlags = 1;
  int32_t mNnClusterizerUseCfRegression = 0;
  int32_t mNnClusterizerBatchedMode = 1;
  int32_t mNnClusterizerTotalClusters = 1;
  int32_t mNnClusterizerVerbosity = 1;
  int32_t mNnClusterizerBoundaryFillValue = -1;
  int32_t mNnClusterizerModelClassNumOutputNodes = -1;
  int32_t mNnClusterizerModelReg1NumOutputNodes = -1;
  int32_t mNnClusterizerModelReg2NumOutputNodes = -1;
  int32_t mNnInferenceInputDType = 0;  // 0: float16, 1: float32
  int32_t mNnInferenceOutputDType = 0; // 0: float16, 1: float32
  int32_t mISector = -1;
  int32_t mDeviceId = -1;

  // charge array boundaries
  int32_t maxFragmentLen = -1;
  int32_t maxAllowedTimebin = -1; // == tpcMaxTimeBin

  // GPU optimizations
  uint32_t mNnClusterizerFullRowSize = 0;
  uint32_t mNnClusterizerFullPadSize = 0;
  uint32_t mNnClusterizerFullTimeSize = 0;
  uint32_t mNnClusterizerPadTimeSize = 0;
  uint32_t mNnClusterizerRowTimeSize = 0;
  uint32_t mNnClusterizerRowTimeSizeFull = 0;

  // Boundary lookup table
  // int32_t mBoundaryMapSizeRow = 0;
  // int32_t mBoundaryMapSizePadsPerRow = 0;
  // int32_t mBoundaryMapSize = 0;
  // int32_t mBoundaryPadding = 11; // Padding on each side of the boundary map to account for pad_offset
  // int8_t* mIsBoundary = nullptr;

  // Index lookup table
  // int32_t mIndexLookupSize = 0;
  // int32_t* mIndexLookup = nullptr;

  // Memory allocation for neural network

  int8_t* mClusterFlags = nullptr; // mSplitInTime, mSplitInPad. Techincally both flags are set in the same way -> ClusterAccumulator.cx=nullptr
  int32_t* mOutputDataClass = nullptr;

  // FP32
  float* mInputData_32 = nullptr;
  float* mModelProbabilities_32 = nullptr;
  float* mOutputDataReg1_32 = nullptr;
  float* mOutputDataReg2_32 = nullptr;

  // FP16
  OrtDataType::Float16_t* mInputData_16 = nullptr;
  OrtDataType::Float16_t* mModelProbabilities_16 = nullptr;
  OrtDataType::Float16_t* mOutputDataReg1_16 = nullptr;
  OrtDataType::Float16_t* mOutputDataReg2_16 = nullptr;

  int16_t mMemoryId = -1;
}; // class GPUTPCNNClusterizer

} // namespace o2::gpu

#endif
