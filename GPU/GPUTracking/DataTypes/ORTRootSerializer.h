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

/// \file   ORTRootSerializer.h
/// \brief  Class to serialize ONNX objects for ROOT snapshots of CCDB objects at runtime
/// \author Christian Sonnabend <christian.sonnabend@cern.ch>

#ifndef ALICEO2_TPC_ORTROOTSERIALIZER_H_
#define ALICEO2_TPC_ORTROOTSERIALIZER_H_

#include "GPUCommonRtypes.h"
#include <vector>
#include <string>

namespace o2::tpc
{

class ORTRootSerializer
{
 public:
  ORTRootSerializer() = default;
  ~ORTRootSerializer() = default;

  void setOnnxModel(const char* onnxModel, uint32_t size);
  const char* getONNXModel() const { return mModelBuffer.data(); }
  uint32_t getONNXModelSize() const { return static_cast<uint32_t>(mModelBuffer.size()); }

 private:
  std::vector<char> mModelBuffer; ///< buffer for serialization
  ClassDefNV(ORTRootSerializer, 1);
};

} // namespace o2::tpc

#endif // ALICEO2_TPC_ORTROOTSERIALIZER_H_
