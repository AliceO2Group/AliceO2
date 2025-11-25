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

/// \file   ORTRootSerializer.cxx
/// \author Christian Sonnabend <christian.sonnabend@cern.ch>

#include "ORTRootSerializer.h"
#include <cstring>

using namespace o2::tpc;

/// Initialize the serialization from a char* buffer containing the model
void ORTRootSerializer::setOnnxModel(const char* onnxModel, uint32_t size)
{
  mModelBuffer.resize(size);
  std::memcpy(mModelBuffer.data(), onnxModel, size);
}
