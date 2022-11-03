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

/// \file CorrectionMapsLoader.h
/// \brief Helper class to access load maps from CCDB
/// \author ruben.shahoian@cern.ch

#ifndef TPC_CORRECTION_MAPS_LOADER_H_
#define TPC_CORRECTION_MAPS_LOADER_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <memory>
#include <vector>
#endif
#include "CorrectionMapsHelper.h"

namespace o2
{
namespace framework
{
class ProcessingContext;
class ConcreteDataMatcher;
class InputSpec;
} // namespace framework

namespace tpc
{

class CorrectionMapsLoader : public o2::gpu::CorrectionMapsHelper
{
 public:
  CorrectionMapsLoader() = default;
  ~CorrectionMapsLoader() = default;
  CorrectionMapsLoader(const CorrectionMapsLoader&) = delete;

#ifndef GPUCA_GPUCODE_DEVICE
  bool accountCCDBInputs(const o2::framework::ConcreteDataMatcher& matcher, void* obj);
  static void requestCCDBInputs(std::vector<o2::framework::InputSpec>& inputs);
  static void extractCCDBInputs(o2::framework::ProcessingContext& pc);
  static void addInput(std::vector<o2::framework::InputSpec>& inputs, o2::framework::InputSpec&& isp);
  void updateVDrift(float vdriftCorr, float vdrifRef);
#endif
};

} // namespace tpc

} // namespace o2

#endif
