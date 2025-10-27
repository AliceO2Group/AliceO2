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

/// \file CorrectionMapsLoaderFull.h
/// \brief Helper class to access load maps from CCDB
/// \author matthias.kleiner@cern.ch

#ifndef TPC_CORRECTION_MAPS_LOADERFULL_H_
#define TPC_CORRECTION_MAPS_LOADERFULL_H_

#include <vector>
#include "CorrectionMapsHelperFull.h"
#include "CorrectionMapsHelper.h"

namespace o2
{
namespace framework
{
class ProcessingContext;
class ConcreteDataMatcher;
class InputSpec;
class ConfigParamSpec;
class InitContext;
} // namespace framework

namespace tpc
{

class CorrectionMapsLoaderFull : public o2::gpu::CorrectionMapsHelperFull
{
 public:
  CorrectionMapsLoaderFull() = default;
  ~CorrectionMapsLoaderFull() = default;
  CorrectionMapsLoaderFull(const CorrectionMapsLoaderFull&) = delete;

  bool accountCCDBInputs(const o2::framework::ConcreteDataMatcher& matcher, void* obj);
  void extractCCDBInputs(o2::framework::ProcessingContext& pc, float tpcScaler = -1.f);
  void init(o2::framework::InitContext& ic, bool idcsAvailable);
  void checkMeanScaleConsistency(float meanLumi, float threshold) const;

  static void requestCCDBInputs(std::vector<o2::framework::InputSpec>& inputs, const o2::tpc::CorrectionMapsLoaderGloOpts& gloOpts);

 protected:
  static void addOption(std::vector<o2::framework::ConfigParamSpec>& options, o2::framework::ConfigParamSpec&& osp);
  static void addInput(std::vector<o2::framework::InputSpec>& inputs, o2::framework::InputSpec&& isp);

  float mInstLumiCTPFactor = 1.0; // multiplicative factor for inst. lumi
  int mLumiCTPSource = 0;         // 0: main, 1: alternative CTP lumi source
  bool mIDC2CTPFallbackActive = false; // flag indicating that fallback from IDC to CTP scaling is active
};

} // namespace tpc

} // namespace o2

#endif
