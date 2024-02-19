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
class ConfigParamSpec;
class ConfigParamRegistry;
class InitContext;
} // namespace framework

namespace tpc
{

struct CorrectionMapsLoaderGloOpts {
  int lumiType = 0; ///< what estimator to used for corrections scaling: 0: no scaling, 1: CTP, 2: IDC
  int lumiMode = 0; ///< what corrections method to use: 0: classical scaling, 1: Using of the derivative map, 2: Using of the derivative map for MC
  bool enableMShapeCorrection = false;
  bool requestCTPLumi = true; //< request CTP Lumi regardless of what is used for corrections scaling

  bool needTPCScalersWorkflow() const
  {
    return lumiType == 2 || enableMShapeCorrection;
  }
};

class CorrectionMapsLoader : public o2::gpu::CorrectionMapsHelper
{
 public:
  CorrectionMapsLoader() = default;
  ~CorrectionMapsLoader() = default;
  CorrectionMapsLoader(const CorrectionMapsLoader&) = delete;

#ifndef GPUCA_GPUCODE_DEVICE
  bool accountCCDBInputs(const o2::framework::ConcreteDataMatcher& matcher, void* obj);
  void extractCCDBInputs(o2::framework::ProcessingContext& pc);
  void updateVDrift(float vdriftCorr, float vdrifRef, float driftTimeOffset = 0);
  void init(o2::framework::InitContext& ic);
  void copySettings(const CorrectionMapsLoader& src);
  void updateInverse(); /// recalculate inverse correction

  static void requestCCDBInputs(std::vector<o2::framework::InputSpec>& inputs, std::vector<o2::framework::ConfigParamSpec>& options, const CorrectionMapsLoaderGloOpts& gloOpts);
  static void addGlobalOptions(std::vector<o2::framework::ConfigParamSpec>& options);
  static void addOptions(std::vector<o2::framework::ConfigParamSpec>& options);
  static CorrectionMapsLoaderGloOpts parseGlobalOptions(const o2::framework::ConfigParamRegistry& opts);

 protected:
  static void addOption(std::vector<o2::framework::ConfigParamSpec>& options, o2::framework::ConfigParamSpec&& osp);
  static void addInput(std::vector<o2::framework::InputSpec>& inputs, o2::framework::InputSpec&& isp);

  float mInstLumiCTPFactor = 1.0; // multiplicative factor for inst. lumi
  int mLumiCTPSource = 0;         // 0: main, 1: alternative CTP lumi source
  std::unique_ptr<GPUCA_NAMESPACE::gpu::TPCFastTransform> mCorrMapMShape{nullptr};
#endif
};

} // namespace tpc

} // namespace o2

#endif
