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

#include "TPCCalibration/CorrectionMapsLoader.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "TPCBase/CDBInterface.h"
#include "Framework/Logger.h"
#include "Framework/ProcessingContext.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/InputRecord.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/InitContext.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "DataFormatsCTP/LumiInfo.h"

using namespace o2::tpc;
using namespace o2::framework;

#ifndef GPUCA_GPUCODE_DEVICE

//________________________________________________________
void CorrectionMapsLoader::updateVDrift(float vdriftCorr, float vdrifRef, float driftTimeOffset)
{
  o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mCorrMap, 0, vdriftCorr, vdrifRef, driftTimeOffset);
  if (mCorrMapRef) {
    o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mCorrMapRef, 0, vdriftCorr, vdrifRef, driftTimeOffset);
  }
}

//________________________________________________________
void CorrectionMapsLoader::extractCCDBInputs(ProcessingContext& pc)
{
  pc.inputs().get<o2::gpu::TPCFastTransform*>("tpcCorrMap");
  pc.inputs().get<o2::gpu::TPCFastTransform*>("tpcCorrMapRef"); // not used at the moment
  if (getUseCTPLumi() && mInstLumiOverride <= 0.) {
    auto lumiObj = pc.inputs().get<o2::ctp::LumiInfo>("CTPLumi");
    setInstLumi(lumiObj.getLumi());
  }
}

//________________________________________________________
void CorrectionMapsLoader::requestCCDBInputs(std::vector<InputSpec>& inputs, std::vector<o2::framework::ConfigParamSpec>& options, bool requestCTPLumi)
{
  addInput(inputs, {"tpcCorrMap", "TPC", "CorrMap", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMap), {}, 1)});          // time-dependent
  addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMapRef), {}, 0)}); // load once
  if (requestCTPLumi) {
    addInput(inputs, {"lumi", "CTP", "LUMI", 0, Lifetime::Timeframe});
  }
  addOptions(options);
}

//________________________________________________________
void CorrectionMapsLoader::addOptions(std::vector<ConfigParamSpec>& options)
{
  addOption(options, ConfigParamSpec{"corrmap-lumi-mean", VariantType::Float, 0.f, {"override TPC corr.map mean lumi (if > 0), disable corrections if < 0"}});
  addOption(options, ConfigParamSpec{"corrmap-lumi-inst", VariantType::Float, 0.f, {"override instantaneous CTP lumi (if > 0) for TPC corr.map scaling, disable corrections if < 0"}});
}

//________________________________________________________
void CorrectionMapsLoader::addInput(std::vector<InputSpec>& inputs, InputSpec&& isp)
{
  if (std::find(inputs.begin(), inputs.end(), isp) == inputs.end()) {
    inputs.emplace_back(isp);
  }
}

//________________________________________________________
void CorrectionMapsLoader::addOption(std::vector<ConfigParamSpec>& options, ConfigParamSpec&& osp)
{
  if (std::find(options.begin(), options.end(), osp) == options.end()) {
    options.emplace_back(osp);
  }
}

//________________________________________________________
bool CorrectionMapsLoader::accountCCDBInputs(const ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("TPC", "CorrMap", 0)) {
    setCorrMap((o2::gpu::TPCFastTransform*)obj);
    mCorrMap->rectifyAfterReadingFromFile();
    if (getMeanLumiOverride() <= 0 && mCorrMap->getLumi() > 0.) {
      setMeanLumi(mCorrMap->getLumi());
    }
    LOGP(debug, "MeanLumiOverride={} MeanLumiMap={} -> meanLumi = {}", getMeanLumiOverride(), mCorrMap->getLumi(), getMeanLumi());
    setUpdatedMap();
    return true;
  }
  if (matcher == ConcreteDataMatcher("TPC", "CorrMapRef", 0)) {
    setCorrMapRef((o2::gpu::TPCFastTransform*)obj);
    mCorrMapRef->rectifyAfterReadingFromFile();
    setUpdatedMapRef();
    return true;
  }
  return false;
}

//________________________________________________________
void CorrectionMapsLoader::init(o2::framework::InitContext& ic)
{
  const auto& inputRouts = ic.services().get<const o2::framework::DeviceSpec>().inputs;
  for (const auto& route : inputRouts) {
    if (route.matcher == InputSpec{"CTPLumi", "CTP", "LUMI", 0, Lifetime::Timeframe}) {
      setUseCTPLumi(true);
      break;
    }
  }
  mMeanLumiOverride = ic.options().get<float>("corrmap-lumi-mean");
  mInstLumiOverride = ic.options().get<float>("corrmap-lumi-inst");
  if (mMeanLumiOverride != 0.) {
    setMeanLumi(mMeanLumiOverride);
  }
  if (mInstLumiOverride != 0.) {
    setInstLumi(mInstLumiOverride);
  }
  LOGP(info, "CTP Lumi request for TPC corr.map scaling={}, override values: lumiMean={} lumiInst={}", getUseCTPLumi() ? "ON" : "OFF", mMeanLumiOverride, mInstLumiOverride);
}
#endif // #ifndef GPUCA_GPUCODE_DEVICE
