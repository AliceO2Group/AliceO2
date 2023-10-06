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
  const int maxDumRep = 5;
  int dumRep = 0;
  o2::ctp::LumiInfo lumiObj;
  static o2::ctp::LumiInfo lumiPrev;
  if (getUseCTPLumi() && mInstLumiOverride <= 0.) {
    if (pc.inputs().get<gsl::span<char>>("CTPLumi").size() == sizeof(o2::ctp::LumiInfo)) {
      lumiPrev = lumiObj = pc.inputs().get<o2::ctp::LumiInfo>("CTPLumi");
    } else {
      if (dumRep < maxDumRep && lumiPrev.nHBFCounted == 0 && lumiPrev.nHBFCountedFV0 == 0) {
        LOGP(alarm, "Previous TF lumi used to substitute dummy input is empty, warning {} of {}", ++dumRep, maxDumRep);
      }
      lumiObj = lumiPrev;
    }
    setInstLumi(mInstLumiFactor * (mCTPLumiSource == 0 ? lumiObj.getLumi() : lumiObj.getLumiAlt()));
  }
}

//________________________________________________________
void CorrectionMapsLoader::requestCCDBInputs(std::vector<InputSpec>& inputs, std::vector<o2::framework::ConfigParamSpec>& options, bool requestCTPLumi, int lumiScaleMode)
{
  addInput(inputs, {"tpcCorrMap", "TPC", "CorrMap", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMap), {}, 1)});          // time-dependent
  if (lumiScaleMode == 0) {
    addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMapRef), {}, 0)}); // load once
  } else if (lumiScaleMode == 1) {
    addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrDerivMap), {}, 1)}); // time-dependent
  } else {
    LOG(fatal) << "Correction mode unknown! Choose either 0 (default) or 1 (derivative map) for flag corrmap-lumi-mode.";
  }

  if (requestCTPLumi) {
    addInput(inputs, {"CTPLumi", "CTP", "LUMI", 0, Lifetime::Timeframe});
  }
  addOptions(options);
}

//________________________________________________________
void CorrectionMapsLoader::addOptions(std::vector<ConfigParamSpec>& options)
{
  addOption(options, ConfigParamSpec{"corrmap-lumi-mean", VariantType::Float, 0.f, {"override TPC corr.map mean lumi (if > 0), disable corrections if < 0"}});
  addOption(options, ConfigParamSpec{"corrmap-lumi-inst", VariantType::Float, 0.f, {"override instantaneous CTP lumi (if > 0) for TPC corr.map scaling, disable corrections if < 0"}});
  addOption(options, ConfigParamSpec{"corrmap-lumi-mode", VariantType::Int, 0, {"scaling mode: (default) 0 = static + scale * full; 1 = full + scale * derivative"}});
  addOption(options, ConfigParamSpec{"ctp-lumi-factor", VariantType::Float, 1.0f, {"scaling to apply to instantaneous lumi from CTP (but not corrmap-lumi-inst)"}});
  addOption(options, ConfigParamSpec{"ctp-lumi-source", VariantType::Int, 0, {"CTP lumi source: 0 = LumiInfo.getLumi(), 1 = LumiInfo.getLumiAlt()"}});
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
    if (getMeanLumiOverride() == 0 && mCorrMap->getLumi() > 0.) {
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
  mLumiScaleMode = ic.options().get<int>("corrmap-lumi-mode");
  mInstLumiFactor = ic.options().get<float>("ctp-lumi-factor");
  mCTPLumiSource = ic.options().get<int>("ctp-lumi-source");
  if (mMeanLumiOverride != 0.) {
    setMeanLumi(mMeanLumiOverride);
  }
  if (mInstLumiOverride != 0.) {
    setInstLumi(mInstLumiOverride);
  }
  LOGP(info, "CTP Lumi request for TPC corr.map scaling={}, override values: lumiMean={} lumiInst={} lumiScaleMode={}, LumiInst scale={}, CTP Lumi source={}",
       getUseCTPLumi() ? "ON" : "OFF", mMeanLumiOverride, mInstLumiOverride, mLumiScaleMode, mInstLumiFactor, mCTPLumiSource);
}

//________________________________________________________
void CorrectionMapsLoader::copySettings(const CorrectionMapsLoader& src)
{
  setInstLumi(src.getInstLumi(), false);
  setMeanLumi(src.getMeanLumi(), false);
  setUseCTPLumi(src.getUseCTPLumi());
  setMeanLumiOverride(src.getMeanLumiOverride());
  setInstLumiOverride(src.getInstLumiOverride());
  setLumiScaleMode(src.getLumiScaleMode());
  mInstLumiFactor = src.mInstLumiFactor;
  mCTPLumiSource = src.mCTPLumiSource;
}

#endif // #ifndef GPUCA_GPUCODE_DEVICE
