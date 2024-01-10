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
#include "TPCCalibration/CorrMapParam.h"
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
  pc.inputs().get<o2::tpc::CorrMapParam*>("tpcCorrPar");
  pc.inputs().get<o2::gpu::TPCFastTransform*>("tpcCorrMap");
  pc.inputs().get<o2::gpu::TPCFastTransform*>("tpcCorrMapRef");
  const int maxDumRep = 5;
  int dumRep = 0;
  o2::ctp::LumiInfo lumiObj;
  static o2::ctp::LumiInfo lumiPrev;
  if (getLumiScaleType() == 1 && mInstLumiOverride <= 0.) {
    if (pc.inputs().get<gsl::span<char>>("CTPLumi").size() == sizeof(o2::ctp::LumiInfo)) {
      lumiPrev = lumiObj = pc.inputs().get<o2::ctp::LumiInfo>("CTPLumi");
    } else {
      if (dumRep < maxDumRep && lumiPrev.nHBFCounted == 0 && lumiPrev.nHBFCountedFV0 == 0) {
        LOGP(alarm, "Previous TF lumi used to substitute dummy input is empty, warning {} of {}", ++dumRep, maxDumRep);
      }
      lumiObj = lumiPrev;
    }
    setInstLumi(mInstLumiFactor * (mCTPLumiSource == 0 ? lumiObj.getLumi() : lumiObj.getLumiAlt()));
  } else if (getLumiScaleType() == 2 && mInstLumiOverride <= 0.) {
    float tpcScaler = pc.inputs().get<float>("tpcscaler");
    setInstLumi(mInstLumiFactor * tpcScaler);
  }
}

//________________________________________________________
void CorrectionMapsLoader::requestCCDBInputs(std::vector<InputSpec>& inputs, std::vector<o2::framework::ConfigParamSpec>& options, const CorrectionMapsLoaderGloOpts& gloOpts)
{
  if (gloOpts.lumiMode == 0) {
    addInput(inputs, {"tpcCorrMap", "TPC", "CorrMap", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMap), {}, 1)});          // time-dependent
    addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMapRef), {}, 0)}); // load once
  } else if (gloOpts.lumiMode == 1) {
    addInput(inputs, {"tpcCorrMap", "TPC", "CorrMap", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMap), {}, 1)});            // time-dependent
    addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrDerivMap), {}, 1)}); // time-dependent
  } else if (gloOpts.lumiMode == 2) {
    // for MC corrections
    addInput(inputs, {"tpcCorrMap", "TPC", "CorrMap", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMapMC), {}, 1)});            // time-dependent
    addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrDerivMapMC), {}, 1)}); // time-dependent
  } else {
    LOG(fatal) << "Correction mode unknown! Choose either 0 (default) or 1 (derivative map) for flag corrmap-lumi-mode.";
  }

  if (gloOpts.lumiType == 1) {
    addInput(inputs, {"CTPLumi", "CTP", "LUMI", 0, Lifetime::Timeframe});
  } else if (gloOpts.lumiType == 2) {
    addInput(inputs, {"tpcscaler", o2::header::gDataOriginTPC, "TPCSCALER", 0, Lifetime::Timeframe});
  }

  addInput(inputs, {"tpcCorrPar", "TPC", "CorrMapParam", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CorrMapParam), {}, 0)}); // load once

  addOptions(options);
}

//________________________________________________________
void CorrectionMapsLoader::addOptions(std::vector<ConfigParamSpec>& options)
{
  // these are options which should be added at the level of device using TPC corrections
  // At the moment - nothing, all options are moved to configurable param CorrMapParam
}

//________________________________________________________
void CorrectionMapsLoader::addGlobalOptions(std::vector<ConfigParamSpec>& options)
{
  // these are options which should be added at the workflow level, since they modify the inputs of the devices
  addOption(options, ConfigParamSpec{"lumi-type", o2::framework::VariantType::Int, 0, {"1 = require CTP lumi for TPC correction scaling, 2 = require TPC scalers for TPC correction scaling"}});
  addOption(options, ConfigParamSpec{"corrmap-lumi-mode", o2::framework::VariantType::Int, 0, {"scaling mode: (default) 0 = static + scale * full; 1 = full + scale * derivative; 2 = full + scale * derivative (for MC)"}});
}

//________________________________________________________
CorrectionMapsLoaderGloOpts CorrectionMapsLoader::parseGlobalOptions(const o2::framework::ConfigParamRegistry& opts)
{
  CorrectionMapsLoaderGloOpts tpcopt;
  tpcopt.lumiType = opts.get<int>("lumi-type");
  tpcopt.lumiMode = opts.get<int>("corrmap-lumi-mode");
  return tpcopt;
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
    if (getMeanLumiRefOverride() == 0) {
      setMeanLumiRef(mCorrMapRef->getLumi());
    }
    LOGP(debug, "MeanLumiRefOverride={} MeanLumiMap={} -> meanLumi = {}", getMeanLumiRefOverride(), mCorrMapRef->getLumi(), getMeanLumiRef());
    setUpdatedMapRef();
    return true;
  }
  if (matcher == ConcreteDataMatcher("TPC", "CorrMapParam", 0)) {
    const auto& par = o2::tpc::CorrMapParam::Instance();
    mMeanLumiOverride = par.lumiMean;
    mMeanLumiRefOverride = par.lumiMeanRef;
    mInstLumiOverride = par.lumiInst;
    mInstLumiFactor = par.lumiInstFactor;
    mCTPLumiSource = par.ctpLumiSource;

    if (mMeanLumiOverride != 0.) {
      setMeanLumi(mMeanLumiOverride, false);
    }
    if (mMeanLumiRefOverride != 0.) {
      setMeanLumiRef(mMeanLumiRefOverride);
    }
    if (mInstLumiOverride != 0.) {
      setInstLumi(mInstLumiOverride, false);
    }
    setUpdatedLumi();
    int scaleType = getLumiScaleType();
    const std::array<std::string, 3> lumiS{"OFF", "CTP", "TPC scaler"};
    if (scaleType >= lumiS.size()) {
      LOGP(fatal, "Wrong lumi-scale-type provided!");
    }
    LOGP(info, "TPC correction map params updated (corr.map scaling type={}): override values: lumiMean={} lumiRefMean={} lumiInst={} lumiScaleMode={}, LumiInst scale={}, CTP Lumi source={}",
         lumiS[scaleType], mMeanLumiOverride, mMeanLumiRefOverride, mInstLumiOverride, mLumiScaleMode, mInstLumiFactor, mCTPLumiSource);
  }
  return false;
}

//________________________________________________________
void CorrectionMapsLoader::init(o2::framework::InitContext& ic)
{
  if (getLumiScaleMode() < 0) {
    LOGP(fatal, "TPC correction lumi scaling mode is not set");
  }
  const auto& inputRouts = ic.services().get<const o2::framework::DeviceSpec>().inputs;
  for (const auto& route : inputRouts) {
    if (route.matcher == InputSpec{"CTPLumi", "CTP", "LUMI", 0, Lifetime::Timeframe}) {
      if (getLumiScaleType() != 1) {
        LOGP(fatal, "Lumi scaling source CTP is not compatible with TPC correction lumi scaler type {}", getLumiScaleType());
      }
      break;
    } else if (route.matcher == InputSpec{"tpcscaler", o2::header::gDataOriginTPC, "TPCSCALER", 0, Lifetime::Timeframe}) {
      if (getLumiScaleType() != 2) {
        LOGP(fatal, "Lumi scaling source TPCScaler is not compatible with TPC correction lumi scaler type {}", getLumiScaleType());
      }
      break;
    }
  }
}

//________________________________________________________
void CorrectionMapsLoader::copySettings(const CorrectionMapsLoader& src)
{
  setInstLumi(src.getInstLumi(), false);
  setMeanLumi(src.getMeanLumi(), false);
  setMeanLumiRef(src.getMeanLumiRef());
  setLumiScaleType(src.getLumiScaleType());
  setMeanLumiOverride(src.getMeanLumiOverride());
  setMeanLumiRefOverride(src.getMeanLumiRefOverride());
  setInstLumiOverride(src.getInstLumiOverride());
  setLumiScaleMode(src.getLumiScaleMode());
  mInstLumiFactor = src.mInstLumiFactor;
  mCTPLumiSource = src.mCTPLumiSource;
}

#endif // #ifndef GPUCA_GPUCODE_DEVICE
