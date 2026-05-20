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
#include "TPCBaseRecSim/CDBTypes.h"
#include "Framework/Logger.h"
#include "Framework/ProcessingContext.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/InitContext.h"
#include "Framework/DeviceSpec.h"
#include "DataFormatsCTP/LumiInfo.h"

using namespace o2::tpc;
using namespace o2::framework;

//________________________________________________________
void CorrectionMapsLoader::extractCCDBInputs(ProcessingContext& pc, float tpcScaler)
{
  pc.inputs().get<o2::tpc::CorrMapParam*>("tpcCorrPar");
  const auto lumiMode = getLumiScaleMode();
  if (lumiMode != LumiScaleMode::NoCorrection && lumiMode != LumiScaleMode::StaticMapOnly) {
    pc.inputs().get<o2::gpu::TPCFastTransform*>("tpcCorrMap");
  }
  if (lumiMode != LumiScaleMode::NoCorrection) {
    pc.inputs().get<o2::gpu::TPCFastTransform*>("tpcCorrMapRef");
  }
  const int maxDumRep = 5;
  int dumRep = 0;
  o2::ctp::LumiInfo lumiObj;
  static o2::ctp::LumiInfo lumiPrev;

  if (getLumiScaleType() == LumiScaleType::TPCScaler || mIDC2CTPFallbackActive) {
    // check if tpcScaler is valid and CTP fallback is allowed
    if (tpcScaler == -1.f) {
      const bool canUseCTPScaling = mCorrMap && mCorrMapRef && mCorrMap->isIDCSet() && mCorrMapRef->isIDCSet() && mCorrMap->isLumiSet() && mCorrMapRef->isLumiSet();
      if (canUseCTPScaling) {
        LOGP(info, "Invalid TPC scaler value {} received for IDC-based scaling! Using CTP fallback", tpcScaler);
        mIDC2CTPFallbackActive = true;
        setMeanLumi(mCorrMap->getLumi(), false);
        setMeanLumiRef(mCorrMapRef->getLumi());
        setLumiScaleType(LumiScaleType::CTPLumi);
      } else if (mCorrMap) {
        // CTP scaling is not possible, dont do any scaling to avoid applying wrong corrections
        const float storedIDC = mCorrMap->getIDC();
        LOGP(warning, "Invalid TPC scaler value {} received for IDC-based scaling! CTP fallback not possible, using stored IDC of {} from the map to avoid applying wrong corrections", tpcScaler, storedIDC);
        setInstLumi(storedIDC);
      }
    } else {
      if (mIDC2CTPFallbackActive) {
        // reset back to normal operation
        LOGP(info, "Valid TPC scaler value {} received, switching back to IDC-based scaling", tpcScaler);
        mIDC2CTPFallbackActive = false;
        setMeanLumi(mCorrMap->getIDC(), false);
        setMeanLumiRef(mCorrMapRef->getIDC());
        setLumiScaleType(LumiScaleType::TPCScaler);
      }
      // correct IDC received
      setInstLumi(tpcScaler);
    }
  }

  if (getLumiCTPAvailable() && mInstCTPLumiOverride <= 0.) {
    if (pc.inputs().get<gsl::span<char>>("CTPLumi").size() == sizeof(o2::ctp::LumiInfo)) {
      lumiPrev = lumiObj = pc.inputs().get<o2::ctp::LumiInfo>("CTPLumi");
    } else {
      if (dumRep < maxDumRep && lumiPrev.nHBFCounted == 0 && lumiPrev.nHBFCountedFV0 == 0) {
        LOGP(alarm, "Previous TF lumi used to substitute dummy input is empty, warning {} of {}", ++dumRep, maxDumRep);
      }
      lumiObj = lumiPrev;
    }
    setInstLumiCTP(mInstLumiCTPFactor * (mLumiCTPSource == 0 ? lumiObj.getLumi() : lumiObj.getLumiAlt()));
    if (getLumiScaleType() == LumiScaleType::CTPLumi) {
      setInstLumi(getInstLumiCTP());
    }
  }

  reportScaling();
}

//________________________________________________________
void CorrectionMapsLoader::requestCCDBInputs(std::vector<InputSpec>& inputs, const CorrectionMapsGloOpts& gloOpts)
{
  LOGP(info, "Requesting CCDB inputs for TPC correction maps with lumiType={} and lumiMode={}", static_cast<int>(gloOpts.lumiType), static_cast<int>(gloOpts.lumiMode));
  if (gloOpts.lumiMode == LumiScaleMode::Linear) {
    addInput(inputs, {"tpcCorrMap", "TPC", "CorrMap", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMap), {}, 1)});          // time-dependent
    addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMapRef), {}, 0)}); // load once
  } else if (gloOpts.lumiMode == LumiScaleMode::DerivativeMap) {
    addInput(inputs, {"tpcCorrMap", "TPC", "CorrMap", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMap), {}, 1)});            // time-dependent
    addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrDerivMap), {}, 1)}); // time-dependent
  } else if (gloOpts.lumiMode == LumiScaleMode::DerivativeMapMC) {
    // for MC corrections
    addInput(inputs, {"tpcCorrMap", "TPC", "CorrMap", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMapMC), {}, 1)});            // time-dependent
    addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrDerivMapMC), {}, 1)}); // time-dependent
  } else if (gloOpts.lumiMode == LumiScaleMode::NoCorrection) {
    // no correction maps needed — a dummy map is created at runtime
  } else if (gloOpts.lumiMode == LumiScaleMode::StaticMapOnly) {
    addInput(inputs, {"tpcCorrMapRef", "TPC", "CorrMapRef", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CalCorrMapRef), {}, 0)}); // load once
  } else {
    LOG(fatal) << "Correction mode unknown! Choose either 0 (default) or 1 (derivative map) for flag corrmap-lumi-mode.";
  }

  if (gloOpts.requestCTPLumi) {
    addInput(inputs, {"CTPLumi", "CTP", "LUMI", 0, Lifetime::Timeframe});
  }

  addInput(inputs, {"tpcCorrPar", "TPC", "CorrMapParam", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CorrMapParam), {}, 0)}); // load once
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
    mCorrMap->setCTP2IDCFallBackThreshold(o2::tpc::CorrMapParam::Instance().CTP2IDCFallBackThreshold);
    if (getMeanLumiOverride() != 0) {
      if (getLumiScaleType() == LumiScaleType::CTPLumi) {
        mCorrMap->setLumi(getMeanLumiOverride());
        LOGP(info, "CorrMap mean lumi rate is overridden to {}", mCorrMap->getLumi());
      } else if (getLumiScaleType() == LumiScaleType::TPCScaler) {
        mCorrMap->setIDC(getMeanLumiOverride());
        LOGP(info, "CorrMap mean IDC rate is overridden to {}", mCorrMap->getIDC());
      }
    }
    float mapMeanRate = 0;
    if (getLumiScaleType() == LumiScaleType::CTPLumi) {
      mapMeanRate = mCorrMap->getLumi();
    } else if (getLumiScaleType() == LumiScaleType::TPCScaler) {
      mapMeanRate = mCorrMap->getIDC();
    }
    if (mCheckCTPIDCConsistency) {
      checkMeanScaleConsistency(mapMeanRate, mCorrMap->getCTP2IDCFallBackThreshold());
    }
    if (getMeanLumiOverride() == 0 && mapMeanRate > 0.) {
      setMeanLumi(mapMeanRate, false);
    }
    LOGP(debug, "MeanLumiOverride={} MeanLumiMap={} -> meanLumi = {}", getMeanLumiOverride(), mapMeanRate, getMeanLumi());
    setUpdatedMap();
    return true;
  }
  if (matcher == ConcreteDataMatcher("TPC", "CorrMapRef", 0)) {
    setCorrMapRef((o2::gpu::TPCFastTransform*)obj);
    mCorrMapRef->rectifyAfterReadingFromFile();
    mCorrMapRef->setCTP2IDCFallBackThreshold(o2::tpc::CorrMapParam::Instance().CTP2IDCFallBackThreshold);
    if (getMeanLumiRefOverride() != 0) {
      if (getLumiScaleType() == LumiScaleType::CTPLumi) {
        mCorrMapRef->setLumi(getMeanLumiRefOverride());
        LOGP(info, "CorrMapRef mean lumi rate is overridden to {}", mCorrMapRef->getLumi());
      } else if (getLumiScaleType() == LumiScaleType::TPCScaler) {
        mCorrMapRef->setIDC(getMeanLumiRefOverride());
        LOGP(info, "CorrMapRef mean IDC rate is overridden to {}", mCorrMapRef->getIDC());
      }
    }
    float mapRefMeanRate = 0;
    if (getLumiScaleType() == LumiScaleType::CTPLumi) {
      mapRefMeanRate = mCorrMapRef->getLumi();
    } else if (getLumiScaleType() == LumiScaleType::TPCScaler) {
      mapRefMeanRate = mCorrMapRef->getIDC();
    }
    if (mCheckCTPIDCConsistency) {
      checkMeanScaleConsistency(mapRefMeanRate, mCorrMapRef->getCTP2IDCFallBackThreshold());
    }
    if (getMeanLumiRefOverride() == 0) {
      setMeanLumiRef(mapRefMeanRate);
    }
    LOGP(debug, "MeanLumiRefOverride={} MeanLumiMap={} -> meanLumi = {}", getMeanLumiRefOverride(), mapRefMeanRate, getMeanLumiRef());
    setUpdatedMapRef();
    return true;
  }
  if (matcher == ConcreteDataMatcher("TPC", "CorrMapParam", 0)) {
    const auto& par = o2::tpc::CorrMapParam::Instance();
    mMeanLumiOverride = par.lumiMean; // negative value switches off corrections !!!
    mMeanLumiRefOverride = par.lumiMeanRef;
    mInstCTPLumiOverride = par.lumiInst;
    mInstLumiCTPFactor = par.lumiInstFactor;
    mLumiCTPSource = par.ctpLumiSource;

    if (mMeanLumiOverride != 0.) {
      setMeanLumi(mMeanLumiOverride, false);
    }
    if (mMeanLumiRefOverride != 0.) {
      setMeanLumiRef(mMeanLumiRefOverride);
    }
    if (mInstCTPLumiOverride != 0.) {
      setInstLumiCTP(mInstCTPLumiOverride * mInstLumiCTPFactor);
      if (getLumiScaleType() == LumiScaleType::CTPLumi) {
        setInstLumi(getInstLumiCTP(), false);
      }
    }
    setUpdatedLumi();
    int scaleType = static_cast<int>(getLumiScaleType());
    const std::array<std::string, 3> lumiS{"OFF", "CTP", "TPC scaler"};
    if (scaleType >= lumiS.size()) {
      LOGP(fatal, "Wrong corrmap-lumi-mode provided!");
    }

    LOGP(info, "TPC correction map params updated: SP corrections: {} (corr.map scaling type={}, override values: lumiMean={} lumiRefMean={} lumiScaleMode={}), CTP Lumi: source={} lumiInstOverride={} , LumiInst scale={} ",
         canUseCorrections() ? "ON" : "OFF",
         lumiS[scaleType], mMeanLumiOverride, mMeanLumiRefOverride, static_cast<int>(getLumiScaleMode()), mLumiCTPSource, mInstCTPLumiOverride, mInstLumiCTPFactor);
  }
  return false;
}

//________________________________________________________
void CorrectionMapsLoader::init(o2::framework::InitContext& ic, bool idcsAvailable)
{
  if (getLumiScaleMode() == LumiScaleMode::Unset) {
    LOGP(fatal, "TPC correction lumi scaling mode is not set");
  }
  const auto& inputRouts = ic.services().get<const o2::framework::DeviceSpec>().inputs;
  bool foundCTP = false;
  for (const auto& route : inputRouts) {
    if (route.matcher == InputSpec{"CTPLumi", "CTP", "LUMI", 0, Lifetime::Timeframe}) {
      foundCTP = true;
    }
  }
  setLumiCTPAvailable(foundCTP);
  if ((getLumiScaleType() == LumiScaleType::CTPLumi && !foundCTP) || (getLumiScaleType() == LumiScaleType::TPCScaler && !idcsAvailable)) {
    LOGP(fatal, "Lumi scaling source {}({}) is not available for TPC correction", static_cast<int>(getLumiScaleType()), getLumiScaleType() == LumiScaleType::CTPLumi ? "CTP" : "TPCScaler");
  }
}

void CorrectionMapsLoader::checkMeanScaleConsistency(float meanLumi, float threshold) const
{
  if (getLumiScaleType() == LumiScaleType::CTPLumi) {
    if (meanLumi < threshold) {
      LOGP(fatal, "CTP Lumi scaling source is requested, but the map mean scale {} is below the threshold {}", meanLumi, threshold);
    }
  } else if (getLumiScaleType() == LumiScaleType::TPCScaler) {
    if (meanLumi > threshold) {
      LOGP(fatal, "IDC scaling source is requested, but the map mean scale {} is above the threshold {}", meanLumi, threshold);
    }
  }
}
