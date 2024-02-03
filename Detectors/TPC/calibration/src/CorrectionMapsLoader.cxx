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
#include "TPCCalibration/TPCFastSpaceChargeCorrectionHelper.h"

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
  if (mCorrMapMShape) {
    o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mCorrMapMShape, 0, vdriftCorr, vdrifRef, driftTimeOffset);
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
    if (getLumiScaleType() == 1) {
      setInstLumi(getInstLumiCTP());
    }
  }
  if (getLumiScaleType() == 2) {
    float tpcScaler = pc.inputs().get<float>("tpcscaler");
    setInstLumi(tpcScaler);
  }
  if (getUseMShapeCorrection()) {
    LOGP(info, "Setting M-Shape map");
    const auto mapMShape = pc.inputs().get<o2::gpu::TPCFastTransform*>("mshape");
    const_cast<o2::gpu::TPCFastTransform*>(mapMShape.get())->rectifyAfterReadingFromFile();
    mCorrMapMShape = std::unique_ptr<TPCFastTransform>(new TPCFastTransform);
    mCorrMapMShape->cloneFromObject(*(mapMShape.get()), nullptr);
    setCorrMapMShape(mCorrMapMShape.get());
    setUpdatedMapMShape();
  }

  // update inverse in case it is requested
  if (!mScaleInverse) {
    updateInverse();
  }
  reportScaling();
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

  if (gloOpts.requestCTPLumi) {
    addInput(inputs, {"CTPLumi", "CTP", "LUMI", 0, Lifetime::Timeframe});
  }

  if (gloOpts.lumiType == 2) {
    addInput(inputs, {"tpcscaler", o2::header::gDataOriginTPC, "TPCSCALER", 0, Lifetime::Timeframe});
  }

  addInput(inputs, {"tpcCorrPar", "TPC", "CorrMapParam", 0, Lifetime::Condition, ccdbParamSpec(CDBTypeMap.at(CDBType::CorrMapParam), {}, 0)}); // load once

  if (gloOpts.enableMShapeCorrection) {
    addInput(inputs, {"mshape", o2::header::gDataOriginTPC, "TPCMSHAPE", 0, Lifetime::Timeframe});
  }
  addOptions(options);
}

//________________________________________________________
void CorrectionMapsLoader::addOptions(std::vector<ConfigParamSpec>& options)
{
  // these are options which should be added at the level of device using TPC corrections
  // At the moment - nothing, all options are moved to configurable param CorrMapParam
  addOption(options, ConfigParamSpec{"do-not-recalculate-inverse-correction", o2::framework::VariantType::Bool, false, {"Do NOT recalculate the inverse correction in case lumi mode 1 or 2 is used"}});
  addOption(options, ConfigParamSpec{"nthreads-inverse-correction", o2::framework::VariantType::Int, 4, {"Number of threads used for calculating the inverse correction (-1=all threads)"}});
}

//________________________________________________________
void CorrectionMapsLoader::addGlobalOptions(std::vector<ConfigParamSpec>& options)
{
  // these are options which should be added at the workflow level, since they modify the inputs of the devices
  addOption(options, ConfigParamSpec{"lumi-type", o2::framework::VariantType::Int, 0, {"1 = use CTP lumi for TPC correction scaling, 2 = use TPC scalers for TPC correction scaling"}});
  addOption(options, ConfigParamSpec{"corrmap-lumi-mode", o2::framework::VariantType::Int, 0, {"scaling mode: (default) 0 = static + scale * full; 1 = full + scale * derivative; 2 = full + scale * derivative (for MC)"}});
  addOption(options, ConfigParamSpec{"enable-M-shape-correction", o2::framework::VariantType::Bool, false, {"Enable M-shape distortion correction"}});
  addOption(options, ConfigParamSpec{"disable-ctp-lumi-request", o2::framework::VariantType::Bool, false, {"do not request CTP lumi (regardless what is used for corrections)"}});
}

//________________________________________________________
CorrectionMapsLoaderGloOpts CorrectionMapsLoader::parseGlobalOptions(const o2::framework::ConfigParamRegistry& opts)
{
  CorrectionMapsLoaderGloOpts tpcopt;
  tpcopt.lumiType = opts.get<int>("lumi-type");
  tpcopt.lumiMode = opts.get<int>("corrmap-lumi-mode");
  tpcopt.enableMShapeCorrection = opts.get<bool>("enable-M-shape-correction");
  tpcopt.requestCTPLumi = !opts.get<bool>("disable-ctp-lumi-request");
  if (!tpcopt.requestCTPLumi && tpcopt.lumiType == 1) {
    LOGP(fatal, "Scaling with CTP Lumi is requested but this input is disabled");
  }
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
      setMeanLumi(mCorrMap->getLumi(), false);
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
      if (getLumiScaleType() == 1) {
        setInstLumi(getInstLumiCTP(), false);
      }
    }
    setUpdatedLumi();
    int scaleType = getLumiScaleType();
    const std::array<std::string, 3> lumiS{"OFF", "CTP", "TPC scaler"};
    if (scaleType >= lumiS.size()) {
      LOGP(fatal, "Wrong lumi-scale-type provided!");
    }

    LOGP(info, "TPC correction map params updated: SP corrections: {} (corr.map scaling type={}, override values: lumiMean={} lumiRefMean={} lumiScaleMode={}), CTP Lumi: source={} lumiInstOverride={} , LumiInst scale={} ",
         canUseCorrections() ? "ON" : "OFF",
         lumiS[scaleType], mMeanLumiOverride, mMeanLumiRefOverride, mLumiScaleMode, mLumiCTPSource, mInstCTPLumiOverride, mInstLumiCTPFactor);
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
  bool foundCTP = false, foundTPCScl = false, foundMShape = false;
  for (const auto& route : inputRouts) {
    if (route.matcher == InputSpec{"CTPLumi", "CTP", "LUMI", 0, Lifetime::Timeframe}) {
      foundCTP = true;
    } else if (route.matcher == InputSpec{"tpcscaler", o2::header::gDataOriginTPC, "TPCSCALER", 0, Lifetime::Timeframe}) {
      foundTPCScl = true;
    } else if (route.matcher == InputSpec{"mshape", o2::header::gDataOriginTPC, "TPCMSHAPE", 0, Lifetime::Timeframe}) {
      foundMShape = true;
    }
  }
  setLumiCTPAvailable(foundCTP);
  enableMShapeCorrection(foundMShape);
  if ((getLumiScaleType() == 1 && !foundCTP) || (getLumiScaleType() == 2 && !foundTPCScl)) {
    LOGP(fatal, "Lumi scaling source {}({}) is not available for TPC correction", getLumiScaleType(), getLumiScaleType() == 1 ? "CTP" : "TPCScaler");
  }

  if ((getLumiScaleMode() == 1) || (getLumiScaleMode() == 2)) {
    mScaleInverse = ic.options().get<bool>("do-not-recalculate-inverse-correction");
  } else {
    mScaleInverse = true;
  }
  const int nthreadsInv = (ic.options().get<int>("nthreads-inverse-correction"));
  (nthreadsInv < 0) ? TPCFastSpaceChargeCorrectionHelper::instance()->setNthreadsToMaximum() : TPCFastSpaceChargeCorrectionHelper::instance()->setNthreads(nthreadsInv);
}

//________________________________________________________
void CorrectionMapsLoader::copySettings(const CorrectionMapsLoader& src)
{
  setInstLumi(src.getInstLumi(), false);
  setInstLumiCTP(src.getInstLumiCTP());
  setMeanLumi(src.getMeanLumi(), false);
  setLumiCTPAvailable(src.getLumiCTPAvailable());
  setMeanLumiRef(src.getMeanLumiRef());
  setLumiScaleType(src.getLumiScaleType());
  setMeanLumiOverride(src.getMeanLumiOverride());
  setMeanLumiRefOverride(src.getMeanLumiRefOverride());
  setInstCTPLumiOverride(src.getInstCTPLumiOverride());
  setLumiScaleMode(src.getLumiScaleMode());
  enableMShapeCorrection(src.getUseMShapeCorrection());
  mInstLumiCTPFactor = src.mInstLumiCTPFactor;
  mLumiCTPSource = src.mLumiCTPSource;
  mLumiScaleMode = src.mLumiScaleMode;
  mScaleInverse = src.getScaleInverse();
}

void CorrectionMapsLoader::updateInverse()
{
  if (mLumiScaleMode == 1 || mLumiScaleMode == 2) {
    LOGP(info, "Recalculating the inverse correction");
    setUpdatedMap();
    std::vector<float> scaling{1, mLumiScale};
    std::vector<o2::gpu::TPCFastSpaceChargeCorrection*> corr{&(mCorrMap->getCorrection()), &(mCorrMapRef->getCorrection())};
    if (mCorrMapMShape) {
      scaling.emplace_back(1);
      corr.emplace_back(&(mCorrMapMShape->getCorrection()));
    }
    TPCFastSpaceChargeCorrectionHelper::instance()->initInverse(corr, scaling, false);
  } else {
    LOGP(info, "Reinitializing inverse correction with lumi scale mode {} not supported for now", mLumiScaleMode);
  }
}

#endif // #ifndef GPUCA_GPUCODE_DEVICE
