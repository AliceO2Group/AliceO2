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

#include "CorrectionMapsHelper.h"
#include "GPUCommonLogger.h"

using namespace o2::gpu;
using namespace o2::tpc;

//________________________________________________________
void CorrectionMapsHelper::clear()
{
  mLumiCTPAvailable = false;
  mCorrMap = nullptr;
  mCorrMapRef = nullptr;
  mCorrMapMShape.reset();
  mUpdatedFlags = 0;
  mInstLumiCTP = 0.f;
  mInstLumi = 0.f;
  mMeanLumi = 0.f;
  mMeanLumiRef = 0.f;
}

void CorrectionMapsHelper::setCorrMapMShape(std::unique_ptr<TPCFastTransform>&& m)
{
  setUpdatedMapMShape();
  mCorrMapMShape = std::move(m);
}

void CorrectionMapsHelper::updateLumiScale(bool report)
{
  if (!canUseCorrections()) {
    if (mLumiScaleMode != LumiScaleMode::NoCorrection) {
      LOGP(warning, "Negative meanLumi={} detected, switching to NoCorrection mode for backward compatibility", mMeanLumi);
      mLumiScaleMode = LumiScaleMode::NoCorrection;
    }
    mLumiScale = -1.f;
  } else if ((mLumiScaleMode == LumiScaleMode::DerivativeMap) || (mLumiScaleMode == LumiScaleMode::DerivativeMapMC)) {
    mLumiScale = mMeanLumiRef ? (mInstLumi - mMeanLumi) / mMeanLumiRef : 0.f;
    LOGP(debug, "mInstLumi: {}  mMeanLumi: {} mMeanLumiRef: {}", mInstLumi, mMeanLumi, mMeanLumiRef);
  } else {
    mLumiScale = mMeanLumi ? mInstLumi / mMeanLumi : 0.f;
  }
  setUpdatedLumi();
  if (report) {
    reportScaling();
  }
}

//________________________________________________________
void CorrectionMapsHelper::reportScaling()
{
  auto lumiTypeName = [](LumiScaleType t) {
    switch (t) {
      case LumiScaleType::NoScaling:
        return "NoScaling";
      case LumiScaleType::CTPLumi:
        return "CTPLumi";
      case LumiScaleType::TPCScaler:
        return "TPCScaler";
      default:
        return "Unknown";
    }
  };

  const bool mshapeValid = (mCorrMapMShape != nullptr) && !isCorrMapMShapeDummy();

  if (mLumiScaleMode == LumiScaleMode::NoCorrection) {
    LOGP(info, "Map scaling update: mode=NoCorrection (corrections disabled, dummy map in use)");
  } else if (mLumiScaleMode == LumiScaleMode::StaticMapOnly) {
    LOGP(info, "Map scaling update: mode=StaticMapOnly (static reference map, no lumi scaling), M-Shape correction: {}", mshapeValid ? "applied" : "not applied");
  } else {
    auto lumiModeName = [](LumiScaleMode m) {
      switch (m) {
        case LumiScaleMode::Linear:
          return "Linear";
        case LumiScaleMode::DerivativeMap:
          return "DerivativeMap";
        case LumiScaleMode::DerivativeMapMC:
          return "DerivativeMapMC";
        default:
          return "Unknown";
      }
    };
    LOGP(info, "Map scaling update: LumiScaleType={} instLumi(CTP)={} instLumi(scaling)={} meanLumiRef={} meanLumi={} -> LumiScale={} lumiScaleMode={}, M-Shape correction: {}",
         lumiTypeName(mLumiScaleType), getInstLumiCTP(), getInstLumi(), getMeanLumiRef(), getMeanLumi(), getLumiScale(),
         lumiModeName(mLumiScaleMode), mshapeValid ? "applied" : "not applied");
  }
}
