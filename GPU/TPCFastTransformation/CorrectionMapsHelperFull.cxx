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

#include "CorrectionMapsHelperFull.h"
#include "Framework/Logger.h"

using namespace o2::gpu;
using namespace o2::tpc;

//________________________________________________________
void CorrectionMapsHelperFull::clear()
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

void CorrectionMapsHelperFull::setCorrMapMShape(std::unique_ptr<TPCFastTransform>&& m)
{
  setUpdatedMapMShape();
  mCorrMapMShape = std::move(m);
}

void CorrectionMapsHelperFull::updateLumiScale(bool report)
{
  if (!canUseCorrections()) {
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
void CorrectionMapsHelperFull::reportScaling()
{
  LOGP(info, "Map scaling update: LumiScaleType={} instLumi(CTP)={} instLumi(scaling)={} meanLumiRef={}, meanLumi={} -> LumiScale={} lumiScaleMode={}, M-Shape map valid: {}, M-Shape default: {}",
       mLumiScaleType == LumiScaleType::NoScaling ? "NoScaling" : (mLumiScaleType == LumiScaleType::CTPLumi ? "LumiCTP" : "TPCScaler"), getInstLumiCTP(), getInstLumi(), getMeanLumiRef(), getMeanLumi(), getLumiScale(),
       mLumiScaleMode == LumiScaleMode::Linear ? "Linear" : "Derivative", (mCorrMapMShape != nullptr), isCorrMapMShapeDummy());
}
