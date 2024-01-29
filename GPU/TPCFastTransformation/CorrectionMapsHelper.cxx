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

using namespace GPUCA_NAMESPACE::gpu;

//________________________________________________________
void CorrectionMapsHelper::clear()
{
  if (mOwner) {
    delete mCorrMap;
    delete mCorrMapRef;
    delete mCorrMapMShape;
  }
  mCorrMap = nullptr;
  mCorrMapRef = nullptr;
  mCorrMapMShape = nullptr;
  mUpdatedFlags = 0;
  mInstLumi = 0.f;
  mMeanLumi = 0.f;
  mMeanLumiRef = 0.f;
  mScaleInverse = false;
}

void CorrectionMapsHelper::setOwner(bool v)
{
  if (mCorrMap || mCorrMapRef) {
    throw std::runtime_error("Must not change ownership while we contain objects");
  }
  mOwner = v;
}

//________________________________________________________
void CorrectionMapsHelper::setCorrMap(TPCFastTransform* m)
{
  if (mOwner) {
    delete mCorrMap;
  }
  mCorrMap = m;
}

//________________________________________________________
void CorrectionMapsHelper::setCorrMapRef(TPCFastTransform* m)
{
  if (mOwner) {
    delete mCorrMapRef;
  }
  mCorrMapRef = m;
}

void CorrectionMapsHelper::setCorrMapMShape(TPCFastTransform* m)
{
  if (mOwner) {
    delete mCorrMapMShape;
  }
  mCorrMapMShape = m;
}

//________________________________________________________
void CorrectionMapsHelper::setCorrMap(std::unique_ptr<TPCFastTransform>&& m)
{
  if (!mOwner) {
    throw std::runtime_error("we must not take the ownership from a unique ptr if mOwner is not set");
  }
  delete mCorrMap;
  mCorrMap = m.release();
}

//________________________________________________________
void CorrectionMapsHelper::setCorrMapRef(std::unique_ptr<TPCFastTransform>&& m)
{
  if (!mOwner) {
    throw std::runtime_error("we must not take the ownership from a unique ptr if mOwner is not set");
  }
  delete mCorrMapRef;
  mCorrMapRef = m.release();
}

void CorrectionMapsHelper::setCorrMapMShape(std::unique_ptr<TPCFastTransform>&& m)
{
  if (!mOwner) {
    throw std::runtime_error("we must not take the ownership from a unique ptr if mOwnerMShape is not set");
  }
  delete mCorrMapMShape;
  mCorrMapMShape = m.release();
}

//________________________________________________________
void CorrectionMapsHelper::reportScaling()
{
  LOGP(info, "Map scaling update: InstLumiOverride={}, LumiScaleType={} -> instLumi={}, meanLumiRef={}, meanLumi={} -> LumiScale={}, lumiScaleMode={}, is M-Shape map valid: {}, is M-Shape default: {}", getInstLumiOverride(), getLumiScaleType(), getInstLumi(), getMeanLumiRef(), getMeanLumi(), getLumiScale(), getLumiScaleMode(), (mCorrMapMShape != nullptr), isCorrMapMShapeDummy());
}
