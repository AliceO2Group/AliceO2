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

using namespace o2::gpu;

//________________________________________________________
void CorrectionMapsHelper::clear()
{
  if (mOwner) {
    delete mCorrMap;
    delete mCorrMapRef;
  }
  mCorrMap = nullptr;
  mCorrMapRef = nullptr;
  mUpdatedFlags = 0;
  mInstLumi = 0.f;
  mMeanLumi = 0.f;
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

//________________________________________________________
void CorrectionMapsHelper::setCorrMap(std::unique_ptr<TPCFastTransform>&& m)
{
  if (mOwner) {
    delete mCorrMap;
  }
  mCorrMap = m.release();
}

//________________________________________________________
void CorrectionMapsHelper::setCorrMapRef(std::unique_ptr<TPCFastTransform>&& m)
{
  if (mOwner) {
    delete mCorrMapRef;
  }
  mCorrMapRef = m.release();
}
