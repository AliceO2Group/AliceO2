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

/// \file  TPCFastTransform.cxx
/// \brief Implementation of TPCFastTransform class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
#include "Rtypes.h"
#endif

#include "TPCFastTransform.h"
#include "GPUCommonLogger.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
#include "TFile.h"
#include "GPUCommonLogger.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

TPCFastTransform::TPCFastTransform()
  : FlatObject(), mTimeStamp(0), mCorrection(), mApplyCorrection(1), mT0(0.f), mVdrift(0.f), mVdriftCorrY(0.f), mLdriftCorr(0.f), mTOFcorr(0.f), mPrimVtxZ(0.f), mLumi(0.f), mLumiError(0.f), mLumiScaleFactor(1.0f)
{
  // Default Constructor: creates an empty uninitialized object
}

void TPCFastTransform::cloneFromObject(const TPCFastTransform& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;

  FlatObject::cloneFromObject(obj, newFlatBufferPtr);

  mTimeStamp = obj.mTimeStamp;
  mApplyCorrection = obj.mApplyCorrection;
  mT0 = obj.mT0;
  mVdrift = obj.mVdrift;
  mVdriftCorrY = obj.mVdriftCorrY;
  mLdriftCorr = obj.mLdriftCorr;
  mTOFcorr = obj.mTOFcorr;
  mPrimVtxZ = obj.mPrimVtxZ;
  mLumi = obj.mLumi;
  mLumiError = obj.mLumiError;
  mLumiScaleFactor = obj.mLumiScaleFactor;
  // variable-size data

  char* distBuffer = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mCorrection.getFlatBufferPtr());
  mCorrection.cloneFromObject(obj.mCorrection, distBuffer);
}

void TPCFastTransform::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  FlatObject::moveBufferTo(newFlatBufferPtr);
  setActualBufferAddress(mFlatBufferPtr);
}

void TPCFastTransform::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  mCorrection.setActualBufferAddress(mFlatBufferPtr);
}

void TPCFastTransform::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = mFlatBufferPtr;

  char* distBuffer = FlatObject::relocatePointer(oldFlatBufferPtr, futureFlatBufferPtr, mCorrection.getFlatBufferPtr());
  mCorrection.setFutureBufferAddress(distBuffer);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

void TPCFastTransform::startConstruction(const TPCFastSpaceChargeCorrection& correction)
{
  /// Starts the construction procedure, reserves temporary memory

  FlatObject::startConstruction();

  assert(correction.isConstructed());

  mTimeStamp = 0;
  mApplyCorrection = 1;
  mT0 = 0.f;
  mVdrift = 0.f;
  mVdriftCorrY = 0.f;
  mLdriftCorr = 0.f;
  mTOFcorr = 0.f;
  mPrimVtxZ = 0.f;
  mLumi = 0.f;
  mLumiError = 0.f;
  mLumiScaleFactor = 1.f;

  // variable-size data

  mCorrection.cloneFromObject(correction, nullptr);
}

void TPCFastTransform::setCalibration(long int timeStamp, float t0, float vDrift, float vDriftCorrY, float lDriftCorr, float tofCorr, float primVtxZ)
{
  /// Sets all drift calibration parameters and the time stamp
  ///
  /// It must be called once during initialization,
  /// but also may be called after to reset these parameters.

  mTimeStamp = timeStamp;
  mT0 = t0;
  mVdrift = vDrift;
  mVdriftCorrY = vDriftCorrY;
  mLdriftCorr = lDriftCorr;
  mTOFcorr = tofCorr;
  mPrimVtxZ = primVtxZ;
  mConstructionMask |= ConstructionExtraState::CalibrationIsSet;
}

void TPCFastTransform::finishConstruction()
{
  /// Finishes initialization: puts everything to the flat buffer, releases temporary memory

  assert(mConstructionMask & ConstructionState::InProgress);            // construction in process
  assert(mConstructionMask & ConstructionExtraState::CalibrationIsSet); // all parameters are set

  FlatObject::finishConstruction(mCorrection.getFlatBufferSize());

  mCorrection.moveBufferTo(mFlatBufferPtr);
}

void TPCFastTransform::print() const
{
#if !defined(GPUCA_GPUCODE)
  LOG(info) << "TPC Fast Transformation: ";
  LOG(info) << "mTimeStamp = " << mTimeStamp;
  LOG(info) << "mApplyCorrection = " << mApplyCorrection;
  LOG(info) << "mT0 = " << mT0;
  LOG(info) << "mVdrift = " << mVdrift;
  LOG(info) << "mVdriftCorrY = " << mVdriftCorrY;
  LOG(info) << "mLdriftCorr = " << mLdriftCorr;
  LOG(info) << "mTOFcorr = " << mTOFcorr;
  LOG(info) << "mPrimVtxZ = " << mPrimVtxZ;
  LOG(info) << "mLumi = " << mLumi;
  LOG(info) << "mLumiError = " << mLumiError;
  LOG(info) << "mLumiScaleFactor = " << mLumiScaleFactor;
  mCorrection.print();
#endif
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB)

int TPCFastTransform::writeToFile(std::string outFName, std::string name)
{
  /// store to file
  assert(isConstructed());

  if (outFName.empty()) {
    outFName = "tpcFastTransform.root";
  }
  if (name.empty()) {
    name = "TPCFastTransform";
  }
  TFile outf(outFName.data(), "recreate");
  if (outf.IsZombie()) {
    LOG(error) << "Failed to open output file " << outFName;
    return -1;
  }

  bool isBufferExternal = !isBufferInternal();
  if (isBufferExternal) {
    adoptInternalBuffer(mFlatBufferPtr);
  }
  outf.WriteObjectAny(this, Class(), name.data());
  outf.Close();
  if (isBufferExternal) {
    clearInternalBufferPtr();
  }
  return 0;
}

void TPCFastTransform::rectifyAfterReadingFromFile()
{
  setActualBufferAddress(mFlatBufferContainer);
}

TPCFastTransform* TPCFastTransform::loadFromFile(std::string inpFName, std::string name)
{
  /// load from file

  if (inpFName.empty()) {
    inpFName = "tpcFastTransform.root";
  }
  if (name.empty()) {
    name = "TPCFastTransform";
  }
  TFile inpf(inpFName.data());
  if (inpf.IsZombie()) {
    LOG(error) << "Failed to open input file " << inpFName;
    return nullptr;
  }
  TPCFastTransform* transform = reinterpret_cast<TPCFastTransform*>(inpf.GetObjectChecked(name.data(), TPCFastTransform::Class()));
  if (!transform) {
    LOG(error) << "Failed to load " << name << " from " << inpFName;
    return nullptr;
  }
  if (transform->mFlatBufferSize > 0 && transform->mFlatBufferContainer == nullptr) {
    LOG(error) << "Failed to load " << name << " from " << inpFName << ": empty flat buffer container";
    return nullptr;
  }
  transform->rectifyAfterReadingFromFile(); // ==   transform->setActualBufferAddress(transform->mFlatBufferContainer);
  return transform;
}

#endif
