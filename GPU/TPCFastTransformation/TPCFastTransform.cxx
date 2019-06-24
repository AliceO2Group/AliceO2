// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCFastTransform.cxx
/// \brief Implementation of TPCFastTransform class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "TPCFastTransform.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

using namespace GPUCA_NAMESPACE::gpu;

TPCFastTransform::TPCFastTransform()
  : FlatObject(), mTimeStamp(0), mDistortion(), mApplyDistortion(1), mT0(0.f), mVdrift(0.f), mVdriftCorrY(0.f), mLdriftCorr(0.f), mTOFcorr(0.f), mPrimVtxZ(0.f)
{
  // Default Constructor: creates an empty uninitialized object
}

void TPCFastTransform::relocateBufferPointers(const char* oldBuffer, char* actualBuffer)
{
  char* distBuffer = FlatObject::relocatePointer(oldBuffer, actualBuffer, mDistortion.getFlatBufferPtr());
  mDistortion.setActualBufferAddress(distBuffer);
}

void TPCFastTransform::cloneFromObject(const TPCFastTransform& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;

  FlatObject::cloneFromObject(obj, newFlatBufferPtr);

  mTimeStamp = obj.mTimeStamp;
  mApplyDistortion = obj.mApplyDistortion;
  mT0 = obj.mT0;
  mVdrift = obj.mVdrift;
  mVdriftCorrY = obj.mVdriftCorrY;
  mLdriftCorr = obj.mLdriftCorr;
  mTOFcorr = obj.mTOFcorr;
  mPrimVtxZ = obj.mPrimVtxZ;

  // variable-size data

  char* distBuffer = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mDistortion.getFlatBufferPtr());
  mDistortion.cloneFromObject(obj.mDistortion, distBuffer);
}

void TPCFastTransform::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  const char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  relocateBufferPointers(oldFlatBufferPtr, mFlatBufferPtr);
}

void TPCFastTransform::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description
  const char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  relocateBufferPointers(oldFlatBufferPtr, mFlatBufferPtr);
}

void TPCFastTransform::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = mFlatBufferPtr;

  char* distBuffer = FlatObject::relocatePointer(oldFlatBufferPtr, futureFlatBufferPtr, mDistortion.getFlatBufferPtr());
  mDistortion.setFutureBufferAddress(distBuffer);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

void TPCFastTransform::startConstruction(const TPCDistortionIRS& distortion)
{
  /// Starts the construction procedure, reserves temporary memory

  FlatObject::startConstruction();

  assert(distortion.isConstructed());

  mTimeStamp = 0;
  mApplyDistortion = 1;
  mT0 = 0.f;
  mVdrift = 0.f;
  mVdriftCorrY = 0.f;
  mLdriftCorr = 0.f;
  mTOFcorr = 0.f;
  mPrimVtxZ = 0.f;

  // variable-size data

  mDistortion.cloneFromObject(distortion, nullptr);
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

  FlatObject::finishConstruction(mDistortion.getFlatBufferSize());

  mDistortion.moveBufferTo(mFlatBufferPtr);
}

void TPCFastTransform::print() const
{
#if !defined(GPUCA_GPUCODE)
  std::cout << "TPC Fast Transformation: " << std::endl;
  std::cout << "mTimeStamp = " << mTimeStamp << std::endl;
  std::cout << "mApplyDistortion = " << mApplyDistortion << std::endl;
  std::cout << "mT0 = " << mT0 << std::endl;
  std::cout << "mVdrift = " << mVdrift << std::endl;
  std::cout << "mVdriftCorrY = " << mVdriftCorrY << std::endl;
  std::cout << "mLdriftCorr = " << mLdriftCorr << std::endl;
  std::cout << "mTOFcorr = " << mTOFcorr << std::endl;
  std::cout << "mPrimVtxZ = " << mPrimVtxZ << std::endl;
  mDistortion.print();
#endif
}
