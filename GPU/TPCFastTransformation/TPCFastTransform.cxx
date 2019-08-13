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

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
#include "TFile.h"
#include "GPUCommonLogger.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

TPCFastTransform::TPCFastTransform()
  : FlatObject(), mTimeStamp(0), mDistortion(), mApplyDistortion(1), mT0(0.f), mVdrift(0.f), mVdriftCorrY(0.f), mLdriftCorr(0.f), mTOFcorr(0.f), mPrimVtxZ(0.f)
{
  // Default Constructor: creates an empty uninitialized object
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
  FlatObject::moveBufferTo(newFlatBufferPtr);
  setActualBufferAddress(mFlatBufferPtr);
}

void TPCFastTransform::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  mDistortion.setActualBufferAddress(mFlatBufferPtr);
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

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB)

int TPCFastTransform::writeToFile(std::string outFName, std::string name)
{
  /// store to file
  assert(isConstructed());

  if (outFName.empty())
    outFName = "tpcFastTransform.root";

  if (name.empty())
    name = "TPCFastTransform";

  TFile outf(outFName.data(), "recreate");
  if (outf.IsZombie()) {
    LOG(ERROR) << "Failed to open output file " << outFName;
    return -1;
  }

  bool isBufferExternal = !isBufferInternal();
  if (isBufferExternal)
    adoptInternalBuffer(mFlatBufferPtr);
  outf.WriteObjectAny(this, Class(), name.data());
  outf.Close();
  if (isBufferExternal)
    clearInternalBufferPtr();
  return 0;
}

TPCFastTransform* TPCFastTransform::loadFromFile(std::string inpFName, std::string name)
{
  /// load from file

  if (inpFName.empty())
    inpFName = "tpcFastTransform.root";

  if (name.empty())
    name = "TPCFastTransform";

  TFile inpf(inpFName.data());
  if (inpf.IsZombie()) {
    LOG(ERROR) << "Failed to open input file " << inpFName;
    return nullptr;
  }
  TPCFastTransform* transform = reinterpret_cast<TPCFastTransform*>(inpf.GetObjectChecked(name.data(), TPCFastTransform::Class()));
  if (!transform) {
    LOG(ERROR) << "Failed to load " << name << " from " << inpFName;
    return nullptr;
  }
  if (transform->mFlatBufferSize > 0 && transform->mFlatBufferContainer == nullptr) {
    LOG(ERROR) << "Failed to load " << name << " from " << inpFName << ": empty flat buffer container";
    return nullptr;
  }
  transform->setActualBufferAddress(transform->mFlatBufferContainer);
  return transform;
}

#endif
