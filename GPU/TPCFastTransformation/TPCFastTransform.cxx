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
  : FlatObject(), mConstructionCounter(0), mConstructionRowInfoBuffer(nullptr), mNumberOfRows(0), mRowInfoPtr(nullptr), mTPCzLengthA(0.f), mTPCzLengthC(0.f), mTimeStamp(0), mDistortion(), mApplyDistortion(1), mT0(0.f), mVdrift(0.f), mVdriftCorrY(0.f), mLdriftCorr(0.f), mTOFcorr(0.f), mPrimVtxZ(0.f), mTPCalignmentZ(0.f)
{
  // Default Constructor: creates an empty uninitialized object
  double dAlpha = 2. * M_PI / NumberOfSlices;
  for (int i = 0; i < NumberOfSlices; i++) {
    SliceInfo& s = mSliceInfos[i];
    double alpha = dAlpha * (i + 0.5);
    s.sinAlpha = sin(alpha);
    s.cosAlpha = cos(alpha);
  }
}

void TPCFastTransform::relocateBufferPointers(const char* oldBuffer, char* actualBuffer)
{
  mRowInfoPtr = FlatObject::relocatePointer(oldBuffer, actualBuffer, mRowInfoPtr);
  char* distBuffer = FlatObject::relocatePointer(oldBuffer, actualBuffer, mDistortion.getFlatBufferPtr());
  mDistortion.setActualBufferAddress(distBuffer);
}

void TPCFastTransform::cloneFromObject(const TPCFastTransform& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;

  FlatObject::cloneFromObject(obj, newFlatBufferPtr);

  mConstructionCounter = 0;
  mConstructionRowInfoBuffer.reset();

  for (int i = 0; i < NumberOfSlices; i++) { // not needed, just for completeness
    mSliceInfos[i] = obj.mSliceInfos[i];
  }

  mNumberOfRows = obj.mNumberOfRows;

  mTPCzLengthA = obj.mTPCzLengthA;
  mTPCzLengthC = obj.mTPCzLengthC;

  mTimeStamp = obj.mTimeStamp;
  mApplyDistortion = obj.mApplyDistortion;
  mT0 = obj.mT0;
  mVdrift = obj.mVdrift;
  mVdriftCorrY = obj.mVdriftCorrY;
  mLdriftCorr = obj.mLdriftCorr;
  mTOFcorr = obj.mTOFcorr;
  mPrimVtxZ = obj.mPrimVtxZ;
  mTPCalignmentZ = obj.mTPCalignmentZ;

  // variable-size data

  mRowInfoPtr = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mRowInfoPtr);
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

  mRowInfoPtr = FlatObject::relocatePointer(oldFlatBufferPtr, futureFlatBufferPtr, mRowInfoPtr);
  char* distBuffer = FlatObject::relocatePointer(oldFlatBufferPtr, futureFlatBufferPtr, mDistortion.getFlatBufferPtr());
  mDistortion.setFutureBufferAddress(distBuffer);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

void TPCFastTransform::startConstruction(int numberOfRows)
{
  /// Starts the construction procedure, reserves temporary memory

  FlatObject::startConstruction();

  mNumberOfRows = numberOfRows;

  mConstructionRowInfoBuffer.reset(new RowInfo[numberOfRows]);
  mConstructionCounter = 0;

  mTPCzLengthA = 0.f;
  mTPCzLengthC = 0.f;

  mTimeStamp = 0;
  mApplyDistortion = 1;

  mT0 = 0.f;
  mVdrift = 0.f;
  mVdriftCorrY = 0.f;
  mLdriftCorr = 0.f;
  mTOFcorr = 0.f;
  mPrimVtxZ = 0.f;
  mTPCalignmentZ = 0.f;

  // variable-size data

  mRowInfoPtr = nullptr;
  mDistortion.destroy();
}

void TPCFastTransform::setTPCgeometry(float tpcZlengthSideA, float tpcZlengthSideC)
{
  /// Sets TPC geometry
  ///
  /// It must be called once during initialization
  /// but also may be called after to reset these parameters.

  mTPCzLengthA = tpcZlengthSideA;
  mTPCzLengthC = tpcZlengthSideC;
  mConstructionMask |= ConstructionExtraState::GeometryIsSet;
}

void TPCFastTransform::setCalibration(long int timeStamp, float t0, float vDrift, float vDriftCorrY, float lDriftCorr, float tofCorr, float primVtxZ, float tpcAlignmentZ)
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
  mTPCalignmentZ = tpcAlignmentZ;
  mConstructionMask |= ConstructionExtraState::CalibrationIsSet;
}

void TPCFastTransform::finishConstruction()
{
  /// Finishes initialization: puts everything to the flat buffer, releases temporary memory

  assert(mConstructionMask & ConstructionState::InProgress);            // construction in process
  assert(mConstructionMask & ConstructionExtraState::GeometryIsSet);    // geometry is  set
  assert(mConstructionMask & ConstructionExtraState::CalibrationIsSet); // all parameters are set
  assert(mDistortion.isConstructed());                                  // distortion is constructed
  assert(mConstructionCounter == mNumberOfRows);                        // all TPC rows are initialized

  size_t rowsSize = sizeof(RowInfo) * mNumberOfRows;
  size_t distortionOffset = FlatObject::alignSize(rowsSize, mDistortion.getBufferAlignmentBytes());

  FlatObject::finishConstruction(distortionOffset + mDistortion.getFlatBufferSize());

  mRowInfoPtr = reinterpret_cast<const RowInfo*>(mFlatBufferPtr);

  char* distBuffer = mFlatBufferPtr + distortionOffset;

  memcpy((void*)mRowInfoPtr, (const void*)mConstructionRowInfoBuffer.get(), rowsSize);

  mDistortion.moveBufferTo(distBuffer);

  mConstructionCounter = 0;
  mConstructionRowInfoBuffer.reset();
}

void TPCFastTransform::Print() const
{
#if !defined(GPUCA_GPUCODE)
  std::cout << "TPC Fast Transformation: " << std::endl;
  std::cout << "mNumberOfRows = " << mNumberOfRows << std::endl;
  std::cout << "mTPCzLengthA = " << mTPCzLengthA << std::endl;
  std::cout << "mTPCzLengthC = " << mTPCzLengthC << std::endl;
  std::cout << "mTimeStamp = " << mTimeStamp << std::endl;
  std::cout << "mApplyDistortion = " << mApplyDistortion << std::endl;
  std::cout << "mT0 = " << mT0 << std::endl;
  std::cout << "mVdrift = " << mVdrift << std::endl;
  std::cout << "mVdriftCorrY = " << mVdriftCorrY << std::endl;
  std::cout << "mLdriftCorr = " << mLdriftCorr << std::endl;
  std::cout << "mTOFcorr = " << mTOFcorr << std::endl;
  std::cout << "mPrimVtxZ = " << mPrimVtxZ << std::endl;
  std::cout << "mTPCalignmentZ = " << mTPCalignmentZ << std::endl;
  std::cout << "TPC Rows : " << std::endl;
  for (int i = 0; i < mNumberOfRows; i++) {
    std::cout << " tpc row " << i << ": x = " << mRowInfoPtr[i].x << " maxPad = " << mRowInfoPtr[i].maxPad << " padWidth = " << mRowInfoPtr[i].padWidth << std::endl;
  }
  mDistortion.Print();
#endif
}
