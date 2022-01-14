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

/// \file  CalibdEdxContainer.cxx
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#include "DataFormatsTPC/CalibdEdxContainer.h"

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
#include "TFile.h"
#include "GPUCommonLogger.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

void CalibdEdxContainer::cloneFromObject(const CalibdEdxContainer& obj, char* newFlatBufferPtr)
{
  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  mCalibResidualdEdx = obj.mCalibResidualdEdx;
  if (obj.mCalibTrackTopologyPol) {
    mCalibTrackTopologyPol = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mCalibTrackTopologyPol);
  }
  if (obj.mCalibTrackTopologySpline) {
    mCalibTrackTopologySpline = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mCalibTrackTopologySpline);
  }
}

void CalibdEdxContainer::moveBufferTo(char* newFlatBufferPtr)
{
  char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = mFlatBufferPtr;
  mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}

void CalibdEdxContainer::destroy()
{
  if (mCalibTrackTopologySpline) {
    mCalibTrackTopologySpline->destroy();
  }
  mCalibTrackTopologySpline = nullptr;
  mCalibTrackTopologyPol = nullptr;
  FlatObject::destroy();
}

void CalibdEdxContainer::setActualBufferAddress(char* actualFlatBufferPtr)
{
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  const size_t buffsize = getFlatBufferSize();
  if (buffsize == 0) {
    mCalibTrackTopologyPol = nullptr;
    mCalibTrackTopologySpline = nullptr;
  } else if (buffsize == sizeOfCalibdEdxTrackTopologyPol()) {
    // if the size of the buffer is equal to the size of the pol class
    // set the pointer to the new location of the buffer
    mCalibTrackTopologyPol = reinterpret_cast<CalibdEdxTrackTopologyPol*>(mFlatBufferPtr);
  } else {
    // set the pointer to the new location of the buffer
    mCalibTrackTopologySpline = reinterpret_cast<CalibdEdxTrackTopologySpline*>(mFlatBufferPtr);

    // set buffer of the spline container class to the correct position
    const std::size_t offset = sizeOfCalibdEdxTrackTopologySpline();
    mCalibTrackTopologySpline->setActualBufferAddress(mFlatBufferPtr + offset);
  }
}

void CalibdEdxContainer::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  const size_t buffsize = getFlatBufferSize();
  if (buffsize == 0) {
    mCalibTrackTopologyPol = nullptr;
    mCalibTrackTopologySpline = nullptr;
  } else if (buffsize == sizeOfCalibdEdxTrackTopologyPol()) {
    // set member to correct new flat buffer
    mCalibTrackTopologyPol = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mCalibTrackTopologyPol);
  } else {
    // set pointer of the spline container to correct new flat buffer
    char* distBuffer = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mCalibTrackTopologySpline->getFlatBufferPtr());
    mCalibTrackTopologySpline->setFutureBufferAddress(distBuffer);

    // set member to correct new flat buffer
    mCalibTrackTopologySpline = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mCalibTrackTopologySpline);
  }
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB)

void CalibdEdxContainer::loadPolTopologyCorrectionFromFile(std::string_view fileName)
{
  FlatObject::startConstruction();
  CalibdEdxTrackTopologyPol calibTrackTopologyPolTmp(fileName);

  // create mFlatBuffer with correct size
  const std::size_t bufferSize = sizeOfCalibdEdxTrackTopologyPol();
  FlatObject::finishConstruction(bufferSize);

  // CalibdEdxTrackTopologyPol* ptrToBuffer = reinterpret_cast<CalibdEdxTrackTopologyPol*>(mFlatBufferPtr); // set pointer to flat buffer
  // *ptrToBuffer = cal; // deep copy of CalibdEdxTrackTopologyPol to buffer

  // setting member of CalibdEdxTrackTopologyPol to correct buffer address
  mCalibTrackTopologyPol = reinterpret_cast<CalibdEdxTrackTopologyPol*>(mFlatBufferPtr);
  *mCalibTrackTopologyPol = calibTrackTopologyPolTmp; // deep copy of CalibdEdxTrackTopologyPol to buffer
}

void CalibdEdxContainer::loadSplineTopologyCorrectionFromFile(std::string_view fileName)
{
  FlatObject::startConstruction();

  // load and set-up spline container
  CalibdEdxTrackTopologySpline calibTrackTopologySplineTmp(fileName.data());

  // get size of the flat buffer of the splines
  const std::size_t flatbufferSize = calibTrackTopologySplineTmp.getFlatBufferSize();

  // size of the dEdx spline container without taking flat buffer into account
  const std::size_t objSize = sizeOfCalibdEdxTrackTopologySpline();

  // create mFlatBuffer with correct size
  const std::size_t totalSize = flatbufferSize + objSize;
  FlatObject::finishConstruction(totalSize);

  // CalibdEdxTrackTopologySpline* ptrToBuffer = reinterpret_cast<CalibdEdxTrackTopologySpline*>(mFlatBufferPtr); // set pointer to flat buffer
  // *ptrToBuffer = calibTrackTopologySplineTmp; // deep copy of CalibdEdxTrackTopologyPol to buffer
  // delete cal;

  // setting member of CalibdEdxTrackTopologyPol to correct buffer address
  mCalibTrackTopologySpline = reinterpret_cast<CalibdEdxTrackTopologySpline*>(mFlatBufferPtr);

  // deep copy of CalibdEdxTrackTopologyPol to buffer without moving the flat buffer to correct address
  *mCalibTrackTopologySpline = calibTrackTopologySplineTmp;

  // seting the buffer of the splines to current buffer
  mCalibTrackTopologySpline->moveBufferTo(objSize + mFlatBufferPtr);
}

#endif
