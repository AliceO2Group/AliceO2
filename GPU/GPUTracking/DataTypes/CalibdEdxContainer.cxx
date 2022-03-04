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

#include "CalibdEdxContainer.h"

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
#include "TFile.h"
#include "TPCBase/CalDet.h"
#include "Framework/Logger.h"
#include "clusterFinderDefs.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
void CalibdEdxContainer::cloneFromObject(const CalibdEdxContainer& obj, char* newFlatBufferPtr)
{
  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  mCalibResidualdEdx = obj.mCalibResidualdEdx;
  mThresholdMap = obj.mThresholdMap;
  if (obj.mCalibTrackTopologyPol) {
    cloneFromObject(mCalibTrackTopologyPol, obj.mCalibTrackTopologyPol, newFlatBufferPtr, oldFlatBufferPtr);
  }
  if (obj.mCalibTrackTopologySpline) {
    cloneFromObject(mCalibTrackTopologySpline, obj.mCalibTrackTopologySpline, newFlatBufferPtr, oldFlatBufferPtr);
  }
}

template <class Type>
void CalibdEdxContainer::cloneFromObject(Type*& obj, const Type* objOld, char* newFlatBufferPtr, const char* oldFlatBufferPtr)
{
  obj = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, objOld);
  obj->cloneFromObject(*objOld, newFlatBufferPtr);
}
#endif

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
  if (mCalibTrackTopologyPol) {
    mCalibTrackTopologyPol->destroy();
  }
  mCalibTrackTopologySpline = nullptr;
  mCalibTrackTopologyPol = nullptr;
  FlatObject::destroy();
}

void CalibdEdxContainer::setActualBufferAddress(char* actualFlatBufferPtr)
{
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  if (mCalibTrackTopologyPol) {
    setActualBufferAddress(mCalibTrackTopologyPol);
  } else if (mCalibTrackTopologySpline) {
    setActualBufferAddress(mCalibTrackTopologySpline);
  } else {
    mCalibTrackTopologyPol = nullptr;
    mCalibTrackTopologySpline = nullptr;
  }
}

template <class Type>
void CalibdEdxContainer::setActualBufferAddress(Type*& obj)
{
  // set the pointer to the new location of the buffer
  obj = reinterpret_cast<Type*>(mFlatBufferPtr);

  // set buffer of the spline container class to the correct position
  obj->setActualBufferAddress(mFlatBufferPtr + sizeOfCalibdEdxTrackTopologyObj<Type>());
}

void CalibdEdxContainer::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  if (mCalibTrackTopologyPol) {
    setFutureBufferAddress(mCalibTrackTopologyPol, futureFlatBufferPtr);
  } else if (mCalibTrackTopologySpline) {
    setFutureBufferAddress(mCalibTrackTopologySpline, futureFlatBufferPtr);
  } else {
    mCalibTrackTopologyPol = nullptr;
    mCalibTrackTopologySpline = nullptr;
  }
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

template <class Type>
void CalibdEdxContainer::setFutureBufferAddress(Type*& obj, char* futureFlatBufferPtr)
{
  // set pointer of the polynomial container to correct new flat buffer
  char* distBuffer = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, obj->getFlatBufferPtr());
  obj->setFutureBufferAddress(distBuffer);

  // set member to correct new flat buffer
  obj = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, obj);
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

float CalibdEdxContainer::getMinZeroSupresssionThreshold() const
{
  if (mCalibTrackTopologyPol) {
    return mCalibTrackTopologyPol->getMinThreshold();
  } else {
    const float minThr = 0;
    LOGP(info, "Topology correction not set! Returning default min threshold of: {}", minThr);
    return minThr;
  }
}

float CalibdEdxContainer::getMaxZeroSupresssionThreshold() const
{
  if (mCalibTrackTopologyPol) {
    return mCalibTrackTopologyPol->getMaxThreshold();
  } else {
    const float maxThr = 1;
    LOGP(info, "Topology correction not set! Returning default max threshold of: {}", maxThr);
    return maxThr;
  }
}

void CalibdEdxContainer::loadPolTopologyCorrectionFromFile(std::string_view fileName)
{
  loadTopologyCorrectionFromFile(fileName, mCalibTrackTopologyPol);
}

void CalibdEdxContainer::loadSplineTopologyCorrectionFromFile(std::string_view fileName)
{
  loadTopologyCorrectionFromFile(fileName, mCalibTrackTopologySpline);
}

void CalibdEdxContainer::setPolTopologyCorrection(const CalibdEdxTrackTopologyPol& calibTrackTopology)
{
  setTopologyCorrection(calibTrackTopology, mCalibTrackTopologyPol);
}

void CalibdEdxContainer::setdefaultPolTopologyCorrection()
{
  CalibdEdxTrackTopologyPol calibTrackTopology;
  calibTrackTopology.setdefaultPolynomials();
  setTopologyCorrection(calibTrackTopology, mCalibTrackTopologyPol);
}

void CalibdEdxContainer::setSplineTopologyCorrection(const CalibdEdxTrackTopologySpline& calibTrackTopology)
{
  setTopologyCorrection(calibTrackTopology, mCalibTrackTopologySpline);
}

void CalibdEdxContainer::loadZeroSupresssionThresholdFromFile(std::string_view fileName, std::string_view objName, const float minCorrectionFactor, const float maxCorrectionFactor)
{
  TFile fInp(fileName.data(), "READ");
  CalDet<float>* threshold = nullptr;
  fInp.GetObject(objName.data(), threshold);
  setZeroSupresssionThreshold(*threshold, minCorrectionFactor, maxCorrectionFactor);
  delete threshold;
}

void CalibdEdxContainer::setZeroSupresssionThreshold(const CalDet<float>& thresholdMap, const float minCorrectionFactor, const float maxCorrectionFactor)
{
  o2::gpu::TPCPadGainCalib thresholdMapTmp(thresholdMap, minCorrectionFactor, maxCorrectionFactor, false);
  mThresholdMap = thresholdMapTmp;
}

void CalibdEdxContainer::setGainMap(const CalDet<float>& gainMap, const float minGain, const float maxGain)
{
  o2::gpu::TPCPadGainCalib gainMapTmp(gainMap, minGain, maxGain, false);
  mGainMap = gainMapTmp;
}

void CalibdEdxContainer::setGainMapResidual(const CalDet<float>& gainMapResidual, const float minResidualGain, const float maxResidualGain)
{
  o2::gpu::TPCPadGainCalib gainMapResTmp(gainMapResidual, minResidualGain, maxResidualGain, false);
  mGainMapResidual = gainMapResTmp;
}

void CalibdEdxContainer::setDefaultZeroSupresssionThreshold()
{
  const float defaultVal = getMinZeroSupresssionThreshold() + (getMaxZeroSupresssionThreshold() - getMinZeroSupresssionThreshold()) / 2;
  mThresholdMap.setMinCorrectionFactor(defaultVal - 0.1f);
  mThresholdMap.setMaxCorrectionFactor(defaultVal + 0.1f);
  for (int sector = 0; sector < o2::tpc::constants::MAXSECTOR; ++sector) {
    for (unsigned short globPad = 0; globPad < TPC_PADS_IN_SECTOR; ++globPad) {
      mThresholdMap.setGainCorrection(sector, globPad, defaultVal);
    }
  }
}

template <class Type>
void CalibdEdxContainer::loadTopologyCorrectionFromFile(std::string_view fileName, Type*& obj)
{
  // load and set-up container
  Type calibTrackTopologyTmp(fileName.data());
  setTopologyCorrection(calibTrackTopologyTmp, obj);
}

template <class Type>
void CalibdEdxContainer::setTopologyCorrection(const Type& calibTrackTopologyTmp, Type*& obj)
{
  FlatObject::startConstruction();

  // get size of the flat buffer of the splines
  const std::size_t flatbufferSize = calibTrackTopologyTmp.getFlatBufferSize();

  // size of the dEdx container without taking flat buffer into account
  const std::size_t objSize = sizeOfCalibdEdxTrackTopologyObj<Type>();

  // create mFlatBuffer with correct size
  const std::size_t totalSize = flatbufferSize + objSize;
  FlatObject::finishConstruction(totalSize);

  // setting member of CalibdEdxTrackTopologyPol to correct buffer address
  obj = reinterpret_cast<Type*>(mFlatBufferPtr);

  // deep copy of CalibdEdxTrackTopologyPol to buffer without moving the flat buffer to correct address
  obj->cloneFromObject(calibTrackTopologyTmp, nullptr);

  // seting the buffer of the splines to current buffer
  obj->moveBufferTo(objSize + mFlatBufferPtr);
}

#endif
