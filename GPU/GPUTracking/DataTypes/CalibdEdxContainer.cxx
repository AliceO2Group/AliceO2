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
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  mCalibResidualdEdx = obj.mCalibResidualdEdx;
  mThresholdMap = obj.mThresholdMap;
  mGainMap = obj.mGainMap;
  mGainMapResidual = obj.mGainMapResidual;
  mDeadChannelMap = obj.mDeadChannelMap;
  mApplyFullGainMap = obj.mApplyFullGainMap;
  mCalibsLoad = obj.mCalibsLoad;
  if (obj.mCalibTrackTopologyPol) {
    subobjectCloneFromObject(mCalibTrackTopologyPol, obj.mCalibTrackTopologyPol);
  }
  if (obj.mCalibTrackTopologySpline) {
    subobjectCloneFromObject(mCalibTrackTopologySpline, obj.mCalibTrackTopologySpline);
  }
}

template <class Type>
void CalibdEdxContainer::subobjectCloneFromObject(Type*& obj, const Type* objOld)
{
  obj = reinterpret_cast<Type*>(mFlatBufferPtr);
  memset((void*)obj, 0, sizeof(*obj));
  obj->cloneFromObject(*objOld, mFlatBufferPtr + sizeOfCalibdEdxTrackTopologyObj<Type>());
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
  mCalibTrackTopologySpline = nullptr;
}

void CalibdEdxContainer::setDefaultPolTopologyCorrection()
{
  CalibdEdxTrackTopologyPol calibTrackTopology;
  calibTrackTopology.setDefaultPolynomials();
  setTopologyCorrection(calibTrackTopology, mCalibTrackTopologyPol);
  mCalibTrackTopologySpline = nullptr;
}

void CalibdEdxContainer::setSplineTopologyCorrection(const CalibdEdxTrackTopologySpline& calibTrackTopology)
{
  setTopologyCorrection(calibTrackTopology, mCalibTrackTopologySpline);
  mCalibTrackTopologyPol = nullptr;
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
  const auto thresholdMapProcessed = processThresholdMap(thresholdMap, maxCorrectionFactor);
  o2::gpu::TPCPadGainCalib thresholdMapTmp(thresholdMapProcessed, minCorrectionFactor, maxCorrectionFactor, false);
  mThresholdMap = thresholdMapTmp;
}

CalDet<float> CalibdEdxContainer::processThresholdMap(const CalDet<float>& thresholdMap, const float maxThreshold, const int nPadsInRowCl, const int nPadsInPadCl) const
{
  CalDet<float> thresholdMapProcessed(thresholdMap);

  for (unsigned int sector = 0; sector < Mapper::NSECTORS; ++sector) {
    for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
      const int maxRow = Mapper::ROWSPERREGION[region] - 1;
      for (int lrow = 0; lrow <= maxRow; ++lrow) {
        // find first row of the cluster
        const int rowStart = std::clamp(lrow - nPadsInRowCl, 0, maxRow);
        const int rowEnd = std::clamp(lrow + nPadsInRowCl, 0, maxRow);
        const int addPadsStart = Mapper::ADDITIONALPADSPERROW[region][lrow];

        for (unsigned int pad = 0; pad < Mapper::PADSPERROW[region][lrow]; ++pad) {
          float sumThr = 0;
          int countThr = 0;
          // loop ove the rows from the cluster
          for (int rowCl = rowStart; rowCl <= rowEnd; ++rowCl) {
            // shift local pad in row in case current row from the cluster has more pads in the row
            const int addPadsCl = Mapper::ADDITIONALPADSPERROW[region][rowCl];
            const int diffAddPads = addPadsCl - addPadsStart;
            const int padClCentre = pad + diffAddPads;

            const int maxPad = Mapper::PADSPERROW[region][rowCl] - 1;
            const int padStart = std::clamp(padClCentre - nPadsInPadCl, 0, maxPad);
            const int padEnd = std::clamp(padClCentre + nPadsInPadCl, 0, maxPad);
            for (int padCl = padStart; padCl <= padEnd; ++padCl) {
              const int globalPad = Mapper::getGlobalPadNumber(rowCl, padCl, region);
              // skip for current cluster position as the charge there is not effected from the thresold
              if (padCl == pad && rowCl == lrow) {
                continue;
              }

              float threshold = thresholdMap.getValue(sector, globalPad);
              if (threshold > maxThreshold) {
                threshold = maxThreshold;
              }

              sumThr += threshold;
              ++countThr;
            }
          }
          const float meanThresold = sumThr / countThr;
          const int globalPad = Mapper::getGlobalPadNumber(lrow, pad, region);
          thresholdMapProcessed.setValue(sector, globalPad, meanThresold);
        }
      }
    }
  }
  return thresholdMapProcessed;
}

void CalibdEdxContainer::setDeadChannelMap(const CalDet<bool>& deadMap)
{
  mDeadChannelMap.setFromMap(deadMap);
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
