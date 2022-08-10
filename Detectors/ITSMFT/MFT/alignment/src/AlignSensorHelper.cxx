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

/// @file AlignSensorHelper.cxx

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

#include <Rtypes.h>
#include "Framework/Logger.h"
#include "MFTAlignment/AlignSensorHelper.h"

using namespace o2::mft;

ClassImp(o2::mft::AlignSensorHelper);

//__________________________________________________________________________
AlignSensorHelper::AlignSensorHelper()
  : mChipIndexOnLadder(0),
    mChipIndexInMft(0),
    mConnector(0),
    mTransceiver(0),
    mLayer(0),
    mZone(0),
    mDisk(0),
    mHalf(0),
    mChipUniqueId(0),
    mTranslation(0, 0, 0),
    mRx(0),
    mRy(0),
    mRz(0),
    mSinRx(0),
    mCosRx(0),
    mSinRy(0),
    mCosRy(0),
    mSinRz(0),
    mCosRz(0),
    mIsTransformExtracted(false)
{
  setGeometry();
  builSymNames();
  LOGF(debug, "AlignSensorHelper instantiated");
}

//__________________________________________________________________________
void AlignSensorHelper::setGeometry()
{
  if (mGeometry == nullptr) {
    mGeometry = o2::mft::GeometryTGeo::Instance();
    mGeometry->fillMatrixCache(
      o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                               o2::math_utils::TransformType::L2G));
    mGeoSymbolicName = mGeometry->composeSymNameMFT();
    mSymNames.fill(mGeoSymbolicName);
  }
}

//__________________________________________________________________________
void AlignSensorHelper::setSensorOnlyInfo(const int chipIndex)
{
  if (chipIndex < mNumberOfSensors) {
    o2::itsmft::ChipMappingMFT chipMapping;
    o2::itsmft::MFTChipMappingData chipMappingData = (chipMapping.getChipMappingData())[chipIndex];
    mChipIndexOnLadder = (UShort_t)chipMappingData.chipOnModule;
    mChipIndexInMft = chipMappingData.globalChipSWID;
    mConnector = (UShort_t)chipMappingData.connector;
    mTransceiver = (UShort_t)chipMappingData.cable;
    mZone = (UShort_t)chipMappingData.zone;
    mLayer = (UShort_t)chipMappingData.layer;
    mDisk = (UShort_t)chipMappingData.disk;
    mHalf = (UShort_t)chipMappingData.half;
  } else {
    LOGF(error, "AlignSensorHelper::setSensorOnlyInfo() - chip index %d >= %d",
         chipIndex, mNumberOfSensors);
  }

  setSensorUid(chipIndex);
  setSymName();
}

//__________________________________________________________________________
std::stringstream AlignSensorHelper::getSensorFullName(bool wSymName)
{
  std::stringstream name;
  if (mGeometry == nullptr) {
    wSymName = false;
  }
  name << "h " << mHalf << " d " << mDisk << " layer " << mLayer
       << " z " << mZone
       << " con " << std::setw(1) << mConnector
       << " tr " << std::setw(2) << mTransceiver
       << " sr " << std::setw(1) << mChipIndexOnLadder
       << " iChip " << std::setw(3) << mChipIndexInMft
       << " uid " << mChipUniqueId;
  if (wSymName) {
    name << " " << mGeoSymbolicName;
  }
  return name;
}

//__________________________________________________________________________
bool AlignSensorHelper::setSensor(const int chipIndex)
{
  resetSensorTransformInfo();
  setSensorOnlyInfo(chipIndex);
  extractSensorTransform();
  return mIsTransformExtracted;
}

//__________________________________________________________________________
void AlignSensorHelper::builSymNames()
{
  if (mGeometry) {
    Int_t iChip = 0;
    Int_t nHalf = mGeometry->getNumberOfHalfs();
    TString sname = mGeometry->composeSymNameMFT();

    for (Int_t hf = 0; hf < nHalf; hf++) {
      Int_t nDisks = mGeometry->getNumberOfDisksPerHalf(hf);
      sname = mGeometry->composeSymNameHalf(hf);

      for (Int_t dk = 0; dk < nDisks; dk++) {
        sname = mGeometry->composeSymNameDisk(hf, dk);

        Int_t nLadders = 0;
        for (Int_t sensor = mGeometry->getMinSensorsPerLadder();
             sensor < mGeometry->getMaxSensorsPerLadder() + 1; sensor++) {
          nLadders += mGeometry->getNumberOfLaddersPerDisk(hf, dk, sensor);
        }

        for (Int_t lr = 0; lr < nLadders; lr++) { // nLadders
          sname = mGeometry->composeSymNameLadder(hf, dk, lr);
          Int_t nSensorsPerLadder = mGeometry->getNumberOfSensorsPerLadder(hf, dk, lr);

          for (Int_t sr = 0; sr < nSensorsPerLadder; sr++) {
            sname = mGeometry->composeSymNameChip(hf, dk, lr, sr);
            mSymNames[iChip] = sname;
            iChip++;
          }
        }
      }
    }
  } else {
    LOG(error) << "SensorInfo::builSymNames() - nullptr to geometry";
  }
}

//__________________________________________________________________________
int AlignSensorHelper::getChipIdGeoFromRO(const int chipIdRO)
{
  int chipIdGeo = 0;
  for (int index = 0; index < mNumberOfSensors; index++) {
    if (o2::itsmft::ChipMappingMFT::mChipIDGeoToRO[index] == chipIdRO) {
      chipIdGeo = index;
      break;
    }
  }
  return chipIdGeo;
}

//__________________________________________________________________________
void AlignSensorHelper::setSensorUid(const int chipIndex)
{
  if (chipIndex < mNumberOfSensors) {
    mChipUniqueId = o2::base::GeometryManager::getSensID(o2::detectors::DetID::MFT,
                                                         chipIndex);
  } else {
    LOGF(error, "AlignSensorHelper::setSensorUid() - chip index %d >= %d",
         chipIndex, mNumberOfSensors);
    mChipUniqueId = o2::base::GeometryManager::getSensID(o2::detectors::DetID::MFT, 0);
  }
}

//__________________________________________________________________________
void AlignSensorHelper::setSymName()
{
  mGeoSymbolicName = mSymNames[getChipIdGeoFromRO(mChipIndexInMft)];
}

//__________________________________________________________________________
void AlignSensorHelper::extractSensorTransform()
{
  if (mIsTransformExtracted) {
    return;
  }
  if (mGeometry == nullptr) {
    mGeometry = o2::mft::GeometryTGeo::Instance();
  }
  mGeometry->fillMatrixCache(
    o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                             o2::math_utils::TransformType::L2G));
  mTransform = mGeometry->getMatrixL2G(mChipIndexInMft);

  Double_t* tra = mTransform.GetTranslation();
  mTranslation.SetX(tra[0]);
  mTranslation.SetY(tra[1]);
  mTranslation.SetZ(tra[2]);

  Double_t* rot = mTransform.GetRotationMatrix();
  mRx = std::atan2(-rot[5], rot[8]);
  mRy = std::asin(rot[2]);
  mRz = std::atan2(-rot[1], rot[0]);

  // force the value of some calculations of sin, cos to avoid numerical errors

  // for MFT sensors, Rx = - Pi/2, or + Pi/2
  if (mRx > 0) {
    mSinRx = 1.0; // std::sin(mRx)
  } else {
    mSinRx = -1.0; // std::sin(mRx)
  }
  mCosRx = 0.0; // std::cos(mRx)

  // for MFT sensors, Ry = 0
  mSinRy = 0.0; // std::sin(mRy);
  mCosRy = 1.0; // std::cos(mRy);

  // for MFT sensors, Rz = 0 or Pi
  mSinRz = 0.0; // std::sin(mRz);
  mCosRz = std::cos(mRz);

  mIsTransformExtracted = true;
}

//__________________________________________________________________________
void AlignSensorHelper::resetSensorTransformInfo()
{
  mIsTransformExtracted = false;

  double rot[9] = {
    0., 0., 0.,
    0., 0., 0.,
    0., 0., 0.};
  double tra[3] = {0., 0., 0.};
  mTransform.SetRotation(rot);
  mTransform.SetTranslation(tra);

  mTranslation.SetX(0.0);
  mTranslation.SetY(0.0);
  mTranslation.SetZ(0.0);

  mRx = 0;
  mRy = 0;
  mRz = 0;

  mSinRx = 0;
  mCosRx = 0;
  mSinRy = 0;
  mCosRy = 0;
  mSinRz = 0;
  mCosRz = 0;
}