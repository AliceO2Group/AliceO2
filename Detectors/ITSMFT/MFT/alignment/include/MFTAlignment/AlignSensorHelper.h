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

/// \file AlignSensorHelper.h
/// \author arakotoz@cern.ch
/// \brief Helper class to access to the global coordinates of the center each MFT sensor

#ifndef ALICEO2_MFT_ALIGN_SENSOR_HELPER_H
#define ALICEO2_MFT_ALIGN_SENSOR_HELPER_H

#include <sstream>

#include <TGeoMatrix.h>
#include <TString.h>
#include <Rtypes.h>

#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "MathUtils/Cartesian.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

namespace o2
{
namespace mft
{

/// \class AlignSensorHelper
class AlignSensorHelper
{
 public:
  /// \brief constructor with a pointer to the geometry
  AlignSensorHelper();

  /// \brief default destructor
  virtual ~AlignSensorHelper() = default;

  /// \brief set pointer to geometry that should already have done fillMatrixCache()
  void setGeometry();

  /// \brief set the studied sensor
  bool setSensor(const int chipIndex);

  /// \brief set the studied sensor
  void setSensorOnlyInfo(const int chipIndex);

  /// \brief return sensor index within the ladder [0, 4]
  UShort_t chipIndexOnLadder() const { return mChipIndexOnLadder; }

  /// \brief return sensor sw index within the MFT [0, 935]
  UShort_t chipIndexInMft() const { return mChipIndexInMft; }

  /// \brief return ladder geo index in this half MFT disk [0, 33]
  UShort_t ladderInHalfDisk() const { return mLadderInHalfDisk; }

  /// \brief return the half number to which belongs the sensor
  UShort_t half() const { return mHalf; }

  /// \brief return the disk number to which belongs the sensor
  UShort_t disk() const { return mDisk; }

  /// \brief return the layer number to which belongs the sensor
  UShort_t layer() const { return mLayer; }

  /// \brief return the zone to which belongs the sensor
  UShort_t zone() const { return mZone; }

  /// \brief return the connector to which the ladder is plugged
  UShort_t connector() const { return mConnector; }

  /// \brief return the transceiver on the RU for this sensor
  UShort_t transceiver() const { return mTransceiver; }

  /// \brief return the ALICE global unique id of the sensor
  Int_t sensorUid() const { return mChipUniqueId; }

  /// \brief return the geo symbolic name for this sensor
  TString geoSymbolicName() { return mGeoSymbolicName; }

  /// \brief return the x component of the translation in the sensor transform
  double translateX() const { return mTranslation.X(); }

  /// \brief return the y component of the translation in the sensor transform
  double translateY() const { return mTranslation.Y(); }

  /// \brief return the z component of the translation in the sensor transform
  double translateZ() const { return mTranslation.Z(); }

  /// \brief return the rotation angle w.r.t. global x-axis in the sensor transform
  double angleRx() const { return mRx; }

  /// \brief return the rotation angle w.r.t. global y-axis in the sensor transform
  double angleRy() const { return mRy; }

  /// \brief return the rotation angle w.r.t. global z-axis in the sensor transform
  double angleRz() const { return mRz; }

  /// \brief return the sin, cos of the rotation angle w.r.t. x-axis
  double sinRx() const { return mSinRx; }
  double cosRx() const { return mCosRx; }

  /// \brief return the sin, cos of the rotation angle w.r.t. y-axis
  double sinRy() const { return mSinRy; }
  double cosRy() const { return mCosRy; }

  /// \brief return the sin, cos of the rotation angle w.r.t. z-axis
  double sinRz() const { return mSinRz; }
  double cosRz() const { return mCosRz; }

  /// \brief return the status of the sensor transform extraction
  bool isTransformExtracted() const { return mIsTransformExtracted; }

  /// \brief return a stringstream filled with the sensor info
  std::stringstream getSensorFullName(bool wSymName = true);

 protected:
  o2::itsmft::ChipMappingMFT mChipMapping;         ///< MFT chip <-> ladder, layer, disk, half mapping
  o2::mft::GeometryTGeo* mGeometry = nullptr;      ///< MFT geometry
  int mNumberOfSensors = mChipMapping.getNChips(); ///< Total number of sensors (detection elements) in the MFT
  UShort_t mChipIndexOnLadder = 0;                 ///< sensor index within the ladder [0, 4]
  UShort_t mChipIndexInMft = 0;                    ///< sensor sw index within the MFT [0, 935]
  UShort_t mLadderInHalfDisk = 0;                  ///< ladder geo index in this half MFT disk [0, 33]
  UShort_t mConnector = 0;                         ///< connector index to which the ladder is plugged in the zone [0, 4]
  UShort_t mTransceiver = 0;                       ///< transceiver id to which the sensor is connected in the zone [0, 24]
  UShort_t mLayer = 0;                             ///< layer id [0, 9]
  UShort_t mZone = 0;                              ///< zone id [0,3]
  UShort_t mDisk = 0;                              ///< disk id [0, 4]
  UShort_t mHalf = 0;                              ///< half id [0, 1]
  Int_t mChipUniqueId = 0;                         ///< ALICE global unique id of the sensor
  TGeoHMatrix mTransform;                          ///< sensor transformation matrix L2G
  o2::math_utils::Point3D<double> mTranslation;    ///< coordinates of the translation between the local system origin (the center of the sensor) and the global origin

  // Euler angles extracted from the sensor transform
  double mRx = 0; ///< rotation angle aroung global x-axis (radian)
  double mRy = 0; ///< rotation angle aroung global y-axis (radian)
  double mRz = 0; ///< rotation angle aroung global z-axis (radian)

  // Cosinus and sinus of the Euler angles
  double mSinRx = 0;
  double mCosRx = 0;
  double mSinRy = 0;
  double mCosRy = 0;
  double mSinRz = 0;
  double mCosRz = 0;

  TString mGeoSymbolicName; ///< symbolic name of this sensor in the geometry

  bool mIsTransformExtracted = false; ///< boolean used to check if the sensor transform was successfully extracted from geometry

 protected:
  /// \brief set the ALICE global unique id of the sensor
  void setSensorUid(const int chipIndex);

  /// \brief set the symbolic name of this sensor in the geometry
  void setSymName();

  /// \brief init the matrix that stores the sensor transform L2G and extract its components
  void extractSensorTransform();

  /// \brief reset all sensor transform related variables
  void resetSensorTransformInfo();

  ClassDef(AlignSensorHelper, 0);
};

} // namespace mft
} // namespace o2
#endif
