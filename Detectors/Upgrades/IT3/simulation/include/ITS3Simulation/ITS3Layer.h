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

/// \file ITS3Layer.h
/// \brief Definition of the ITS3Layer class
/// \author Fabrizio Grosa <fgrosa@cern.ch>

#ifndef ALICEO2_ITS3_ITS3LAYER_H
#define ALICEO2_ITS3_ITS3LAYER_H

#include <TGeoManager.h>

class TGeoVolume;

namespace o2
{
namespace its3
{
/// This class defines the Geometry for the ITS3  using TGeo. This is a work class used
/// to study different configurations during the development of the new ITS structure
class ITS3Layer : public TObject
{
 public:
  // Default constructor
  ITS3Layer() = default;

  /// Constructor setting layer number
  ITS3Layer(int lay);

  /// Copy constructor
  ITS3Layer(const ITS3Layer&) = default;

  /// Assignment operator
  ITS3Layer& operator=(const ITS3Layer&) = default;

  /// Default destructor
  ~ITS3Layer() override;

  void createLayer(TGeoVolume* motherVolume);
  void setSensorThick(double thick) { mSensorThickness = thick; }
  void setLayerRadius(double radius) { mRadius = radius; }
  void setLayerZLen(double zLen) { mZLen = zLen; }
  void setGapBetweenEmispheres(double gap) { mGap = gap; }
  void setChipID(int chipID) { mChipTypeID = chipID; }

 private:
  int mLayerNumber{0};         //! layer number
  double mSensorThickness{0.}; //! sensor thickness
  double mRadius{0.};          //! radius of layer
  double mZLen{0.};            //! length of a layer
  double mGap{0.};             //! gap between emispheres
  int mChipTypeID{0};          //! chip ID

  ClassDefOverride(ITS3Layer, 0); // ITS3 geometry
};                                // namespace its3
} // namespace its3
} // namespace o2

#endif
