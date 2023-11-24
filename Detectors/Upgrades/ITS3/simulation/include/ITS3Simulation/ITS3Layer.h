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
/// \author felix.schlepper@cern.ch

#ifndef ALICEO2_ITS3_ITS3LAYER_H
#define ALICEO2_ITS3_ITS3LAYER_H

#include <TGeoVolume.h>

class TGeoVolume;

namespace o2::its3
{

/// This class defines the Geometry for the ITS3  using TGeo.
class ITS3Layer
{
  // The hierarchy will be the following:
  // ITS2          ->       ITS3
  // ---------------------------------
  // Sensor                 PixelArray
  // Chip                   Tile
  // Module                 RSU
  // HalfStave              Segment
  // Stave                  Chip
  // HalfBarrel             CarbonForm
  // Layer                  Layer
 public:
  // Create one layer of ITS3 and attach it to the motherVolume.
  void createLayer(TGeoVolume* motherVolume, int layer = 0);

 private:
  void init();
  void createPixelArray();
  void createTile();
  void createRSU();
  void createSegment();
  void createChip();
  void createCarbonForm();
  void createLayerImpl();

 private:
  uint8_t mNLayer{0}; // Layer number
  double mR{0};       // Middle Radius
  double mRmin{};     // Minimum Radius
  double mRmax{0};    // Maximum Radius

  // Individual Pieces
  TGeoVolume* mPixelArray{nullptr};
  TGeoVolumeAssembly* mTile{nullptr};
  TGeoVolumeAssembly* mRSU{nullptr};
  TGeoVolumeAssembly* mSegment{nullptr};
  TGeoVolumeAssembly* mChip{nullptr};
  TGeoVolumeAssembly* mCarbonForm{nullptr};
  TGeoVolumeAssembly* mLayer{nullptr};

  ClassDefNV(ITS3Layer, 0);
};
} // namespace o2::its3

#endif
