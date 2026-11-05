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

#ifndef ALICEO2_IOTOF_TF3CHIP_H_
#define ALICEO2_IOTOF_TF3CHIP_H_

#include <TGeoManager.h> // for gGeoManager
#include "Rtypes.h"      // for Int_t, Double_t, Bool_t, UInt_t, etc

class TGeoVolume;
class TGeoManager;

namespace o2
{

namespace iotof
{

/// TF3Chip class creates a TGeoVolume representing the TF3 Chip

class TF3Chip
{
 public:
  TF3Chip() = default;
  ~TF3Chip() = default;

  /// Creates the TF3Chip
  /// Returns the chip as a TGeoVolume
  /// \param yc Y chip half lengths
  /// \param ys sensor half thickness
  /// \param chipName sensName default volume names
  /// \param dummy if true creates a dummy air volume (for material budget studies)
  /// \param mgr The GeoManager (used only to get the proper material)
  static TGeoVolume* createChip(Double_t yc, Double_t ys,
                                char const* chipName = "TF3Chip", char const* sensName = "TF3Sensor",
                                Bool_t dummy = kFALSE, const TGeoManager* mgr = gGeoManager);

  static void createMaterials(Int_t id, Int_t fieldType, Float_t maxField);

  static const double getChipSizeX() { return sChipSizeX; };
  static const double getChipSizeY() { return sChipSizeZ; };

  static const double sChipSizeX;
  static const double sChipSizeZ;

  ClassDefNV(TF3Chip, 0); // TF3Chip geometry
};

} // namespace iotof
} // namespace o2

#endif
