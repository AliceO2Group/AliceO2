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

/// \file V3Cage.h
/// \brief Definition of the V3Cage class
/// \author Mario Sitta <sitta@to.infn.it>

#ifndef ALICEO2_ITS_UPGRADEV3CAGE_H_
#define ALICEO2_ITS_UPGRADEV3CAGE_H_

#include <TGeoManager.h>               // for gGeoManager
#include "Rtypes.h"                    // for Double_t, Int_t, Bool_t, etc
#include "ITSSimulation/V11Geometry.h" // for V11Geometry
#include "ITSSimulation/Detector.h"    // for Detector, Detector::Model

class TGeoXtru;

class TGeoCombiTrans;

class TGeoVolume;

namespace o2
{
namespace its
{

/// This class defines the Geometry for the Cage of the ITS Upgrade using TGeo.
class V3Cage : public V11Geometry
{

 public:
  // Default constructor
  V3Cage();

  /// Copy constructor
  V3Cage(const V3Cage&) = default;

  /// Assignment operator
  V3Cage& operator=(const V3Cage&) = default;

  /// Default destructor
  ~V3Cage() override;

  /// Steering method to create and place the whole Cage
  /// \param mother The mother volume to place the Cage into
  /// \param mgr  The GeoManager (used only to get the proper material)
  void createAndPlaceCage(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

 private:
  /// Creates a single Cage cover element
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createCageCover(const TGeoManager* mgr = gGeoManager);

  /// Creates a single Cage cover rib element
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createCageCoverRib(const TGeoManager* mgr = gGeoManager);

  // Parameters
  static const Double_t sCageYInBarrel; ///< Global Y translation

  // Cover element (each)
  static const Double_t sCageCoverZLength;     ///< Cover Length along Z
  static const Double_t sCageCoverRint;        ///< Cover internal radius
  static const Double_t sCageCoverRext;        ///< Cover external radius
  static const Double_t sCageCoverXWidth;      ///< Cover Width along X
  static const Double_t sCageCoverXBaseInt;    ///< Cover internal Base X
  static const Double_t sCageCoverXBaseExt;    ///< Cover external Base X
  static const Double_t sCageCoverYBaseHeight; ///< Cover Base Height on Y
  static const Double_t sCageCoverCoreRint;    ///< Cover Core internal radius
  static const Double_t sCageCoverCoreRext;    ///< Cover Core external radius
  static const Double_t sCageCoverSheetThick;  ///< Cover sheet thickness
  static const Double_t sCageCoverRibZLength;  ///< Cover rib Length along Z
  static const Double_t sCageCoverRibRint;     ///< Cover rib internal radius
  static const Double_t sCageCoverRibRext;     ///< Cover rib external radius
  static const Double_t sCageCoverRibXWidth;   ///< Cover rib Width along X
  static const Double_t sCageCoverRibXBaseInt; ///< Cover rib internal Base X
  static const Double_t sCageCoverRibYBaseHi;  ///< Cover rib Base Height on Y
  static const Double_t sCageCoverRibFoldHi;   ///< Cover rib Fold Height

  ClassDefOverride(V3Cage, 0); // ITS v3 support geometry
};
} // namespace its
} // namespace o2

#endif
