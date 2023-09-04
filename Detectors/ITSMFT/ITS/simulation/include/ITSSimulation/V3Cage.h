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

class TGeoXtru;

class TGeoCombiTrans;

class TGeoVolume;

class TGeoCompositeShape;

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

  // Standard constructor
  V3Cage(const char* name);

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

  /// Creates the Cage Side Panel element
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createCageSidePanel(const TGeoManager* mgr = gGeoManager);

  /// Creates the shape of the Cage Side Panel core and foil
  TGeoCompositeShape* createCageSidePanelCoreFoil(const Double_t thickness, const char* prefix);

  /// Creates the shape of a Cage Side Panel rail
  TGeoCompositeShape* createCageSidePanelRail(const Double_t length, const Int_t index);

  /// Creates the Cage End Cap element
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createCageEndCap(const TGeoManager* mgr = gGeoManager);

  /// Creates the Al frame of Cage End Cap Cable Crossing hole
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoCompositeShape* createCageEndCapCableCross(const TGeoManager* mgr = gGeoManager);

  /// Creates the Beam Pipe Support inside the Cage on the A side
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createBeamPipeSupport(const TGeoManager* mgr = gGeoManager);

  /// Creates the Titanium lower part of the collar supporting the beam pipe
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoCompositeShape* createBPSuppLowerCollar();

  /// Creates the Titanium upper part of the collar supporting the beam pipe
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoCompositeShape* createBPSuppUpperCollar();

  /// Creates the CF lateral bar of the beam pipe support (aka collar beam)
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoCompositeShape* createBPSuppCollarBeam();

  /// Creates the Titanium lateral bracket of the beam pipe support
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoCompositeShape* createBPSuppBracket();

  /// Creates the lateral clamps holding the beam pipe support to the Cage
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoCompositeShape* createBPSuppClamp();

  /// Creates the Cage Closing Cross element
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createCageClosingCross(const TGeoManager* mgr = gGeoManager);

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

  // Cage Side Panels and Rails
  static const Double_t sCageSidePanelLength;    ///< Side panel length along Z
  static const Double_t sCageSidePanelWidth;     ///< Side panel width along Y
  static const Double_t sCageSidePanelFoilThick; ///< Side panel foil thickness
  static const Double_t sCageSidePanelCoreThick; ///< Side panel core thickness
  static const Double_t sCageSidePanelXDist;     ///< Side panel distance on X
  static const Double_t sCageSidePanelRail1Len;  ///< Side panel 1st rail length
  static const Double_t sCageSidePanelRail2Len;  ///< Side panel 2nd rail length
  static const Double_t sCageSidePanelRail3Len;  ///< Side panel 3rd rail length
  static const Double_t sCageSidePanelRailWidth; ///< Side panel rail Y width
  static const Double_t sCageSidePanelRailSpan;  ///< Side panel rail X span
  static const Double_t sCageSidePanelRailHThik; ///< Side panel rail horiz thickness
  static const Double_t sCageSidePanelRailVThik; ///< Side panel rail vert thickness
  static const Double_t sCageSidePanelGuideLen;  ///< Side panel guide Z length
  static const Double_t sCageSidePanelGuideInHi; ///< Side panel guide in-height
  static const Double_t sCageSidePanelGuideWide; ///< Side panel guide X width
  static const Double_t sCageSidePanelGuidThik1; ///< Side panel guide thickness
  static const Double_t sCageSidePanelGuidThik2; ///< Side panel guide thickness
  static const Double_t sCageSidePanelMidBarWid; ///< Side panel middle bar width
  static const Double_t sCageSidePanelSidBarWid; ///< Side panel side bar width

  static const Double_t sCageSidePanelRail1Ypos[2]; ///< Side panel rail 1 Y pos
  static const Double_t sCageSidePanelRail2Ypos;    ///< Side panel rail 2 Y pos
  static const Double_t sCageSidePanelRail3Ypos[3]; ///< Side panel rail 3 Y pos

  // Cage End Cap
  static const Double_t sCageEndCapDext;        ///< End Cap ext diameter
  static const Double_t sCageEndCapDint;        ///< End Cap int diameter
  static const Double_t sCageEndCapFoamThick;   ///< End Cap foam thickness
  static const Double_t sCageEndCapFabThick;    ///< End Cap fabric thickness
  static const Double_t sCageEndCapXWidth;      ///< End Cap Width along X
  static const Double_t sCageEndCapSideHoleR;   ///< End Cap Side Hole rad
  static const Double_t sCageEndCapSideHoleX;   ///< End Cap Side Hole X dist
  static const Double_t sCageEndCapCableCutWid; ///< End Cap Width of cable cut
  static const Double_t sCageEndCapCableCutR;   ///< End Cap R pos of cable cut
  static const Double_t sCageEndCapCableCutPhi; ///< End Cap angle of cable cut
  static const Double_t sCageECRoundCrossDmin;  ///< End Cap min D of Al ring
  static const Double_t sCageECRoundCrossDmid;  ///< End Cap mid D of Al ring
  static const Double_t sCageECRoundCrossDmax;  ///< End Cap max D of Al ring
  static const Double_t sCageECRoundCrossZext;  ///< End Cap ext Z of Al ring
  static const Double_t sCageECRoundCrossZint;  ///< End Cap int Z of Al ring
  static const Double_t sCageECCableCrosTotHi;  ///< EC Cable Cut total height
  static const Double_t sCageECCableCrosTotZ;   ///< EC Cable Cut total Z len
  static const Double_t sCageECCableCrosInXWid; ///< EC Cable Cut inner width
  static const Double_t sCageECCableCrosInThik; ///< EC Cable Cut inner thick
  static const Double_t sCageECCableCrosInZLen; ///< EC Cable Cut inner length
  static const Double_t sCageECCableCrosSidWid; ///< EC Cable Cut Y side len

  // Beam Pipe Support (A Side)
  static const Double_t sBPSuppCollarIntD;     ///< BP support collar int diam
  static const Double_t sBPSuppCollarExtD;     ///< BP support collar ext diam
  static const Double_t sBPSuppCollarBushD;    ///< BP support collar bushing diam
  static const Double_t sBPSuppUpperCollarLen; ///< BP support upper collar len
  static const Double_t sBPSuppUpperCollarHei; ///< BP support upper collar high
  static const Double_t sBPSuppLowerCollarLen; ///< BP support lower collar len
  static const Double_t sBPSuppLowerCollarTlX; ///< BP support lower collar tail X
  static const Double_t sBPSuppLowCollHolDist; ///< BP support lower collar hole dist
  static const Double_t sBPSuppLowCollTailHei; ///< BP support lower collar tail hei
  static const Double_t sBPSuppCollarBeamLen;  ///< BP support collar beam len
  static const Double_t sBPSuppCollarBeamWid;  ///< BP support collar beam wide
  static const Double_t sBPSuppCollarBeamHei;  ///< BP support collar beam high
  static const Double_t sBPSuppBracketTotLen;  ///< BP support bracket total len
  static const Double_t sBPSuppBracketWidth;   ///< BP support bracket width
  static const Double_t sBPSuppBracketInLen;   ///< BP support bracket internal len
  static const Double_t sBPSuppBracketInHei;   ///< BP support bracket internal height
  static const Double_t sBPSuppBracketTailLen; ///< BP support bracket tail len
  static const Double_t sBPSuppBracketTailHei; ///< BP support bracket tail hei
  static const Double_t sBPSuppBrktCentHoleX;  ///< BP support bracket central hole X pos
  static const Double_t sBPSuppBrktCentHoleD;  ///< BP support bracket central hole diameter
  static const Double_t sBPSuppBrktLatHoleX;   ///< BP support bracket lateral hole X pos
  static const Double_t sBPSuppBrktLatHoleD;   ///< BP support bracket lateral hole diameter
  static const Double_t sBPSuppBrktLatHoleW;   ///< BP support bracket lateral hole width
  static const Double_t sBPSuppBrktLatHoleH;   ///< BP support bracket lateral hole height
  static const Double_t sBPSuppBrktHolesY;     ///< BP support bracket holes Y pos
  static const Double_t sBPSuppCollarM4High;   ///< BP support collar screw head height
  static const Double_t sBPSuppCollarM4Diam;   ///< BP support collar screw head diameter
  static const Double_t sBPSuppCollarM4XDist;  ///< BP support collar screw X dist
  static const Double_t sBPSuppCollarM4ZPos;   ///< BP support collar screw Z pos
  static const Double_t sBPSuppClampTotLen;    ///< BP support clamp length
  static const Double_t sBPSuppClampTotWid;    ///< BP support clamp width
  static const Double_t sBPSuppClampTotHei;    ///< BP support clamp height
  static const Double_t sBPSuppClampLatThick;  ///< BP support clamp lateral thick
  static const Double_t sBPSuppClampShelfLen;  ///< BP support clamp shelf len
  static const Double_t sBPSuppClampShelfHei;  ///< BP support clamp shelf hei
  static const Double_t sBPSuppClampsXDist;    ///< BP support clamps X distance
  static const Double_t sBPSuppClampInsDmin;   ///< BP support clamp insert Dmin
  static const Double_t sBPSuppClampInsDmax;   ///< BP support clamp insert Dmax
  static const Double_t sBPSuppClampInsH;      ///< BP support clamp insert H
  static const Double_t sBPSuppClampInsXPos;   ///< BP support clamp insert X
  static const Double_t sBPSuppClampInsZPos;   ///< BP support clamp insert Z
  static const Double_t sBPSuppClampShimLen;   ///< BP support clamp shim length
  static const Double_t sBPSuppClampShimWid;   ///< BP support clamp shim width
  static const Double_t sBPSuppClampShimThick; ///< BP support clamp shim thick
  static const Double_t sBPSuppClampM5High;    ///< BP support clamp screw head height
  static const Double_t sBPSuppClampM5Diam;    ///< BP support clamp screw head diameter
  static const Double_t sBPSuppClampM5ZPos;    ///< BP support clamp screw Z pos
  static const Double_t sBPSuppZPos;           ///< BP support global Z pos

  // Closing Cross
  static const Double_t sCageCrossXWidthTot;  ///< Closing cross total X wid
  static const Double_t sCageCrossXWidthExt;  ///< Closing cross external X wid
  static const Double_t sCageCrossXWidthInt;  ///< Closing cross internal X wid
  static const Double_t sCageCrossYHeightTot; ///< Closing cross total Y h
  static const Double_t sCageCrossYHeightInt; ///< Closing cross internal Y h
  static const Double_t sCageCrossYMid;       ///< Closing cross Y mid point
  static const Double_t sCageCrossZLength;    ///< Closing cross Z length
  static const Double_t sCageCrossBarThick;   ///< Closing cross bar thickness
  static const Double_t sCageCrossBarPhi;     ///< Closing cross bar angle

  ClassDefOverride(V3Cage, 0); // ITS v3 support geometry
};
} // namespace its
} // namespace o2

#endif
