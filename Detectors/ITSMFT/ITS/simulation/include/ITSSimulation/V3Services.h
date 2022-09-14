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

/// \file V3Services.h
/// \brief Definition of the V3Services class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Parinya Namwongsa <parinya.namwongsa@cern.ch>

#ifndef ALICEO2_ITS_UPGRADEV3SERVICES_H_
#define ALICEO2_ITS_UPGRADEV3SERVICES_H_

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

/// This class defines the Geometry for the Services of the ITS Upgrade using TGeo.
class V3Services : public V11Geometry
{

 public:
  // Default constructor
  V3Services();

  /// Copy constructor
  V3Services(const V3Services&) = default;

  /// Assignment operator
  V3Services& operator=(const V3Services&) = default;

  /// Default destructor
  ~V3Services() override;

  /// Creates the Inner Barrel End Wheels on Side A
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createIBEndWheelsSideA(const TGeoManager* mgr = gGeoManager);

  /// Creates the Inner Barrel End Wheels on Side C
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createIBEndWheelsSideC(const TGeoManager* mgr = gGeoManager);

  /// Creates the CYSS Assembly (i.e. the supporting half cylinder and cone)
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createCYSSAssembly(const TGeoManager* mgr = gGeoManager);

  /// Creates the Middle Barrel End Wheels on Side A
  /// \param mother the TGeoVolume owing the volume structure
  /// \param mgr  The GeoManager (used only to get the proper material)
  void createMBEndWheelsSideA(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Middle Barrel End Wheels on Side C
  /// \param mother the TGeoVolume owing the volume structure
  /// \param mgr  The GeoManager (used only to get the proper material)
  void createMBEndWheelsSideC(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel End Wheels on Side A
  /// \param mother the TGeoVolume owing the volume structure
  /// \param mgr  The GeoManager (used only to get the proper material)
  void createOBEndWheelsSideA(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel End Wheels on Side C
  /// \param mother the TGeoVolume owing the volume structure
  /// \param mgr  The GeoManager (used only to get the proper material)
  void createOBEndWheelsSideC(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel Cone on Side A
  /// \param mother the TGeoVolume owing the volume structure
  /// \param mgr  The GeoManager (used only to get the proper material)
  void createOBConeSideA(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel Cone on Side C
  /// \param mother the TGeoVolume owing the volume structure
  /// \param mgr  The GeoManager (used only to get the proper material)
  void createOBConeSideC(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel CYSS Cylinder
  /// \param mother the TGeoVolume owing the volume structure
  /// \param mgr  The GeoManager (used only to get the proper material)
  void createOBCYSSCylinder(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel Gamma Conversion Wire
  /// \param mother the TGeoVolume owing the volume structure
  /// \param mgr  The GeoManager (used only to get the proper material)
  void createOBGammaConvWire(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

 private:
  /// Creates a single Inner Barrel End Wheel on Side A
  /// \param iLay  the layer number
  /// \param endWheel  the End Wheel volume assembly
  /// \param mgr  The GeoManager (used only to get the proper material)
  void ibEndWheelSideA(const Int_t iLay, TGeoVolume* endWheel, const TGeoManager* mgr = gGeoManager);

  /// Creates a single Inner Barrel End Wheel on Side C
  /// \param iLay  the layer number
  /// \param endWheel  the End Wheel volume assembly
  /// \param mgr  The GeoManager (used only to get the proper material)
  void ibEndWheelSideC(const Int_t iLay, TGeoVolume* endWheel, const TGeoManager* mgr = gGeoManager);

  /// Creates the shape of a Rib on Side A
  /// \param iLay  the layer number
  TGeoXtru* ibEndWheelARibShape(const Int_t iLay);

  /// Creates the CYSS cylinder of the Inner Barrel
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* ibCyssCylinder(const TGeoManager* mgr = gGeoManager);

  /// Creates the CYSS cone of the Inner Barrel
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* ibCyssCone(const TGeoManager* mgr = gGeoManager);

  /// Creates the CYSS Flange on Side A of the Inner Barrel
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* ibCyssFlangeSideA(const TGeoManager* mgr = gGeoManager);

  /// Creates the hollows in the CYSS Flange on Side A of the Inner Barrel
  /// \param zlen  the thickness of the ring where the hollows are
  TString ibCreateHollowsCyssFlangeSideA(const Double_t zlen);

  /// Creates the CYSS Flange on Side C of the Inner Barrel
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* ibCyssFlangeSideC(const TGeoManager* mgr = gGeoManager);

  /// Creates a single Middle/Outer Barrel End Wheel on Side A
  /// \param iLay  the layer number
  /// \param mother  the volume containing the created wheel
  /// \param mgr  The GeoManager (used only to get the proper material)
  void obEndWheelSideA(const Int_t iLay, TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates a single Middle Barrel End Wheel on Side C
  /// \param iLay  the layer number
  /// \param mother  the volume containing the created wheel
  /// \param mgr  The GeoManager (used only to get the proper material)
  void mbEndWheelSideC(const Int_t iLay, TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates a single Outer Barrel End Wheel on Side C
  /// \param iLay  the layer number
  /// \param mother  the volume containing the created wheel
  /// \param mgr  The GeoManager (used only to get the proper material)
  void obEndWheelSideC(const Int_t iLay, TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel Cone on Side A
  /// \param mother  the volume containing the created wheel
  /// \param mgr  The GeoManager (used only to get the proper material)
  void obConeSideA(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel Cone on Side C
  /// \param mother  the volume containing the created wheel
  /// \param mgr  The GeoManager (used only to get the proper material)
  void obConeSideC(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel Cone Trays on Side A
  /// \param mother  the volume containing the created wheel
  /// \param mgr  The GeoManager (used only to get the proper material)
  void obConeTraysSideA(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel CYSS volume 11
  /// \param mother  the volume containing the created wheel
  /// \param mgr  The GeoManager (used only to get the proper material)
  void obCYSS11(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  /// Creates the Outer Barrel Gamma Conversion Wire
  /// \param mother the TGeoVolume owing the volume structure
  /// \param mgr  The GeoManager (used only to get the proper material)
  void obConvWire(TGeoVolume* mother, const TGeoManager* mgr = gGeoManager);

  // Parameters
  static constexpr Int_t sNumberInnerLayers = 3; ///< Number of inner layers in ITSU
  static constexpr Int_t sNumberMiddlLayers = 2; ///< Number of middle layers in ITSU
  static constexpr Int_t sNumberOuterLayers = 2; ///< Number of outer layers in ITSU

  // Common parameters for IB services
  static const Double_t sIBWheelACZdist;    ///< IB Z distance between wheels
  static const Double_t sIBCYSSFlangeCZPos; ///< IB Z position of CYSS C Flange

  // Common parameters for OB services
  static const Double_t sOBWheelThickness; ///< MB/OB Wheels Thickness
  static const Double_t sMBWheelsZpos;     ///< MB Wheels Z position
  static const Double_t sOBWheelsZpos;     ///< OB Wheels Z position
  static const Double_t sOBConesZpos;      ///< OB Cones A & C Z position

  ClassDefOverride(V3Services, 0); // ITS v3 support geometry
};
} // namespace its
} // namespace o2

#endif
