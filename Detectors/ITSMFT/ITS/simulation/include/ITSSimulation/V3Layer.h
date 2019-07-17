// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file V3Layer.h
/// \brief Definition of the V3Layer class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Chinorat Kobdaj (kobdaj@g.sut.ac.th)

#ifndef ALICEO2_ITS_UPGRADEV3LAYER_H_
#define ALICEO2_ITS_UPGRADEV3LAYER_H_

#include <TGeoManager.h>               // for gGeoManager
#include "Rtypes.h"                    // for Double_t, Int_t, Bool_t, etc
#include "ITSSimulation/V11Geometry.h" // for V11Geometry
#include "ITSSimulation/Detector.h"    // for Detector, Detector::Model

class TGeoXtru;

class TGeoCombiTrans;

class TGeoVolume; // lines 15-15

namespace o2
{
namespace its
{

/// This class defines the Geometry for the ITS  using TGeo. This is a work class used
/// to study different configurations during the development of the new ITS structure
class V3Layer : public V11Geometry
{

 public:
  enum { kStave, kHalfStave, kModule, kChip, kNHLevels };

  // Default constructor
  V3Layer();

  /// Constructor setting layer number and debugging level
  /// for a "turbo" layer (i.e. where staves overlap in phi)
  V3Layer(Int_t lay, Bool_t turbo = kFALSE, Int_t debug = 0);

  /// Copy constructor
  V3Layer(const V3Layer&) = default;

  /// Assignment operator
  V3Layer& operator=(const V3Layer&) = default;

  /// Default destructor
  ~V3Layer() override;

  Bool_t hasGammaConversionRods() const { return mAddGammaConv; };

  Bool_t isTurbo() const { return mIsTurbo; };

  Double_t getChipThick() const { return mChipThickness; };

  Double_t getStaveTilt() const { return mStaveTilt; };

  Double_t getStaveWidth() const { return mStaveWidth; };

  Double_t getSensorThick() const { return mSensorThickness; };

  Double_t getNumberOfStaves() const { return mNumberOfStaves; };

  Double_t getNumberOfChips() const { return mNumberOfChips; };

  Double_t getRadius() const { return mLayerRadius; };

  Double_t getPhi0() const { return mPhi0; };

  Double_t getIBModuleZLength() const { return mIBModuleZLength; };

  Double_t getOBModuleZLength() const { return mOBModuleZLength; };

  Int_t getChipType() const { return mChipTypeID; }

  Int_t getNumberOfStavesPerParent() const { return mHierarchy[kStave]; }

  Int_t getNumberOfHalfStavesPerParent() const { return mHierarchy[kHalfStave]; }

  Int_t getNumberOfModulesPerParent() const { return mHierarchy[kModule]; }

  Int_t getNumberOfChipsPerParent() const { return mHierarchy[kChip]; }

  Int_t getBuildLevel() const { return mBuildLevel; }

  Detector::Model getStaveModel() const { return mStaveModel; }

  void setChipThick(Double_t t) { mChipThickness = t; };

  /// Gets the Gamma Conversion Rod diameter
  Double_t getGammaConversionRodDiam();

  /// Gets the Gamma Conversion Rod X position
  Double_t getGammaConversionRodXPos();

  /// Sets the Stave tilt angle (for turbo layers only)
  /// \param t The stave tilt angle
  void setStaveTilt(Double_t t);

  /// Sets the Stave width (for turbo layers only)
  /// \param w The stave width
  void setStaveWidth(Double_t w);

  void setSensorThick(Double_t t) { mSensorThickness = t; };

  void setNumberOfStaves(Int_t n) { mHierarchy[kStave] = mNumberOfStaves = n; };

  /// Sets the number of units in a stave:
  ///      for the Inner Barrel: the number of chips per stave
  ///      for the Outer Barrel: the number of modules per half stave
  /// \param u the number of units
  void setNumberOfUnits(Int_t u);

  void setRadius(Double_t r) { mLayerRadius = r; };

  void setPhi0(Double_t phi) { mPhi0 = phi; }

  void setChipType(Int_t tp) { mChipTypeID = tp; }

  void setBuildLevel(Int_t buildLevel) { mBuildLevel = buildLevel; }

  void setStaveModel(o2::its::Detector::Model model) { mStaveModel = model; }

  /// Adds the Gamma Conversion Rods to the geometry
  /// \param diam the diameter of each rod
  /// \param xPos the X position of each rod
  void addGammaConversionRods(const Double_t diam, const Double_t xPos);

  /// Creates the actual Layer and places inside its mother volume
  /// \param motherVolume the TGeoVolume owing the volume structure
  virtual void createLayer(TGeoVolume* motherVolume);

 private:
  /// Creates the actual Layer and places inside its mother volume
  /// A so-called "turbo" layer is a layer where staves overlap in phi
  /// User can set width and tilt angle, no check is performed here
  /// to avoid volume overlaps
  /// \param motherVolume The TGeoVolume owing the volume structure
  void createLayerTurbo(TGeoVolume* motherVolume);

  /// Computes the inner radius of the air container for the Turbo configuration
  /// as the radius of either the circle tangent to the stave or the circle
  /// passing for the stave's lower vertex. Returns the radius of the container
  /// if >0, else flag to use the lower vertex
  Double_t radiusOmTurboContainer();

  /// Creates the actual Stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStave(const TGeoManager* mgr = gGeoManager);

  /// Creates the IB Module: (only the chips for the time being)
  /// Returns the module as a TGeoVolume
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createModuleInnerB(const TGeoManager* mgr = gGeoManager);

  /// Creates the OB Module: HIC + FPC + Carbon plate
  /// Returns the module as a TGeoVolume
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createModuleOuterB(const TGeoManager* mgr = gGeoManager);

  /// Creates the IB FPC Aluminum Ground layer
  /// Returns the layer as a TGeoVolume
  /// \param x, z X, Z layer half lengths
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createIBFPCAlGnd(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Creates the IB FPC Aluminum Anode layer
  /// Returns the layer as a TGeoVolume
  /// \param x, z X, Z layer half lengths
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createIBFPCAlAnode(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Creates the IB FPC capacitors
  /// \param modvol The IB module mother volume
  /// \param zchip The chip half Z length
  /// \param yzero The Y base position of capacitors
  /// \param mgr The GeoManager (used only to get the proper material)
  void createIBCapacitors(TGeoVolume* modvol, Double_t zchip, Double_t yzero, const TGeoManager* mgr = gGeoManager);

  /// Create the chip stave for the Inner Barrel(Here we fake the halfstave volume to have the
  /// same formal geometry hierarchy as for the Outer Barrel)
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveInnerB(const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveStructInnerB(const TGeoManager* mgr = gGeoManager);

  /// Create a dummy stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelInnerBDummy(const TGeoManager* mgr = gGeoManager) const;

  /// Create the mechanical stave structure for Model 4 of TDR
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelInnerB4(const TGeoManager* mgr = gGeoManager);

  /// Create the Inner Barrel End Stave connectors
  /// \param mgr The GeoManager (used only to get the proper material)
  void createIBConnectors(const TGeoManager* mgr = gGeoManager);

  /// Create the Inner Barrel End Stave connectors on Side A
  /// \param mgr The GeoManager (used only to get the proper material)
  void createIBConnectorsASide(const TGeoManager* mgr = gGeoManager);

  /// Create the Inner Barrel End Stave connectors on Side C
  /// \param mgr The GeoManager (used only to get the proper material)
  void createIBConnectorsCSide(const TGeoManager* mgr = gGeoManager);

  /// Creates the OB FPC Copper Ground layer
  /// Returns the FPC as a TGeoVolume
  /// \param z Z module half lengths
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createOBFPCCuGnd(Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Creates the OB FPC Copper Signal layer
  /// Returns the FPC as a TGeoVolume
  /// \param z Z module half lengths
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createOBFPCCuSig(Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Creates the OB Power and Bias Buses
  /// Returns a TGeoVolume with both buses
  /// \param z Z stave half lengths
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createOBPowerBiasBuses(Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create the chip stave for the Outer Barrel
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveOuterB(const TGeoManager* mgr = gGeoManager);

  /// Create dummy stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelOuterBDummy(const TGeoManager* mgr = gGeoManager) const;

  /// Create the mechanical half stave structure or the Outer Barrel as in TDR
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelOuterB2(const TGeoManager* mgr = gGeoManager);

  /// Create the Cold Plate connectors
  void createOBColdPlateConnectors();

  /// Create the A-Side end-stave connectors for OB staves
  void createOBColdPlateConnectorsASide();

  /// Create the C-Side end-stave connectors for OB staves
  void createOBColdPlateConnectorsCSide();

  /// Create the space frame for the Outer Barrel
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createSpaceFrameOuterB(const TGeoManager* mgr = gGeoManager);

  /// Create dummy stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createSpaceFrameOuterBDummy(const TGeoManager* mgr = gGeoManager) const;

  /// Create the space frame for the Outer Barrel (Model 2)
  /// Returns a TGeoVolume with the Space Frame of a stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createSpaceFrameOuterB2(const TGeoManager* mgr = gGeoManager);

  /// Create the space frame building blocks for the Outer Barrel
  /// \param mgr The GeoManager (used only to get the proper material)
  void createOBSpaceFrameObjects(const TGeoManager* mgr = gGeoManager);

  /// Creates the V-shaped sides of the OB space frame (from a similar method with same
  /// name and function in V11GeometrySDD class by L.Gaudichet)
  /// \param name The volume name
  /// \param dz The half Z length
  /// \param alpha The first rotation angle
  /// \param beta The second rotation angle
  /// \param L The stave length
  /// \param H The stave height
  /// \param top True to create the top corner, False to create the side one
  TGeoXtru* createStaveSide(const char* name, Double_t dz, Double_t alpha, Double_t beta, Double_t L, Double_t H,
                            Bool_t top);

  /// Help method to create a TGeoCombiTrans matrix from a similar method with same name and
  /// function in V11GeometrySDD class by L.Gaudichet)
  /// Returns the TGeoCombiTrans which make a translation in y and z and a rotation in phi
  /// in the global coord system. If planeSym = true, the rotation places the object
  /// symetrically (with respect to the transverse plane) to its position in the
  /// case planeSym = false
  TGeoCombiTrans* createCombiTrans(const char* name, Double_t dy, Double_t dz, Double_t dphi, Bool_t planeSym = kFALSE);

  /// Help method to add a translation to a TGeoCombiTrans matrix (from a similar method
  /// with same name and function in V11GeometrySDD class by L.Gaudichet)
  void addTranslationToCombiTrans(TGeoCombiTrans* ct, Double_t dx = 0, Double_t dy = 0, Double_t dz = 0) const;

  Int_t mLayerNumber;        ///< Current layer number
  Double_t mPhi0;            ///< lab phi of 1st stave, in degrees!!!
  Double_t mLayerRadius;     ///< Inner radius of this layer
  Double_t mSensorThickness; ///< Sensor thickness
  Double_t mChipThickness;   ///< Chip thickness
  Double_t mStaveWidth;      ///< Stave width (for turbo layers only)
  Double_t mStaveTilt;       ///< Stave tilt angle (for turbo layers only) in degrees
  Int_t mNumberOfStaves;     ///< Number of staves in this layer
  Int_t mNumberOfModules;    ///< Number of modules per container if defined (HalfStave, Stave, whatever is
  ///< container)
  Int_t mNumberOfChips; ///< Number chips per container (module, HalfStave, Stave, whatever is
  /// container)
  Int_t mHierarchy[kNHLevels]; ///< array to query number of staves, hstaves, modules, chips per its parent volume

  UInt_t mChipTypeID; ///< detector type id
  Bool_t mIsTurbo;    ///< True if this layer is a "turbo" layer
  Int_t mBuildLevel;  ///< Used for material studies

  Detector::Model mStaveModel; ///< The stave model

  Bool_t mAddGammaConv;    ///< True to add Gamma Conversion Rods
  Double_t mGammaConvDiam; ///< Gamma Conversion Rod Diameter
  Double_t mGammaConvXPos; ///< Gamma Conversion Rod X Position

  // Dimensions computed during geometry build-up
  Double_t mIBModuleZLength; ///< IB Module Length along Z
  Double_t mOBModuleZLength; ///< OB Module Length along Z

  // Parameters for the  geometry

  // General Parameters
  static const Int_t sNumberOfInnerLayers; ///< Number of IB Layers

  static const Double_t sDefaultSensorThick; ///< Default sensor thickness
  static const Double_t sMetalLayerThick;    ///< Metal layer thickness

  // Inner Barrel Parameters
  static const Int_t sIBChipsPerRow; ///< IB chips per row in module
  static const Int_t sIBNChipRows;   ///< IB chip rows in module
  static const Double_t sIBChipZGap; ///< Gap between IB chips on Z

  static const Double_t sIBModuleZLength;      ///< IB Module Length along Z
  static const Double_t sIBFPCWiderXPlus;      ///< FPC protrusion at X>0
  static const Double_t sIBFPCWiderXNeg;       ///< FPC protrusion at X<0
  static const Double_t sIBFlexCableAlThick;   ///< Thickness of FPC Aluminum
  static const Double_t sIBFPCAlGNDWidth;      ///< Width of total FPC Al Gnd
  static const Double_t sIBFPCAlAnodeWidth1;   ///< Width of FPC Al Anode
  static const Double_t sIBFPCAlAnodeWidth2;   ///< Width of FPC Al Anode
  static const Double_t sIBFlexCableKapThick;  ///< Thickness of FPC Kapton
  static const Double_t sIBFlexCablePolyThick; ///< Thickness of FPC Coverlay
  static const Double_t sIBFlexCapacitorXWid;  ///< IB capaictor X width
  static const Double_t sIBFlexCapacitorYHi;   ///< IB capaictor Y height
  static const Double_t sIBFlexCapacitorZLen;  ///< IB capaictor Z length
  static const Double_t sIBColdPlateWidth;     ///< IB cold plate X width
  static const Double_t sIBColdPlateZLen;      ///< IB cold plate Z length
  static const Double_t sIBGlueThick;          ///< IB glue thickness
  static const Double_t sIBCarbonFleeceThick;  ///< IB carbon fleece thickness
  static const Double_t sIBCarbonPaperThick;   ///< IB Carbon Paper Thickness
  static const Double_t sIBCarbonPaperWidth;   ///< IB Carbon Paper X Width
  static const Double_t sIBCarbonPaperZLen;    ///< IB Carbon Paper Z Length
  static const Double_t sIBK13D2UThick;        ///< IB k13d2u prepreg thickness
  static const Double_t sIBCoolPipeInnerD;     ///< IB cooling inner diameter
  static const Double_t sIBCoolPipeThick;      ///< IB cooling pipe thickness
  static const Double_t sIBCoolPipeXDist;      ///< IB cooling pipe separation
  static const Double_t sIBCoolPipeZLen;       ///< IB cooling pipe length
  static const Double_t sIBTopVertexWidth1;    ///< IB TopVertex width
  static const Double_t sIBTopVertexWidth2;    ///< IB TopVertex width
  static const Double_t sIBTopVertexHeight;    ///< IB TopVertex height
  static const Double_t sIBTopVertexAngle;     ///< IB TopVertex aperture angle
  static const Double_t sIBSideVertexWidth;    ///< IB SideVertex width
  static const Double_t sIBSideVertexHeight;   ///< IB SideVertex height
  static const Double_t sIBTopFilamentSide;    ///< IB TopFilament side
  static const Double_t sIBTopFilamentAlpha;   ///< IB TopFilament angle
  static const Double_t sIBTopFilamentInterZ;  ///< IB TopFilament Z interdist
  static const Double_t sIBEndSupportThick;    ///< IB end support thickness
  static const Double_t sIBEndSupportZLen;     ///< IB end support length
  static const Double_t sIBEndSupportXUp;      ///< IB end support X up wide
  static const Double_t sIBEndSupportOpenPhi;  ///< IB end support opening phi

  static const Double_t sIBConnectorXWidth;    ///< IB Connectors Width
  static const Double_t sIBConnectorYTot;      ///< IB Connectors total height
  static const Double_t sIBConnectBlockZLen;   ///< IB Connector Block Z length
  static const Double_t sIBConnBodyYHeight;    ///< IB Connector Body Y height
  static const Double_t sIBConnTailYMid;       ///< IB Connector Tail Y mid pt
  static const Double_t sIBConnTailYShift;     ///< IB Connector Tail Y shift
  static const Double_t sIBConnTailZLen;       ///< IB Connector Tail Z length
  static const Double_t sIBConnTailOpenPhi;    ///< IB Connector Tail Angle
  static const Double_t sIBConnRoundHoleD;     ///< IB Connector Hole diameter
  static const Double_t sIBConnRoundHoleZ;     ///< IB Connector Hole Z pos
  static const Double_t sIBConnSquareHoleX;    ///< IB Connector Hole X len
  static const Double_t sIBConnSquareHoleZ;    ///< IB Connector Hole Z len
  static const Double_t sIBConnSquareHoleZPos; ///< IB Connector Hole Z pos
  static const Double_t sIBConnInsertHoleD;    ///< IB Connector Insert diam
  static const Double_t sIBConnInsertHoleZPos; ///< IB Connector Insert Z pos
  static const Double_t sIBConnTubeHole1D;     ///< IB Connector Tube1 diam
  static const Double_t sIBConnTubeHole1ZLen;  ///< IB Connector Tube1 Z len
  static const Double_t sIBConnTubeHole1ZLen2; ///< IB Conn Tube1 Z len 2'side
  static const Double_t sIBConnTubeHole2D;     ///< IB Connector Tube2 diam
  static const Double_t sIBConnTubeHole3XPos;  ///< IB Connector Tube3 X pos
  static const Double_t sIBConnTubeHole3ZPos;  ///< IB Connector Tube3 Z pos
  static const Double_t sIBConnTubesXDist;     ///< IB Connector Tubes X dist
  static const Double_t sIBConnTubesYPos;      ///< IB Connector Tubes Y pos
  static const Double_t sIBConnInsertD;        ///< IB Connector Insert diam
  static const Double_t sIBConnInsertHeight;   ///< IB Connector Insert height
  static const Double_t sIBConnSideHole1D;     ///< IB Conn Side 1st hole D
  static const Double_t sIBConnSideHole1YPos;  ///< IB Conn Side 1st hole Y pos
  static const Double_t sIBConnSideHole1ZPos;  ///< IB Conn Side 1st hole Z pos
  static const Double_t sIBConnSideHole1XWid;  ///< IB Conn Side 1st hole X wid
  static const Double_t sIBConnSideHole2YPos;  ///< IB Conn Side 2nd hole Y pos
  static const Double_t sIBConnSideHole2ZPos;  ///< IB Conn Side 2nd hole Z pos
  static const Double_t sIBConnSideHole2XWid;  ///< IB Conn Side 2nd hole X wid
  static const Double_t sIBConnSideHole2YWid;  ///< IB Conn Side 2nd hole Y wid
  static const Double_t sIBConnSideHole2ZWid;  ///< IB Conn Side 2nd hole Z wid
  static const Double_t sIBConnectAFitExtD;    ///< IB ConnectorA Fitting ext D
  static const Double_t sIBConnectAFitIntD;    ///< IB ConnectorA Fitting int D
  static const Double_t sIBConnectAFitZLen;    ///< IB ConnectorA Fitting Z len
  static const Double_t sIBConnectAFitZOut;    ///< IB ConnectorA Fitting Z Out
  static const Double_t sIBConnPlugInnerD;     ///< IB Connector Plug int diam
  static const Double_t sIBConnPlugTotLen;     ///< IB Connector Plug tot len
  static const Double_t sIBConnPlugInnerLen;   ///< IB Connector Plug int len

  static const Double_t sIBStaveHeight; ///< IB Stave Total Y Height

  // Outer Barrel Parameters
  static const Int_t sOBChipsPerRow; ///< OB chips per row in module
  static const Int_t sOBNChipRows;   ///< OB chip rows in module

  static const Double_t sOBChipThickness; ///< Default OB chip thickness

  static const Double_t sOBHalfStaveWidth;    ///< OB Half Stave Width
  static const Double_t sOBModuleGap;         ///< Gap between OB modules
  static const Double_t sOBChipXGap;          ///< Gap between OB chips on X
  static const Double_t sOBChipZGap;          ///< Gap between OB chips on Z
  static const Double_t sOBFlexCableXWidth;   ///< OB FPC X width
  static const Double_t sOBFlexCableAlThick;  ///< Thickness of FPC Aluminum
  static const Double_t sOBFlexCableKapThick; ///< Thickness of FPC Kapton
  static const Double_t sOBFPCSoldMaskThick;  ///< Thickness of FPC Solder Mask
  static const Double_t sOBFPCCopperThick;    ///< Thickness of FPC Copper
  static const Double_t sOBFPCCuAreaFracGnd;  ///< Fraction of Cu on Gnd FPC
  static const Double_t sOBFPCCuAreaFracSig;  ///< Fraction of Cu on Sig FPC
  static const Double_t sOBGlueFPCThick;      ///< Thickness of Glue to FPC
  static const Double_t sOBGlueColdPlThick;   ///< Thickness of Glue to Cold Pl
  static const Double_t sOBPowerBusXWidth;    ///< OB Power Bus X width
  static const Double_t sOBPowerBusAlThick;   ///< OB Power Bus Al thickness
  static const Double_t sOBPowerBusAlFrac;    ///< Fraction of Al on OB PB
  static const Double_t sOBPowerBusDielThick; ///< OB Power Bus Dielectric thick
  static const Double_t sOBPowerBusKapThick;  ///< OB Power Bus Kapton thick
  static const Double_t sOBBiasBusXWidth;     ///< OB Bias Bus X width
  static const Double_t sOBBiasBusAlThick;    ///< OB Bias Bus Al thickness
  static const Double_t sOBBiasBusAlFrac;     ///< Fraction of Al on OB BB
  static const Double_t sOBBiasBusDielThick;  ///< OB Bias Bus Dielectric thick
  static const Double_t sOBBiasBusKapThick;   ///< OB Bias Bus Kapton thick
  static const Double_t sOBColdPlateXWidth;   ///< OB Cold Plate X width
  static const Double_t sOBColdPlateZLenML;   ///< OB ML Cold Plate Z length
  static const Double_t sOBColdPlateZLenOL;   ///< OB OL Cold Plate Z length
  static const Double_t sOBColdPlateThick;    ///< OB Cold Plate Thickness
  static const Double_t sOBHalfStaveYPos;     ///< OB half staves Y position
  static const Double_t sOBHalfStaveYTrans;   ///< OB half staves Y transl.
  static const Double_t sOBHalfStaveXOverlap; ///< OB half staves X overlap
  static const Double_t sOBGraphiteFoilThick; ///< OB graphite foil thickness
  static const Double_t sOBCarbonFleeceThick; ///< OB carbon fleece thickness
  static const Double_t sOBCoolTubeInnerD;    ///< OB cooling inner diameter
  static const Double_t sOBCoolTubeThick;     ///< OB cooling tube thickness
  static const Double_t sOBCoolTubeXDist;     ///< OB cooling tube separation

  static const Double_t sOBCPConnectorXWidth; ///< OB Cold Plate Connect Width
  static const Double_t sOBCPConnBlockZLen;   ///< OB CP Connect Block Z len
  static const Double_t sOBCPConnBlockYHei;   ///< OB CP Connect Block Z len
  static const Double_t sOBCPConnHollowZLen;  ///< OB CP Connect Block Z len
  static const Double_t sOBCPConnHollowYHei;  ///< OB CP Connect Block Z len
  static const Double_t sOBCPConnSquareHoleX; ///< OB Conn Square Hole X len
  static const Double_t sOBCPConnSquareHoleZ; ///< OB Conn Square Hole Z len
  static const Double_t sOBCPConnSqrHoleZPos; ///< OB Conn Square Hole Z pos
  static const Double_t sOBCPConnSqrInsertRZ; ///< OB Conn Square Insert RZpos
  static const Double_t sOBCPConnRoundHoleD;  ///< OB Conn Round Hole diam
  static const Double_t sOBCPConnRndHoleZPos; ///< OB Conn Round Hole Z pos
  static const Double_t sOBCPConnTubesXDist;  ///< OB Connector Tubes X dist
  static const Double_t sOBCPConnTubesYPos;   ///< OB Connector Tubes Y pos
  static const Double_t sOBCPConnTubeHole1D;  ///< OB Connector Tube1 diam
  static const Double_t sOBCPConnTubeHole1Z;  ///< OB Connector Tube1 Z len
  static const Double_t sOBCPConnTubeHole2D;  ///< OB Connector Tube2 diam
  static const Double_t sOBCPConnFitHoleD;    ///< OB Connector Fit Hole diam
  static const Double_t sOBCPConnTubeHole3XP; ///< OB Connector Tube3 X pos
  static const Double_t sOBCPConnTubeHole3ZP; ///< OB Connector Tube3 Z pos
  static const Double_t sOBCPConnInstZThick;  ///< OB Connector Insert height
  static const Double_t sOBCPConnInsertYHei;  ///< OB Connector Insert height
  static const Double_t sOBCPConnAFitExtD;    ///< OB ConnectorA Fitting ext D
  static const Double_t sOBCPConnAFitThick;   ///< OB ConnectorA Fitting thick
  static const Double_t sOBCPConnAFitZLen;    ///< OB ConnectorA Fitting Z len
  static const Double_t sOBCPConnAFitZIn;     ///< OB ConnectorA Fitting Z ins
  static const Double_t sOBCPConnPlugInnerD;  ///< OB Connector Plug int diam
  static const Double_t sOBCPConnPlugTotLen;  ///< OB Connector Plug tot le
  static const Double_t sOBCPConnPlugThick;   ///< OB Connector Plug thickness

  static const Double_t sOBSpaceFrameZLen[2]; ///< OB Space Frame Length
  static const Int_t sOBSpaceFrameNUnits[2];  ///< OB Number of SF Units
  static const Double_t sOBSpaceFrameUnitLen; ///< OB Space Frame Unit length
  static const Double_t sOBSpaceFrameWidth;   ///< OB Space Frame Width
  static const Double_t sOBSpaceFrameHeight;  ///< OB Space Frame Height
  static const Double_t sOBSpaceFrameTopVL;   ///< Parameters defining...
  static const Double_t sOBSpaceFrameTopVH;   ///< ...the Top V shape
  static const Double_t sOBSpaceFrameSideVL;  ///< Parameters defining...
  static const Double_t sOBSpaceFrameSideVH;  ///< ...the Side V shape
  static const Double_t sOBSpaceFrameVAlpha;  ///< Angles of aperture...
  static const Double_t sOBSpaceFrameVBeta;   ///< ...of the V shapes
  static const Double_t sOBSFrameBaseRibDiam; ///< OB SFrame Base Rib Diam
  static const Double_t sOBSFrameBaseRibPhi;  ///< OB SF base beam angle
  static const Double_t sOBSFrameSideRibDiam; ///< OB SFrame Side Rib Diam
  static const Double_t sOBSFrameSideRibPhi;  ///< OB SF side beam angle
  static const Double_t sOBSFrameULegLen;     ///< OB SF U-Leg length
  static const Double_t sOBSFrameULegWidth;   ///< OB SF U-Leg width
  static const Double_t sOBSFrameULegHeight1; ///< OB SF U-Leg height
  static const Double_t sOBSFrameULegHeight2; ///< OB SF U-Leg height
  static const Double_t sOBSFrameULegThick;   ///< OB SF U-Leg thickness
  static const Double_t sOBSFrameULegXPos;    ///< OB SF U-Leg X position

  ClassDefOverride(V3Layer, 0) // ITS v3 geometry
};
}
}

#endif
