// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliITSUv1Layer.cxx
/// \brief Definition of the AliITSUv1Layer class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Chinorat Kobdaj (kobdaj@g.sut.ac.th)

#ifndef ALICEO2_ITS_UPGRADEV1LAYER_H_
#define ALICEO2_ITS_UPGRADEV1LAYER_H_

#include <TGeoManager.h>               // for gGeoManager
#include "Rtypes.h"                    // for Double_t, Int_t, Bool_t, etc
#include "ITSSimulation/V11Geometry.h" // for V11Geometry
#include "ITSSimulation/Detector.h"    // for Detector, Detector::Model

class TGeoArb8;

class TGeoCombiTrans;

class TGeoVolume; // lines 15-15

namespace o2
{
namespace its
{

/// This class defines the Geometry for the ITS  using TGeo. This is a work class used
/// to study different configurations during the development of the new ITS structure
class V1Layer : public V11Geometry
{

 public:
  enum {
    kStave,
    kHalfStave,
    kModule,
    kChip,
    kNHLevels
  };

  // Default constructor
  V1Layer();

  // Constructor setting debugging level
  V1Layer(Int_t debug);

  // Constructor setting layer number and debugging level
  V1Layer(Int_t lay, Int_t debug);

  /// Constructor setting layer number and debugging level
  /// for a "turbo" layer (i.e. where staves overlap in phi)
  V1Layer(Int_t lay, Bool_t turbo, Int_t debug);

  /// Copy constructor
  V1Layer(const V1Layer& source);

  /// Assignment operator
  V1Layer& operator=(const V1Layer& source);

  /// Default destructor
  ~V1Layer() override;

  Bool_t isTurbo() const
  {
    return mIsTurbo;
  };

  Double_t getStaveThick() const
  {
    return mStaveThickness;
  };

  Double_t getStaveTilt() const
  {
    return mStaveTilt;
  };

  Double_t getStaveWidth() const
  {
    return mStaveWidth;
  };

  Double_t getSensorThick() const
  {
    return mSensorThickness;
  };

  Double_t getNumberOfStaves() const
  {
    return mNumberOfStaves;
  };

  Double_t getNumberOfChips() const
  {
    return mNumberOfChips;
  };

  Double_t getRadius() const
  {
    return mLayerRadius;
  };

  Double_t getPhi0() const
  {
    return mPhi0;
  };

  Double_t getZLength() const
  {
    return mZLength;
  };

  Int_t getChipType() const
  {
    return mChipTypeID;
  }

  Int_t getNumberOfStavesPerParent() const
  {
    return mHierarchy[kStave];
  }

  Int_t getNumberOfHalfStavesPerParent() const
  {
    return mHierarchy[kHalfStave];
  }

  Int_t getNumberOfModulesPerParent() const
  {
    return mHierarchy[kModule];
  }

  Int_t getNumberOfChipsPerParent() const
  {
    return mHierarchy[kChip];
  }

  Detector::Model getStaveModel() const
  {
    return mStaveModel;
  }

  void setStaveThick(Double_t t)
  {
    mStaveThickness = t;
  };

  /// Sets the Stave tilt angle (for turbo layers only)
  /// \param t The stave tilt angle
  void setStaveTilt(Double_t t);

  /// Sets the Stave width (for turbo layers only)
  /// \param w The stave width
  void setStaveWidth(Double_t w);

  void setSensorThick(Double_t t)
  {
    mSensorThickness = t;
  };

  void setNumberOfStaves(Int_t n)
  {
    mHierarchy[kStave] = mNumberOfStaves = n;
  };

  /// Sets the number of units in a stave:
  ///      for the Inner Barrel: the number of chips per stave
  ///      for the Outer Barrel: the number of modules per half stave
  /// \param u the number of units
  void setNumberOfUnits(Int_t u);

  void setRadius(Double_t r)
  {
    mLayerRadius = r;
  };

  void setPhi0(Double_t phi)
  {
    mPhi0 = phi;
  }

  void setZLength(Double_t z)
  {
    mZLength = z;
  };

  void setChipType(Int_t tp)
  {
    mChipTypeID = tp;
  }

  void setBuildLevel(Int_t buildLevel)
  {
    mBuildLevel = buildLevel;
  }

  void setStaveModel(o2::its::Detector::Model model)
  {
    mStaveModel = model;
  }

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

  // TGeoVolume* createChip(Double_t x, Double_t z, const TGeoManager *mgr=gGeoManager);

  /// Creates the IB Module: (only the chips for the time being)
  /// Returns the module as a TGeoVolume
  /// \param xmod, ymod, zmod X, Y, Z module half lengths
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createModuleInnerB(Double_t x, Double_t y, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Creates the actual Chip
  /// \param xchip,ychip,zchip The chip dimensions
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createChipInnerB(Double_t x, Double_t y, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Creates the OB Module: HIC + FPC + Carbon plate
  /// Returns the module as a TGeoVolume
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createModuleOuterB(const TGeoManager* mgr = gGeoManager);

  /// Create the chip stave for the Inner Barrel(Here we fake the halfstave volume to have the
  /// same formal geometry hierarchy as for the Outer Barrel)
  /// \param xsta, ysta, zsta X, Y, Z stave half lengths
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveInnerB(Double_t x, Double_t y, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveStructInnerB(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create a dummy stave
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelInnerBDummy(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager) const;

  /// Create the mechanical stave structure for Model 0 of TDR
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelInnerB0(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure for Model 1 of TDR
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelInnerB1(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure for Model 2.1 of TDR
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelInnerB21(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure for Model 2.2 of TDR
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelInnerB22(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure for Model 3 of TDR
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelInnerB3(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create the chip stave for the Outer Barrel
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveOuterB(const TGeoManager* mgr = gGeoManager);

  /// Create dummy stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelOuterBDummy(const TGeoManager* mgr = gGeoManager) const;

  /// Creation of the mechanical stave structure for the Outer Barrel as in v0
  /// (we fake the module and halfstave volumes to have always
  /// the same formal geometry hierarchy)
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelOuterB0(const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical half stave structure or the Outer Barrel as in TDR
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createStaveModelOuterB1(const TGeoManager* mgr = gGeoManager);

  /// Create the space frame for the Outer Barrel
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createSpaceFrameOuterB(const TGeoManager* mgr = gGeoManager);

  /// Create dummy stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createSpaceFrameOuterBDummy(const TGeoManager* mgr = gGeoManager) const;

  /// Create the space frame for the Outer Barrel (Model 1)
  /// Returns a TGeoVolume with the Space Frame of a stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* createSpaceFrameOuterB1(const TGeoManager* mgr = gGeoManager);

  /// Creates the V-shaped sides of the OB space frame (from a similar method with same
  /// name and function in V11GeometrySDD class by L.Gaudichet)
  TGeoArb8* createStaveSide(const char* name, Double_t dz, Double_t angle, Double_t xSign, Double_t L, Double_t H,
                            Double_t l);

  /// Help method to create a TGeoCombiTrans matrix from a similar method with same name and
  /// function in V11GeometrySDD class by L.Gaudichet)
  /// Returns the TGeoCombiTrans which make a translation in y and z and a rotation in phi
  /// in the global coord system. If planeSym = true, the rotation places the object
  /// symetrically (with respect to the transverse plane) to its position in the
  /// case planeSym = false
  TGeoCombiTrans* createCombiTrans(const char* name, Double_t dy, Double_t dz, Double_t dphi,
                                   Bool_t planeSym = kFALSE);

  /// Help method to add a translation to a TGeoCombiTrans matrix (from a similar method
  /// with same name and function in V11GeometrySDD class by L.Gaudichet)
  void addTranslationToCombiTrans(TGeoCombiTrans* ct, Double_t dx = 0, Double_t dy = 0, Double_t dz = 0) const;

  Int_t mLayerNumber;        ///< Current layer number
  Double_t mPhi0;            ///< lab phi of 1st stave, in degrees!!!
  Double_t mLayerRadius;     ///< Inner radius of this layer
  Double_t mZLength;         ///< Z length of this layer
  Double_t mSensorThickness; ///< Sensor thickness
  Double_t mStaveThickness;  ///< Stave thickness
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

  // Parameters for the  geometry

  // General Parameters
  static const Int_t sNumberOmInnerLayers; ///< Number of IB Layers

  static const Double_t sDefaultSensorThick; ///< Default sensor thickness
  static const Double_t sDefaultStaveThick;  ///< Default stave thickness

  // Inner Barrel Parameters
  static const Int_t sIBChipsPerRow; ///< IB chips per row in module
  static const Int_t sIBNChipRows;   ///< IB chip rows in module

  // Outer Barrel Parameters
  static const Int_t sOBChipsPerRow; ///< OB chips per row in module
  static const Int_t sOBNChipRows;   ///< OB chip rows in module

  static const Double_t sOBHalfStaveWidth;    ///< OB Half Stave Width
  static const Double_t sOBModuleWidth;       ///< OB Module Width
  static const Double_t sOBModuleGap;         ///< Gap between OB modules
  static const Double_t sOBChipXGap;          ///< Gap between OB chips on X
  static const Double_t sOBChipZGap;          ///< Gap between OB chips on Z
  static const Double_t sOBFlexCableAlThick;  ///< Thickness of FPC Aluminum
  static const Double_t sOBFlexCableKapThick; ///< Thickness of FPC Kapton
  static const Double_t sOBBusCableAlThick;   ///< Thickness of Bus Aluminum
  static const Double_t sOBBusCableKapThick;  ///< Thickness of Bus Kapton
  static const Double_t sOBCarbonPlateThick;  ///< OB Carbon Plate Thickness
  static const Double_t sOBColdPlateThick;    ///< OB Cold Plate Thickness
  static const Double_t sOBGlueThick;         ///< OB Glue total Thickness
  static const Double_t sOBModuleZLength;     ///< OB Chip Length along Z
  static const Double_t sOBHalfStaveYTrans;   ///< OB half staves Y transl.
  static const Double_t sOBHalfStaveXOverlap; ///< OB half staves X overlap
  static const Double_t sOBGraphiteFoilThick; ///< OB graphite foil thickness
  static const Double_t sOBCoolTubeInnerD;    ///< OB cooling inner diameter
  static const Double_t sOBCoolTubeThick;     ///< OB cooling tube thickness
  static const Double_t sOBCoolTubeXDist;     ///< OB cooling tube separation

  static const Double_t sOBSpaceFrameWidth;   ///< OB Space Frame Width
  static const Double_t sOBSpaceFrameTotHigh; ///< OB Total Y Height
  static const Double_t sOBSFrameBeamRadius;  ///< OB Space Frame Beam Radius
  static const Double_t sOBSpaceFrameLa;      ///< Parameters defining...
  static const Double_t sOBSpaceFrameHa;      ///< ...the V side shape...
  static const Double_t sOBSpaceFrameLb;      ///< ...of the carbon...
  static const Double_t sOBSpaceFrameHb;      ///< ...OB Space Frame
  static const Double_t sOBSpaceFrameL;       ///< OB SF
  static const Double_t sOBSFBotBeamAngle;    ///< OB SF bottom beam angle
  static const Double_t sOBSFrameBeamSidePhi; ///< OB SF side beam angle

  ClassDefOverride(V1Layer, 0); // ITS v1 geometry
};
} // namespace its
} // namespace o2

#endif
