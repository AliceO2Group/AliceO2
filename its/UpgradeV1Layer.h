/// \file AliITSUv1Layer.cxx
/// \brief Definition of the AliITSUv1Layer class
/// \author Mario Sitta <sitta@to.infn.it>
/// \author Chinorat Kobdaj (kobdaj@g.sut.ac.th)

#ifndef ALICEO2_ITS_UPGRADEV1LAYER_H_
#define ALICEO2_ITS_UPGRADEV1LAYER_H_

#include "V11Geometry.h"
#include "Detector.h"
#include <TGeoManager.h>
#include <TGeoCompositeShape.h>
#include <TGeoXtru.h>

class TGeoVolume;

namespace AliceO2 {
namespace ITS {

/// This class defines the Geometry for the ITS Upgrade using TGeo. This is a work class used
/// to study different configurations during the development of the new ITS structure
class UpgradeV1Layer : public V11Geometry {

public:
  enum { kStave, kHalfStave, kModule, kChip, kNHLevels };

  // Default constructor
  UpgradeV1Layer();

  // Constructor setting debugging level
  UpgradeV1Layer(Int_t debug);

  // Constructor setting layer number and debugging level
  UpgradeV1Layer(Int_t lay, Int_t debug);

  /// Constructor setting layer number and debugging level
  /// for a "turbo" layer (i.e. where staves overlap in phi)
  UpgradeV1Layer(Int_t lay, Bool_t turbo, Int_t debug);

  /// Copy constructor
  UpgradeV1Layer(const UpgradeV1Layer& source);

  /// Assignment operator
  UpgradeV1Layer& operator=(const UpgradeV1Layer& source);

  /// Default destructor
  virtual ~UpgradeV1Layer();

  Bool_t IsTurbo() const
  {
    return mIsTurbo;
  };

  Double_t GetStaveThick() const
  {
    return mStaveThickness;
  };
  Double_t GetStaveTilt() const
  {
    return mStaveTilt;
  };
  Double_t GetStaveWidth() const
  {
    return mStaveWidth;
  };
  Double_t GetSensorThick() const
  {
    return mSensorThickness;
  };
  Double_t GetNStaves() const
  {
    return mNumberOfStaves;
  };
  Double_t GetNChips() const
  {
    return mNumberOfChips;
  };
  Double_t GetRadius() const
  {
    return mLayerRadius;
  };
  Double_t GetPhi0() const
  {
    return mPhi0;
  };
  Double_t GetZLength() const
  {
    return mZLength;
  };
  Int_t GetChipType() const
  {
    return mChipTypeID;
  }

  Int_t GetNStavesPerParent() const
  {
    return mHierarchy[kStave];
  }
  Int_t GetNHalfStavesPerParent() const
  {
    return mHierarchy[kHalfStave];
  }
  Int_t GetNModulesPerParent() const
  {
    return mHierarchy[kModule];
  }
  Int_t GetNChipsPerParent() const
  {
    return mHierarchy[kChip];
  }

  AliceO2::ITS::Detector::UpgradeModel GetStaveModel() const
  {
    return mStaveModel;
  }

  void SetStaveThick(Double_t t)
  {
    mStaveThickness = t;
  };

  /// Sets the Stave tilt angle (for turbo layers only)
  /// \param t The stave tilt angle
  void SetStaveTilt(Double_t t);

  /// Sets the Stave width (for turbo layers only)
  /// \param w The stave width
  void SetStaveWidth(Double_t w);
  void SetSensorThick(Double_t t)
  {
    mSensorThickness = t;
  };
  void SetNumberOfStaves(Int_t n)
  {
    mHierarchy[kStave] = mNumberOfStaves = n;
  };

  /// Sets the number of units in a stave:
  ///      for the Inner Barrel: the number of chips per stave
  ///      for the Outer Barrel: the number of modules per half stave
  /// \param u the number of units
  void SetNumberOfUnits(Int_t u);

  void SetRadius(Double_t r)
  {
    mLayerRadius = r;
  };
  void SetPhi0(Double_t phi)
  {
    mPhi0 = phi;
  }
  void SetZLength(Double_t z)
  {
    mZLength = z;
  };
  void SetChipType(Int_t tp)
  {
    mChipTypeID = tp;
  }
  void SetBuildLevel(Int_t buildLevel)
  {
    mBuildLevel = buildLevel;
  }
  void SetStaveModel(AliceO2::ITS::Detector::UpgradeModel model)
  {
    mStaveModel = model;
  }

  /// Creates the actual Layer and places inside its mother volume
  /// \param motherVolume the TGeoVolume owing the volume structure
  virtual void CreateLayer(TGeoVolume* motherVolume);

private:
  /// Creates the actual Layer and places inside its mother volume
  /// A so-called "turbo" layer is a layer where staves overlap in phi
  /// User can set width and tilt angle, no check is performed here
  /// to avoid volume overlaps
  /// \param motherVolume The TGeoVolume owing the volume structure
  void CreateLayerTurbo(TGeoVolume* motherVolume);

  /// Computes the inner radius of the air container for the Turbo configuration
  /// as the radius of either the circle tangent to the stave or the circle
  /// passing for the stave's lower vertex. Returns the radius of the container
  /// if >0, else flag to use the lower vertex
  Double_t RadiusOmTurboContainer();

  /// Creates the actual Stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStave(const TGeoManager* mgr = gGeoManager);

  // TGeoVolume* CreateChip(Double_t x, Double_t z, const TGeoManager *mgr=gGeoManager);

  /// Creates the IB Module: (only the chips for the time being)
  /// Returns the module as a TGeoVolume
  /// \param xmod, ymod, zmod X, Y, Z module half lengths
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateModuleInnerB(Double_t x, Double_t y, Double_t z,
                                 const TGeoManager* mgr = gGeoManager);

  /// Creates the actual Chip
  /// \param xchip,ychip,zchip The chip dimensions
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateChipInnerB(Double_t x, Double_t y, Double_t z,
                               const TGeoManager* mgr = gGeoManager);

  /// Creates the OB Module: HIC + FPC + Carbon plate
  /// Returns the module as a TGeoVolume
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateModuleOuterB(const TGeoManager* mgr = gGeoManager);

  /// Create the chip stave for the Inner Barrel(Here we fake the halfstave volume to have the
  /// same formal geometry hierarchy as for the Outer Barrel)
  /// \param xsta, ysta, zsta X, Y, Z stave half lengths
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveInnerB(Double_t x, Double_t y, Double_t z,
                                const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr  The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveStructInnerB(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create a dummy stave
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveModelInnerBDummy(Double_t x, Double_t z,
                                          const TGeoManager* mgr = gGeoManager) const;

  /// Create the mechanical stave structure for Model 0 of TDR
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveModelInnerB0(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure for Model 1 of TDR
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveModelInnerB1(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure for Model 2.1 of TDR
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveModelInnerB21(Double_t x, Double_t z,
                                       const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure for Model 2.2 of TDR
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveModelInnerB22(Double_t x, Double_t z,
                                       const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical stave structure for Model 3 of TDR
  /// \param xsta X length
  /// \param zsta Z length
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveModelInnerB3(Double_t x, Double_t z, const TGeoManager* mgr = gGeoManager);

  /// Create the chip stave for the Outer Barrel
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveOuterB(const TGeoManager* mgr = gGeoManager);

  /// Create dummy stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveModelOuterBDummy(const TGeoManager* mgr = gGeoManager) const;

  /// Creation of the mechanical stave structure for the Outer Barrel as in v0
  /// (we fake the module and halfstave volumes to have always
  /// the same formal geometry hierarchy)
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveModelOuterB0(const TGeoManager* mgr = gGeoManager);

  /// Create the mechanical half stave structure or the Outer Barrel as in TDR
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateStaveModelOuterB1(const TGeoManager* mgr = gGeoManager);

  /// Create the space frame for the Outer Barrel
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateSpaceFrameOuterB(const TGeoManager* mgr = gGeoManager);

  /// Create dummy stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateSpaceFrameOuterBDummy(const TGeoManager* mgr = gGeoManager) const;

  /// Create the space frame for the Outer Barrel (Model 1)
  /// Returns a TGeoVolume with the Space Frame of a stave
  /// \param mgr The GeoManager (used only to get the proper material)
  TGeoVolume* CreateSpaceFrameOuterB1(const TGeoManager* mgr = gGeoManager);

  /// Creates the V-shaped sides of the OB space frame (from a similar method with same
  /// name and function in V11GeometrySDD class by L.Gaudichet)
  TGeoArb8* CreateStaveSide(const char* name, Double_t dz, Double_t angle, Double_t xSign,
                            Double_t L, Double_t H, Double_t l);

  /// Help method to create a TGeoCombiTrans matrix from a similar method with same name and
  /// function in V11GeometrySDD class by L.Gaudichet)
  /// Returns the TGeoCombiTrans which make a translation in y and z and a rotation in phi
  /// in the global coord system. If planeSym = true, the rotation places the object
  /// symetrically (with respect to the transverse plane) to its position in the
  /// case planeSym = false
  TGeoCombiTrans* CreateCombiTrans(const char* name, Double_t dy, Double_t dz, Double_t dphi,
                                   Bool_t planeSym = kFALSE);

  /// Help method to add a translation to a TGeoCombiTrans matrix (from a similar method
  /// with same name and function in V11GeometrySDD class by L.Gaudichet)
  void AddTranslationToCombiTrans(TGeoCombiTrans* ct, Double_t dx = 0, Double_t dy = 0,
                                  Double_t dz = 0) const;

  Int_t mLayerNumber;        ///< Current layer number
  Double_t mPhi0;            ///< lab phi of 1st stave, in degrees!!!
  Double_t mLayerRadius;     ///< Inner radius of this layer
  Double_t mZLength;         ///< Z length of this layer
  Double_t mSensorThickness; ///< Sensor thickness
  Double_t mStaveThickness;  ///< Stave thickness
  Double_t mStaveWidth;      ///< Stave width (for turbo layers only)
  Double_t mStaveTilt;       ///< Stave tilt angle (for turbo layers only) in degrees
  Int_t mNumberOfStaves;     ///< Number of staves in this layer
  Int_t
    mNumberOfModules; ///< Number of modules per container if defined (HalfStave, Stave, whatever is
                      ///< container)
  Int_t mNumberOfChips; ///< Number chips per container (module, HalfStave, Stave, whatever is
  /// container)
  Int_t mHierarchy
    [kNHLevels]; ///< array to query number of staves, hstaves, modules, chips per its parent volume

  UInt_t mChipTypeID; ///< detector type id
  Bool_t mIsTurbo;    ///< True if this layer is a "turbo" layer
  Int_t mBuildLevel;  ///< Used for material studies

  AliceO2::ITS::Detector::UpgradeModel mStaveModel; ///< The stave model

  // Parameters for the Upgrade geometry

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

  ClassDef(UpgradeV1Layer, 0) // ITS Upgrade v1 geometry
};
}
}

#endif
