#ifndef ALICEO2_ITS_UPGRADEV1LAYER_H_
#define ALICEO2_ITS_UPGRADEV1LAYER_H_
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


//*************************************************************************
// This class Defines the Geometry for the ITS Upgrade using TGeo
// This is a work class used to study different configurations
// during the development of the new ITS structure.
//
//  Mario Sitta <sitta@to.infn.it>
//*************************************************************************

/*
  $Id: AliITSUv1Layer.h
 */

#include "V11Geometry.h"
#include "Detector.h"
#include <TGeoManager.h>
#include <TGeoCompositeShape.h>
#include <TGeoXtru.h>

class TGeoVolume;

namespace AliceO2 {
namespace ITS {

class UpgradeV1Layer : public V11Geometry {
  public:
  enum {kStave,kHalfStave,kModule,kChip,kNHLevels};

  public:
    UpgradeV1Layer();
    UpgradeV1Layer(Int_t debug);
    UpgradeV1Layer(Int_t lay, Int_t debug);
    UpgradeV1Layer(Int_t lay, Bool_t turbo, Int_t debug);
    UpgradeV1Layer(const UpgradeV1Layer &source);
    UpgradeV1Layer& operator=(const UpgradeV1Layer &source);
    virtual ~UpgradeV1Layer();

    Bool_t    IsTurbo() const {return fIsTurbo;};

    Double_t  GetStaveThick() const {return fStaveThick;};
    Double_t  GetStaveTilt()  const {return fStaveTilt;};
    Double_t  GetStaveWidth() const {return fStaveWidth;};
    Double_t  GetSensorThick() const {return fSensorThick;};
    Double_t  GetNStaves()    const {return fNStaves;};
    Double_t  GetNChips()      const {return fNChips;};
    Double_t  GetRadius()      const {return fLayRadius;};
    Double_t  GetPhi0()        const {return fPhi0;};
    Double_t  GetZLength()     const {return fZLength;};
    Int_t     GetChipType()    const {return fChipTypeID;}
    //
    Int_t     GetNStavesPerParent()     const {return fHierarchy[kStave];}
    Int_t     GetNHalfStavesPerParent() const {return fHierarchy[kHalfStave];}
    Int_t     GetNModulesPerParent()    const {return fHierarchy[kModule];}
    Int_t     GetNChipsPerParent()      const {return fHierarchy[kChip];}
    //
    AliceO2::ITS::Detector::AliITSUModel_t GetStaveModel() const {return fStaveModel;}
    //
    void      SetStaveThick(Double_t t)      {fStaveThick = t;};
    void      SetStaveTilt(Double_t t);
    void      SetStaveWidth(Double_t w);
    void      SetSensorThick(Double_t t)     {fSensorThick = t;};
    void      SetNStaves(Int_t n)            {fHierarchy[kStave] = fNStaves = n;};
    void      SetNUnits(Int_t u);
    void      SetRadius(Double_t r)          {fLayRadius = r;};
    void      SetPhi0(Double_t phi)          {fPhi0 = phi;}
    void      SetZLength(Double_t z)         {fZLength   = z;};
    void      SetChipType(Int_t tp)          {fChipTypeID = tp;}
    void      SetBuildLevel(Int_t buildLevel){fBuildLevel=buildLevel;}
    void      SetStaveModel(AliceO2::ITS::Detector::AliITSUModel_t model) {fStaveModel=model;}
    virtual void CreateLayer(TGeoVolume *moth);

  private:
    void CreateLayerTurbo(TGeoVolume *moth);

    Double_t RadiusOfTurboContainer();

    TGeoVolume* CreateStave(const TGeoManager *mgr=gGeoManager);
    //TGeoVolume* CreateChip(Double_t x, Double_t z, const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateModuleInnerB(Double_t x,Double_t y, Double_t z, const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateChipInnerB(Double_t x,Double_t y, Double_t z, const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateModuleOuterB(const TGeoManager *mgr=gGeoManager);


    TGeoVolume* CreateStaveInnerB(Double_t x, Double_t y, Double_t z, const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateStaveStructInnerB(Double_t x,Double_t z, const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateStaveModelInnerBDummy(Double_t x,Double_t z, const TGeoManager *mgr=gGeoManager) const;
    TGeoVolume* CreateStaveModelInnerB0(Double_t x,Double_t z, const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateStaveModelInnerB1(Double_t x,Double_t z, const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateStaveModelInnerB21(Double_t x,Double_t z, const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateStaveModelInnerB22(Double_t x,Double_t z, const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateStaveModelInnerB3(Double_t x,Double_t z, const TGeoManager *mgr=gGeoManager);

    TGeoVolume* CreateStaveOuterB(const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateStaveModelOuterBDummy(const TGeoManager *mgr=gGeoManager) const;
    TGeoVolume* CreateStaveModelOuterB0(const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateStaveModelOuterB1(const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateSpaceFrameOuterB(const TGeoManager *mgr=gGeoManager);
    TGeoVolume* CreateSpaceFrameOuterBDummy(const TGeoManager *mgr=gGeoManager) const;
    TGeoVolume* CreateSpaceFrameOuterB1(const TGeoManager *mgr=gGeoManager);

    TGeoArb8* CreateStaveSide(const char *name,
			       Double_t dz, Double_t angle, Double_t xSign,
			       Double_t L, Double_t H, Double_t l);
    TGeoCombiTrans* CreateCombiTrans( const char *name,
				      Double_t dy, Double_t dz, Double_t dphi,
				      Bool_t planeSym=kFALSE);
    void AddTranslationToCombiTrans( TGeoCombiTrans* ct,
				     Double_t dx=0, Double_t dy=0,
				     Double_t dz=0) const;


    Int_t     fLayerNumber; // Current layer number
    Double_t  fPhi0;        // lab phi of 1st stave, in degrees!!!
    Double_t  fLayRadius;   // Inner radius of this layer
    Double_t  fZLength;     // Z length of this layer
    Double_t  fSensorThick; // Sensor thickness
    Double_t  fStaveThick;  // Stave thickness
    Double_t  fStaveWidth;  // Stave width (for turbo layers only)
    Double_t  fStaveTilt;   // Stave tilt angle (for turbo layers only) in degrees
    Int_t     fNStaves;     // Number of staves in this layer
    Int_t     fNModules;    // Number of modules per container if defined (HalfStave, Stave, whatever is container)
    Int_t     fNChips;      // N. chips per container (module, HalfStave, Stave, whatever is container)
    Int_t     fHierarchy[kNHLevels]; // array to query number of staves, hstaves, modules, chips per its parent volume
    //    
    UInt_t    fChipTypeID;  // detector type id
    Bool_t    fIsTurbo;     // True if this layer is a "turbo" layer
    Int_t     fBuildLevel;  // Used for material studies

    AliceO2::ITS::Detector::AliITSUModel_t fStaveModel; // The stave model

    // Parameters for the Upgrade geometry

    // General Parameters
    static const Int_t    fgkNumberOfInnerLayers;// Number of IB Layers

    static const Double_t fgkDefaultSensorThick; // Default sensor thickness
    static const Double_t fgkDefaultStaveThick;  // Default stave thickness

    // Inner Barrel Parameters
    static const Int_t    fgkIBChipsPerRow;      // IB chips per row in module
    static const Int_t    fgkIBNChipRows;        // IB chip rows in module

    // Outer Barrel Parameters
    static const Int_t    fgkOBChipsPerRow;      // OB chips per row in module
    static const Int_t    fgkOBNChipRows;        // OB chip rows in module

    static const Double_t fgkOBHalfStaveWidth;   // OB Half Stave Width
    static const Double_t fgkOBModuleWidth;      // OB Module Width
    static const Double_t fgkOBModuleGap;        // Gap between OB modules
    static const Double_t fgkOBChipXGap;         // Gap between OB chips on X
    static const Double_t fgkOBChipZGap;         // Gap between OB chips on Z
    static const Double_t fgkOBFlexCableAlThick; // Thickness of FPC Aluminum
    static const Double_t fgkOBFlexCableKapThick;// Thickness of FPC Kapton
    static const Double_t fgkOBBusCableAlThick;  // Thickness of Bus Aluminum
    static const Double_t fgkOBBusCableKapThick; // Thickness of Bus Kapton
    static const Double_t fgkOBCarbonPlateThick; // OB Carbon Plate Thickness
    static const Double_t fgkOBColdPlateThick;   // OB Cold Plate Thickness
    static const Double_t fgkOBGlueThick;        // OB Glue total Thickness
    static const Double_t fgkOBModuleZLength;    // OB Chip Length along Z
    static const Double_t fgkOBHalfStaveYTrans;  // OB half staves Y transl.
    static const Double_t fgkOBHalfStaveXOverlap;// OB half staves X overlap
    static const Double_t fgkOBGraphiteFoilThick;// OB graphite foil thickness
    static const Double_t fgkOBCoolTubeInnerD;   // OB cooling inner diameter
    static const Double_t fgkOBCoolTubeThick;    // OB cooling tube thickness
    static const Double_t fgkOBCoolTubeXDist;    // OB cooling tube separation

    static const Double_t fgkOBSpaceFrameWidth;  // OB Space Frame Width
    static const Double_t fgkOBSpaceFrameTotHigh;// OB Total Y Height
    static const Double_t fgkOBSFrameBeamRadius; // OB Space Frame Beam Radius
    static const Double_t fgkOBSpaceFrameLa;     // Parameters defining...
    static const Double_t fgkOBSpaceFrameHa;     // ...the V side shape...
    static const Double_t fgkOBSpaceFrameLb;     // ...of the carbon...
    static const Double_t fgkOBSpaceFrameHb;     // ...OB Space Frame
    static const Double_t fgkOBSpaceFrameL;      // OB SF
    static const Double_t fgkOBSFBotBeamAngle;   // OB SF bottom beam angle
    static const Double_t fgkOBSFrameBeamSidePhi;// OB SF side beam angle


  ClassDef(UpgradeV1Layer,0) // ITS Upgrade v1 geometry
};
}
}

#endif
