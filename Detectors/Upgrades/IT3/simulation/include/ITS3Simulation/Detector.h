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

/// \file Detector.h
/// \brief Definition of the Detector class

#ifndef ALICEO2_ITS3_DETECTOR_H_
#define ALICEO2_ITS3_DETECTOR_H_

#include <vector>                             // for vector
#include "DetectorsBase/GeometryManager.h"    // for getSensID
#include "DetectorsBase/Detector.h"           // for Detector
#include "DetectorsCommonDataFormats/DetID.h" // for Detector
#include "ITSMFTSimulation/Hit.h"             // for Hit
#include "Rtypes.h"                           // for Int_t, Double_t, Float_t, Bool_t, etc
#include "TArrayD.h"                          // for TArrayD
#include "TGeoManager.h"                      // for gGeoManager, TGeoManager (ptr only)
#include "TLorentzVector.h"                   // for TLorentzVector
#include "TVector3.h"                         // for TVector3

class FairVolume;
class TGeoVolume;

class TParticle;

class TString;

namespace o2
{
namespace itsmft
{
class Hit;
}
} // namespace o2

namespace o2
{
namespace its3
{
class GeometryTGeo;
}
} // namespace o2
namespace o2
{
namespace its3
{
class V3Layer;
}
} // namespace o2

namespace o2
{
namespace its3
{
class V3Layer;
class V3Services;

class Detector : public o2::base::DetImpl<Detector>
{
 public:
  enum Model {
    kIBModelDummy = 0,
    kIBModel0 = 1,
    kIBModel1 = 2,
    kIBModel21 = 3,
    kIBModel22 = 4,
    kIBModel3 = 5,
    kIBModel4 = 10,
    kOBModelDummy = 6,
    kOBModel0 = 7,
    kOBModel1 = 8,
    kOBModel2 = 9
  };

  static constexpr Int_t sNumberInnerLayers = 4;                                  ///< Number of inner layers in ITS3
  static constexpr Int_t sNumberOuterLayers = 4;                                  ///< Number of outer layers in ITS3
  static constexpr Int_t sNumberLayers = sNumberInnerLayers + sNumberOuterLayers; ///< Total number of layers in ITS3
  static constexpr Int_t sNumberOfWrapperVolumes = 3;                             ///< Number of wrapper volumes

  /// Name : Detector Name
  /// Active: kTRUE for active detectors (ProcessHits() will be called)
  ///         kFALSE for inactive detectors
  Detector(Bool_t active);

  /// Special version for ITS3: add number on Inner Layers
  Detector(Bool_t active, Int_t nlay);

  /// Default constructor
  Detector();

  /// Default destructor
  ~Detector() override;

  /// Initialization of the detector is done here
  void InitializeO2Detector() override;

  /// This method is called for each step during simulation (see FairMCApplication::Stepping())
  Bool_t ProcessHits(FairVolume* v = nullptr) override;

  /// Registers the produced collections in FAIRRootManager
  void Register() override;

  /// Gets the produced collections
  std::vector<o2::itsmft::Hit>* getHits(Int_t iColl) const
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

 public:
  /// Has to be called after each event to reset the containers
  void Reset() override;

  /// Base class to create the detector geometry
  void ConstructGeometry() override;

  /// Sets the layer parameters
  /// \param nlay layer number
  /// \param phi0 layer phi0
  /// \param r layer radius
  /// \param nstav number of staves
  /// \param nunit IB: number of chips per stave
  /// \param OB: number of modules per half stave
  /// \param lthick stave thickness (if omitted, defaults to 0)
  /// \param dthick detector thickness (if omitted, defaults to 0)
  /// \param dettypeID ??
  /// \param buildLevel (if 0, all geometry is build, used for material budget studies)
  void defineLayer(Int_t nlay, Double_t phi0, Double_t r, Int_t nladd, Int_t nmod, Double_t lthick = 0.,
                   Double_t dthick = 0., UInt_t detType = 0, Int_t buildFlag = 0) override;

  /// Sets the layer parameters for new ITS3 geo
  /// \param nlay layer number
  /// \param r layer radius
  /// \param zlen layer length
  /// \param dthick detector thickness (if omitted, defaults to 0)
  /// \param dettypeID ??
  /// \param buildLevel (if 0, all geometry is build, used for material budget studies)
  void defineInnerLayerITS3(Int_t nlay, Double_t r, Double_t zlen,
                            Double_t dthick = 0., UInt_t detType = 0, Int_t buildFlag = 0);

  /// Gets the layer parameters
  /// \param nlay layer number
  /// \param phi0 phi of 1st stave
  /// \param r layer radius
  /// \param nstav number of staves
  /// \param nmod IB: number of chips per stave
  /// \param OB: number of modules per half stave
  /// \param width stave width
  /// \param tilt stave tilt angle
  /// \param lthick stave thickness
  /// \param dthick detector thickness
  /// \param dettype detector type
  virtual void getLayerParameters(Int_t nlay, Double_t& phi0, Double_t& r, Int_t& nladd, Int_t& nmod, Double_t& lthick, Double_t& mthick, UInt_t& dettype) const;

  /// This method is an example of how to add your own point of type Hit to the clones array
  o2::itsmft::Hit* addHit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
                          const TVector3& startMom, double startE, double endTime, double eLoss,
                          unsigned char startStatus, unsigned char endStatus);

  /// Set per wrapper volume parameters
  void defineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax, Double_t zspan) override;

  /// Add alignable top volumes
  void addAlignableVolumes() const override;

  /// Add alignable Layer volumes
  /// \param lr layer number
  /// \param parent path of the parent volume
  /// \param lastUID on output, UID of the last volume
  void addAlignableVolumesLayer(Int_t lr, TString& parent, Int_t& lastUID) const;

  /// Add alignable Stave volumes
  /// \param lr layer number
  /// \param st stave number
  /// \param parent path of the parent volume
  /// \param lastUID on output, UID of the last volume
  void addAlignableVolumesStave(Int_t lr, Int_t st, TString& parent, Int_t& lastUID) const;

  /// Add alignable HalfStave volumes
  /// \param lr layer number
  /// \param st stave number
  /// \param hst half stave number
  /// \param parent path of the parent volume
  /// \param lastUID on output, UID of the last volume
  void addAlignableVolumesHalfStave(Int_t lr, Int_t st, Int_t hst, TString& parent, Int_t& lastUID) const;

  /// Add alignable Module volumes
  /// \param lr layer number
  /// \param st stave number
  /// \param hst half stave number
  /// \param md module number
  /// \param parent path of the parent volume
  /// \param lastUID on output, UID of the last volume
  void addAlignableVolumesModule(Int_t lr, Int_t st, Int_t hst, Int_t md, TString& parent, Int_t& lastUID) const;

  /// Add alignable Chip volumes
  /// \param lr layer number
  /// \param st stave number
  /// \param hst half stave number
  /// \param md module number
  /// \param ch chip number
  /// \param parent path of the parent volume
  /// \param lastUID on output, UID of the last volume
  void addAlignableVolumesChip(Int_t lr, Int_t st, Int_t hst, Int_t md, Int_t ch, TString& parent,
                               Int_t& lastUID) const;

  /// Return Chip Volume UID
  /// \param id volume id
  Int_t chipVolUID(Int_t id) const { return o2::base::GeometryManager::getSensID(o2::detectors::DetID::IT3, id); }

  void EndOfEvent() override;

  void FinishPrimary() override { ; }
  virtual void finishRun() { ; }
  void BeginPrimary() override { ; }
  void PostTrack() override { ; }
  void PreTrack() override { ; }

  /// Returns the number of layers
  Int_t getNumberOfLayers() const { return sNumberLayers; }

  void createOuterBarrel(const Bool_t ob) { mCreateOuterBarrel = ob; };
  Bool_t isCreateOuterBarrel() { return mCreateOuterBarrel; };

  o2::its3::GeometryTGeo* mGeometryTGeo; //! access to geometry details

 protected:
  Int_t* mLayerID;     //! [sNumberLayers] layer identifier
  TString* mLayerName; //! [sNumberLayers] layer identifier

 private:
  /// this is transient data about track passing the sensor
  struct TrackData {               // this is transient
    bool mHitStarted;              //! hit creation started
    unsigned char mTrkStatusStart; //! track status flag
    TLorentzVector mPositionStart; //! position at entrance
    TLorentzVector mMomentumStart; //! momentum
    double mEnergyLoss;            //! energy loss
  } mTrackData;                    //!

  Int_t mNumberOfInnerLayers; //! number of ITS3 inner layers
  Int_t mTotalNumberOfLayers; //! total number of ITS3 layers (IB+OB)
  Bool_t mCreateOuterBarrel;  //! true to create the Outer Barrel

  Int_t mNumberOfDetectors;

  Bool_t mModifyGeometry;

  Double_t mWrapperMinRadius[sNumberOfWrapperVolumes]; //! Min radius of wrapper volume
  Double_t mWrapperMaxRadius[sNumberOfWrapperVolumes]; //! Max radius of wrapper volume
  Double_t mWrapperZSpan[sNumberOfWrapperVolumes];     //! Z span of wrapper volume
  Int_t* mWrapperLayerId;                              //! Id of wrapper layer to which layer belongs (-1 if not wrapped)

  Bool_t* mITS3Layer;           //! True for new ITS3 layers
  Double_t* mLayerPhi0;         //! Vector of layer's 1st stave phi in lab
  Double_t* mLayerRadii;        //! Vector of layer radii
  Double_t* mLayerZLen;         //! Vector of layer lengths
  Int_t* mStavePerLayer;        //! Vector of number of staves per layer
  Int_t* mUnitPerStave;         //! Vector of number of "units" per stave
  Double_t* mChipThickness;     //! Vector of chip thicknesses
  Double_t* mDetectorThickness; //! Vector of detector thicknesses
  UInt_t* mChipTypeID;          //! Vector of detector type id
  Int_t* mBuildLevel;           //! Vector of Material Budget Studies

  /// Container for hit data
  std::vector<o2::itsmft::Hit>* mHits;

  /// We need this as a method to access members
  void configITS3();

  /// Creates all needed arrays
  void createAllArrays();

  /// Creates an air-filled wrapper cylindrical volume
  TGeoVolume* createWrapperVolume(const Int_t nLay);

  /// Create the detector materials
  virtual void createMaterials();

  /// Construct the detector geometry
  void constructDetectorGeometry();

  /// Define the sensitive volumes of the geometry
  void defineSensitiveVolumes();

  /// Creates the Inner Barrel Services
  /// \param motherVolume the TGeoVolume owing the volume structure
  void createInnerBarrelServices(TGeoVolume* motherVolume);

  /// Creates the Middle Barrel Services
  /// \param motherVolume the TGeoVolume owing the volume structure
  void createMiddlBarrelServices(TGeoVolume* motherVolume);

  /// Creates the Outer Barrel Services
  /// \param motherVolume the TGeoVolume owing the volume structure
  void createOuterBarrelServices(TGeoVolume* motherVolume);

  /// Creates the Outer Barrel Supports
  /// \param motherVolume the TGeoVolume owing the volume supports
  void createOuterBarrelSupports(TGeoVolume* motherVolume);

  Detector(const Detector&);

  Detector& operator=(const Detector&);

  V3Layer** mGeometry;           //! Geometry
  V3Services* mServicesGeometry; //! Services Geometry

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};
} // namespace its3
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::its3::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif
