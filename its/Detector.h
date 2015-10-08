/// \file Detector.h
/// \brief Definition of the Detector class

#ifndef ALICEO2_ITS_DETECTOR_H_
#define ALICEO2_ITS_DETECTOR_H_

#include "Base/Detector.h"   // for Detector
#include "Rtypes.h"          // for Int_t, Double_t, Float_t, Bool_t, etc
#include "TArrayD.h"         // for TArrayD
#include "TGeoManager.h"     // for gGeoManager, TGeoManager (ptr only)
#include "TLorentzVector.h"  // for TLorentzVector
#include "TVector3.h"        // for TVector3
class FairModule;
class FairVolume;  // lines 16-16
class TClonesArray;  // lines 17-17
class TGeoVolume;
class TParticle;
class TString;
namespace AliceO2 { namespace ITS { class GeometryHandler; } }
namespace AliceO2 { namespace ITS { class MisalignmentParameter; } }
namespace AliceO2 { namespace ITS { class Point; } }  // lines 22-22
namespace AliceO2 { namespace ITS { class UpgradeGeometryTGeo; } }
namespace AliceO2 { namespace ITS { class UpgradeV1Layer; } }  // lines 23-23

namespace AliceO2 {
namespace ITS {

class Point;
class UpgradeV1Layer;

class Detector : public AliceO2::Base::Detector {

public:
  enum UpgradeModel {
    kIBModelDummy = 0,
    kIBModel0 = 1,
    kIBModel1 = 2,
    kIBModel21 = 3,
    kIBModel22 = 4,
    kIBModel3 = 5,
    kOBModelDummy = 6,
    kOBModel0 = 7,
    kOBModel1 = 8
  };

  /// Name : Detector Name
  /// Active: kTRUE for active detectors (ProcessHits() will be called)
  ///         kFALSE for inactive detectors
  Detector(const char* Name, Bool_t Active, const Int_t nlay);

  /// Default constructor
  Detector();

  /// Default destructor
  virtual ~Detector();

  /// Initialization of the detector is done here
  virtual void Initialize();

  /// This method is called for each step during simulation (see FairMCApplication::Stepping())
  virtual Bool_t ProcessHits(FairVolume* v = 0);

  /// Registers the produced collections in FAIRRootManager
  virtual void Register();

  /// Gets the produced collections
  virtual TClonesArray* GetCollection(Int_t iColl) const;

  /// Has to be called after each event to reset the containers
  virtual void Reset();

  /// Base class to create the detector geometry
  void ConstructGeometry();

  /// Creates the Service Barrel (as a simple cylinder) for IB and OB
  /// \param innerBarrel if true, build IB service barrel, otherwise for OB
  /// \param dest the mother volume holding the service barrel
  /// \param mgr  the gGeoManager pointer (used to get the material)
  void createServiceBarrel(const Bool_t innerBarrel, TGeoVolume* dest, const TGeoManager* mgr = gGeoManager);

  /// Initialize the parameter containers
  virtual void initializeParameterContainers();

  void setParameterContainers();

  /// Sets the layer parameters
  /// \param nlay layer number
  /// \param phi0 layer phi0
  /// \param r layer radius
  /// \param zlen layer length
  /// \param nstav number of staves
  /// \param nunit IB: number of chips per stave
  /// \param OB: number of modules per half stave
  /// \param lthick stave thickness (if omitted, defaults to 0)
  /// \param dthick detector thickness (if omitted, defaults to 0)
  /// \param dettypeID ??
  /// \param buildLevel (if 0, all geometry is build, used for material budget studies)
  virtual void defineLayer(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen, Int_t nladd, Int_t nmod,
                           Double_t lthick = 0., Double_t dthick = 0., UInt_t detType = 0, Int_t buildFlag = 0);

  /// Sets the layer parameters for a "turbo" layer
  /// (i.e. a layer whose staves overlap in phi)
  /// \param nlay layer number
  /// \param phi0 phi of 1st stave
  /// \param r layer radius
  /// \param zlen layer length
  /// \param nstav number of staves
  /// \param nunit IB: number of chips per stave
  /// \param OB: number of modules per half stave
  /// \param width stave width
  /// \param tilt layer tilt angle (degrees)
  /// \param lthick stave thickness (if omitted, defaults to 0)
  /// \param dthick detector thickness (if omitted, defaults to 0)
  /// \param dettypeID ??
  /// \param buildLevel (if 0, all geometry is build, used for material budget studies)
  virtual void defineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen, Int_t nladd, Int_t nmod,
                                Double_t width, Double_t tilt, Double_t lthick = 0., Double_t dthick = 0.,
                                UInt_t detType = 0, Int_t buildFlag = 0);

  /// Gets the layer parameters
  /// \param nlay layer number
  /// \param phi0 phi of 1st stave
  /// \param r layer radius
  /// \param zlen layer length
  /// \param nstav number of staves
  /// \param nmod IB: number of chips per stave
  /// \param OB: number of modules per half stave
  /// \param width stave width
  /// \param tilt stave tilt angle
  /// \param lthick stave thickness
  /// \param dthick detector thickness
  /// \param dettype detector type
  virtual void getLayerParameters(Int_t nlay, Double_t& phi0, Double_t& r, Double_t& zlen, Int_t& nladd, Int_t& nmod,
                                  Double_t& width, Double_t& tilt, Double_t& lthick, Double_t& mthick,
                                  UInt_t& dettype) const;

  /// This method is an example of how to add your own point of type Point to the clones array
  Point* addHit(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 pos, TVector3 mom, Double_t startTime,
                Double_t time, Double_t length, Double_t eLoss, Int_t shunt, Int_t status, Int_t statusStart);

  /// Book arrays for wrapper volumes
  virtual void setNumberOfWrapperVolumes(Int_t n);

  /// Set per wrapper volume parameters
  virtual void defineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax, Double_t zspan);

  // The following methods can be implemented if you need to make
  // any optional action in your detector during the transport

  virtual void CopyClones(TClonesArray* cl1, TClonesArray* cl2, Int_t offset)
  {
    ;
  }
  virtual void SetSpecialPhysicsCuts()
  {
    ;
  }
  virtual void EndOfEvent();
  virtual void FinishPrimary()
  {
    ;
  }
  virtual void finishRun()
  {
    ;
  }
  virtual void BeginPrimary()
  {
    ;
  }
  virtual void PostTrack()
  {
    ;
  }
  virtual void PreTrack()
  {
    ;
  }
  virtual void BeginEvent()
  {
    ;
  }

  /// Returns the pointer to the TParticle for the particle that created
  /// this hit. From the TParticle all kinds of information about this
  /// particle can be found. See the TParticle class.
  virtual TParticle* GetParticle() const;

  // SetTrack and GetTrack methods from AliHit.h

  virtual void SetTrack(Int_t track)
  {
    mTrackNumber = track;
  }
  virtual Int_t GetTrack() const
  {
    return mTrackNumber;
  }

  /// Prints out the content of this class in ASCII format
  /// \param ostream *os The output stream
  void Print(std::ostream* os) const;

  /// Reads in the content of this class in the format of Print
  /// \param istream *is The input stream
  void Read(std::istream* is);

  /// Returns the number of layers
  Int_t getNumberOfLayers() const
  {
    return mNumberLayers;
  }

  virtual void setStaveModelIB(UpgradeModel model)
  {
    mStaveModelInnerBarrel = model;
  }
  virtual void setStaveModelOB(UpgradeModel model)
  {
    mStaveModelOuterBarrel = model;
  }
  virtual UpgradeModel getStaveModelIB() const
  {
    return mStaveModelInnerBarrel;
  }
  virtual UpgradeModel getStaveModelOB() const
  {
    return mStaveModelOuterBarrel;
  }

  /// Clone this object (used in MT mode only)
  virtual FairModule* CloneModule() const;

  UpgradeGeometryTGeo* mGeometryTGeo; //! access to geometry details

protected:
  Int_t* mLayerID;               //! [mNumberLayers] layer identifier
  Int_t mNumberLayers;           //! Number of layers
  Int_t mStatus;                 //! Track Status
  Int_t mModule;                 //! Module number
  Float_t mParticlePx;           //! PX of particle at the point of the hit
  Float_t mParticlePy;           //! PY of particle at the point of the hit
  Float_t mParticlePz;           //! PZ of particle at the point of the hit
  Float_t mEnergyDepositionStep; //! Energy deposited in the current step
  Float_t mTof;                  //! Time of flight at the point of the hit
  Int_t mStatus0;                //! Track Status of Starting point
  Float_t mStartingStepX;        //! Starting point of this step
  Float_t mStartingStepY;        //! Starting point of this step
  Float_t mStartingStepZ;        //! Starting point of this step
  Float_t mStartingStepT;        //! Starting point of this step
  Int_t mTrackNumber;            //! Track number
  Float_t mPositionX;            //! X position of the hit
  Float_t mPositionY;            //! Y position of the hit
  Float_t mPositionZ;            //! Z position of the hit
  TString* mLayerName;           //![mNumberLayers] layer identifier

private:
  /// Track information to be stored until the track leaves the
  /// active volume.
  Int_t mTrackNumberID;             //! track index
  Int_t mVolumeID;                  //! volume id
  Int_t mShunt;                     //! shunt
  Int_t mTrkStatusFlag;             //! track status flag
  TLorentzVector mPosition;         //! position
  TLorentzVector mEntrancePosition; //! position at entrance
  TLorentzVector mMomentum;         //! momentum
  Double32_t mEntranceTime;         //! time at entrance
  Double32_t mTime;                 //! time
  Double32_t mLength;               //! length
  Double32_t mEnergyLoss;           //! energy loss

  Int_t mNumberOfDetectors;
  TArrayD mShiftX;
  TArrayD mShiftY;
  TArrayD mShiftZ;
  TArrayD mRotX;
  TArrayD mRotY;
  TArrayD mRotZ;

  Bool_t mModifyGeometry;

  Int_t mNumberOfWrapperVolumes; //! number of wrapper volumes
  Double_t* mWrapperMinRadius;   //! min radius of wrapper volume
  Double_t* mWrapperMaxRadius;   //! max radius of wrapper volume
  Double_t* mWrapperZSpan;       //! Z span of wrapper volume
  Int_t* mWrapperLayerId;        //! id of wrapper layer to which layer belongs (-1 if not wrapped)
  Bool_t* mTurboLayer;           //! True for "turbo" layers
  Double_t* mLayerPhi0;          //! Vector of layer's 1st stave phi in lab
  Double_t* mLayerRadii;         //! Vector of layer radii
  Double_t* mLayerZLength;       //! Vector of layer length along Z
  Int_t* mStavePerLayer;         //! Vector of number of staves per layer
  Int_t* mUnitPerStave;          //! Vector of number of "units" per stave
  Double_t* mStaveThickness;     //! Vector of stave thicknesses
  Double_t* mStaveWidth;         //! Vector of stave width (only used for turbo)
  Double_t* mStaveTilt;          //! Vector of stave tilt (only used for turbo)
  Double_t* mDetectorThickness;  //! Vector of detector thicknesses
  UInt_t* mChipTypeID;           //! Vector of detector type id
  Int_t* mBuildLevel;            //! Vector of Material Budget Studies

  /// Container for data points
  TClonesArray* mPointCollection;

  /// Creates an air-filled wrapper cylindrical volume
  TGeoVolume* createWrapperVolume(const Int_t nLay);

  /// Create the detector materials
  virtual void createMaterials();

  /// Construct the detector geometry
  void constructDetectorGeometry();

  /// Define the sensitive volumes of the geometry
  void defineSensitiveVolumes();

  Detector(const Detector&);
  Detector& operator=(const Detector&);

  GeometryHandler* mGeometryHandler;
  MisalignmentParameter* mMisalignmentParameter;

  UpgradeV1Layer** mUpgradeGeometry;   //! Geometry
  UpgradeModel mStaveModelInnerBarrel; //! The stave model for the Inner Barrel
  UpgradeModel mStaveModelOuterBarrel; //! The stave model for the Outer Barrel

  ClassDef(Detector, 1)
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);
}
}

#endif
