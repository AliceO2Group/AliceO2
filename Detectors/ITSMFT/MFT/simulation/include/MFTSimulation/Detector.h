/// \file Detector.h
/// \brief Definition of the Detector class
/// \author antonio.uras@cern.ch, bogdan.vulpescu@cern.ch 
/// \date 01/08/2016

#ifndef ALICEO2_MFT_DETECTOR_H_
#define ALICEO2_MFT_DETECTOR_H_

#include "TLorentzVector.h"

#include "DetectorsBase/Detector.h"

class TClonesArray;
class TVector3;

namespace o2 { namespace MFT { class GeometryTGeo; } }
namespace o2 { namespace ITSMFT { class Point; } }

namespace o2 {
namespace MFT {

class Detector : public o2::Base::Detector {

public:

  /// Default constructor
  Detector();

  /// Default destructor
  ~Detector() override;

  /// Initialization of the detector is done here
  void Initialize() override;

  /// This method is called for each step during simulation (see FairMCApplication::Stepping())
  Bool_t ProcessHits(FairVolume* v = nullptr) override;

  /// Has to be called after each event to reset the containers
  void Reset() override;

  /// Registers the produced collections in FAIRRootManager
  void Register() override; 

  /// Gets the produced collections
  TClonesArray* GetCollection(Int_t iColl) const override;

  void EndOfEvent() override;

  void CopyClones(TClonesArray* cl1, TClonesArray* cl2, Int_t offset) override {;}
  void FinishPrimary() override {;}
  void FinishRun() override {;}
  void BeginPrimary() override {;}
  void PostTrack() override {;}
  void PreTrack() override {;}
  void BeginEvent() override {;}
  void SetSpecialPhysicsCuts() override {;}

  void ConstructGeometry() override;  // inherited from FairModule

  //

  Int_t isVersion() const { return mVersion; }

  GeometryTGeo* getGeometryTGeo() const { return mGeometryTGeo; }
  
  /// Creating materials for the detector

  void createMaterials();

  enum EMedia{Zero, Air, Vacuum, Si, Readout, Support, Carbon, Be, Alu, Water, SiO2, Inox, Kapton, Epoxy, CarbonFiber, CarbonEpoxy, Rohacell, Polyimide, PEEK, FR4, Cu, X7R, X7Rw, CarbonFleece, SE4445};  // media IDs used in CreateMaterials

  void setDensitySupportOverSi(Double_t density) { 
    if (density > 1e-6) mDensitySupportOverSi = density; 
    else mDensitySupportOverSi = 1e-6; 
  }

  void createGeometry();
  void defineSensitiveVolumes();

protected:

  Int_t mVersion;                  //
  GeometryTGeo *mGeometryTGeo;     //!
  Double_t mDensitySupportOverSi;  //
  TClonesArray *mPoints;           //!
 
private:

  Detector(const Detector&);
  Detector& operator=(const Detector&);

  o2::ITSMFT::Point* addHit(int trackID, int detID, TVector3 startPos, TVector3 endPos, TVector3 startMom, double startE, double endTime, double eLoss, unsigned char startStatus, unsigned char endStatus);

  /// this is transient data about track passing the sensor
  struct TrackData {                  // this is transient 
    bool  mHitStarted;                //! hit creation started
    unsigned char mTrkStatusStart;    //! track status flag
    TLorentzVector mPositionStart;    //! position at entrance
    TLorentzVector mMomentumStart;    //! momentum
    double mEnergyLoss;               //! energy loss
  } mTrackData;                       //! 

  ClassDefOverride(Detector,1)

};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);
}
}

#endif
