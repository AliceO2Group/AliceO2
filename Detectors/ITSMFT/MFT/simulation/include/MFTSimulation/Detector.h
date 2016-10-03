/// \file Detector.h
/// \brief Definition of the Detector class
/// \author antonio.uras@cern.ch, bogdan.vulpescu@cern.ch 
/// \date 01/08/2016

#ifndef ALICEO2_MFT_DETECTOR_H_
#define ALICEO2_MFT_DETECTOR_H_

#include "DetectorsBase/Detector.h"

namespace AliceO2 { namespace MFT { class GeometryTGeo; } }

namespace AliceO2 {
namespace MFT {

class Detector : public AliceO2::Base::Detector {

public:

  /// Default constructor
  Detector();

  /// Default destructor
  virtual ~Detector();

  Int_t IsVersion() const { return fVersion; }

  /// Initialization of the detector is done here
  virtual void Initialize();

  /// This method is called for each step during simulation (see FairMCApplication::Stepping())
  virtual Bool_t ProcessHits(FairVolume* v = 0);

  virtual void CopyClones(TClonesArray* cl1, TClonesArray* cl2, Int_t offset)
  {
    ;
  }
  virtual void EndOfEvent()
  {
    ;
  }
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
  virtual void SetSpecialPhysicsCuts()
  {
    ;
  }

  /// Has to be called after each event to reset the containers
  virtual void Reset()
  {
    ;
  }

  /// Registers the produced collections in FAIRRootManager
  virtual void Register() 
  {
    ;
  }

  /// Gets the produced collections
  virtual TClonesArray* GetCollection(Int_t iColl) const 
  {
    return NULL;
  }

  GeometryTGeo* GetGeometryTGeo() const { return fGeometryTGeo; }
  
  /// Creating materials for the detector

  void CreateMaterials();

  enum EMedia{kZero,kAir, kVacuum, kSi, kReadout, kSupport, kCarbon, kBe, kAlu, kWater, kSiO2, kInox, kKapton, kEpoxy, kCarbonFiber, kCarbonEpoxy, kRohacell, kPolyimide, kPEEK, kFR4, kCu, kX7R, kX7Rw, kCarbonFleece, kSE4445};  // media IDs used in CreateMaterials

  void SetDensitySupportOverSi(Double_t density) { 
    if (density > 1e-6) fDensitySupportOverSi = density; 
    else fDensitySupportOverSi = 1e-6; 
  }

  /// Constructing the geometry

  void ConstructGeometry();  // inherited from FairModule
  void CreateGeometry();

protected:

  Int_t fVersion;
  GeometryTGeo *fGeometryTGeo;

  Double_t fDensitySupportOverSi;
 
private:

  Detector(const Detector&);
  Detector& operator=(const Detector&);

  ClassDef(Detector,1)

};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);
}
}

#endif
