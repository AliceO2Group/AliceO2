/// \file Detector.h
/// \brief Definition of the Detector class

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
    ;
  }

  const GeometryTGeo* GetGeometryTGeo() const { return mGeometryTGeo; }
  /*
  void CreateMaterials();

  enum EMedia{kZero,kAir, kVacuum, kSi, kReadout, kSupport, kCarbon, kBe, kAlu, kWater, kSiO2, kInox, kKapton, kEpoxy, kCarbonFiber, kCarbonEpoxy, kRohacell, kPolyimide, kPEEK, kFR4, kCu, kX7R, kX7Rw, kCarbonFleece, kSE4445};  // media IDs used in CreateMaterials
  */
protected:

  Int_t fVersion;

private:

  Detector(const Detector&);
  Detector& operator=(const Detector&);

  GeometryTGeo *mGeometryTGeo;

  ClassDef(Detector,1)

};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);
}
}

#endif
