// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Detector.h
/// \brief Definition of the Detector class

#ifndef ALICEO2_FIT_DETECTOR_H_
#define ALICEO2_FIT_DETECTOR_H_

#include "DetectorsBase/Detector.h"   // for Detector
#include "SimulationDataFormat/BaseHits.h"
#include "FITBase/Geometry.h"
#include <TGraph.h>

class FairModule;

class FairVolume;
class TClonesArray;
class TGeoVolume;

class TParticle;

class TString;

namespace o2 { namespace FIT { class Geometry; }}

namespace o2 {
namespace FIT {
using HitType = o2::BasicXYZEHit<float>;
//class Hit;
class Geometry;
class Detector : public o2::Base::Detector
{

 public:
  enum constants {kAir=1, kVac=3, kGlass=6, kOpAir=7, kAl=15, kOpGlass=16, kOpGlassCathode=19,kSensAir=22}; //materials
  
  /// Name : Detector Name
  /// Active: kTRUE for active detectors (ProcessHits() will be called)
  ///         kFALSE for inactive detectors
  Detector(const char *Name, Bool_t Active);
  
  /// Default constructor
  Detector() {}
   
  /// Initialization of the detector is done here
  void Initialize() override;

  /// This method is called for each step during simulation (see FairMCApplication::Stepping())
  Bool_t ProcessHits(FairVolume *v = nullptr) override;
  o2::BasicXYZEHit<float>* AddHit(float x, float y, float z, float time, float energy, Int_t trackId, Int_t detId);
  
  void Register() override;
  
  TClonesArray* GetCollection(Int_t iColl) const final;
  
  void Reset() override;
  
  
  /// Base class to create the detector geometry
  void CreateMaterials();
  void ConstructGeometry() override;
  void SetOneMCP(TGeoVolume *stl);
// Optical properties reader: e-Energy, abs-AbsorptionLength[cm], n-refractive index
  Int_t ReadOptProperties(const std::string inputFilePath, Float_t **e,
			  Double_t **de, Float_t **abs, Float_t **n, Float_t **qe, Int_t &kNbins) const;
  void DefineOpticalProperties();
  void FillOtherOptProperties(Float_t **efficAll, Float_t **rindexAir, Float_t **absorAir,
			      Float_t **rindexCathodeNext, Float_t **absorbCathodeNext,
			      Double_t **efficMet, Double_t **aReflMet, const Int_t kNbins) const;  
  void DeleteOptPropertiesArr(Float_t **e, Double_t **de, Float_t **abs,
			 Float_t **n, Float_t **efficAll, Float_t **rindexAir, Float_t **absorAir,
			 Float_t **rindexCathodeNext, Float_t **absorbCathodeNext,
			 Double_t **efficMet, Double_t **aReflMet) const;
   Bool_t RegisterPhotoE(Double_t energy);
 
   //  Geometry* GetGeometry();
    
   /// Prints out the content of this class in ASCII format
   /// \param ostream *os The output stream
   void Print(std::ostream *os) const;
   
   /// Reads in the content of this class in the format of Print
   /// \param istream *is The input stream
   void Read(std::istream *is);
   

   /// Clone this object (used in MT mode only)
   // FairModule *CloneModule() const override;
    
    
 private:
     
  Int_t mIdSens1; // Sensetive volume  in T0
  TGraph *mPMTeff; //pmt registration effeicincy

  /// Container for data points
  TClonesArray *mHitCollection;


  /// Define the sensitive volumes of the geometry
  void defineSensitiveVolumes();

  Detector(const Detector &);
  
  Detector &operator=(const Detector &);
    
  Geometry *mGeometry;   //! Geometry
  
  ClassDefOverride(Detector, 1)
};
 
// Input and output function for standard C++ input/output.
std::ostream &operator<<(std::ostream &os, Detector &source);

std::istream &operator>>(std::istream &os, Detector &source);
}
}

#endif
