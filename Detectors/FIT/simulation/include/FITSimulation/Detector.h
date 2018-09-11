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

#include "SimulationDataFormat/BaseHits.h"
#include "DetectorsBase/Detector.h" // for Detector
#include "FITBase/Geometry.h"

class FairVolume;
class TGeoVolume;
class TGraph;

namespace o2
{
namespace fit
{
class Geometry;
}
} // namespace o2

namespace o2
{
namespace fit
{
using HitType = o2::BasicXYZEHit<float>;
class Geometry;
class Detector : public o2::Base::DetImpl<Detector>
{
 public:
  enum constants {
    kAir = 1,
    kVac = 3,
    kGlass = 6,
    kOpAir = 7,
    kAl = 15,
    kOpGlass = 16,
    kOpGlassCathode = 19,
    kSensAir = 22
  }; // materials

  /// Name : Detector Name
  /// Active: kTRUE for active detectors (ProcessHits() will be called)
  ///         kFALSE for inactive detectors
  Detector(Bool_t Active);

  /// Default constructor
  Detector() = default;
 
  /// Initialization of the detector is done here
  void InitializeO2Detector() override;

  /// This method is called for each step during simulation (see FairMCApplication::Stepping())
  Bool_t ProcessHits(FairVolume* v) override;
  HitType* AddHit(float x, float y, float z, float time, float energy, Int_t trackId, Int_t detId);

  void Register() override;

  std::vector<HitType>* getHits(Int_t iColl)
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  void Reset() override;
  void EndOfEvent() override { Reset(); }

  /// Base class to create the detector geometry
  void CreateMaterials();
  void ConstructGeometry() override;
  void SetOneMCP(TGeoVolume* stl);

  // Optical properties reader: e-Energy, abs-AbsorptionLength[cm], n-refractive index
  void DefineOpticalProperties();
  Int_t ReadOptProperties(const std::string inputFilePath);
  void FillOtherOptProperties();
  Bool_t RegisterPhotoE(float energy);

  //  Geometry* GetGeometry();

  /// Prints out the content of this class in ASCII format
  /// \param ostream *os The output stream
  void Print(std::ostream* os) const;

  /// Reads in the content of this class in the format of Print
  /// \param istream *is The input stream
  void Read(std::istream* is);

 private:
  Int_t mIdSens1;            // Sensetive volume  in T0
  TGraph* mPMTeff = nullptr; // pmt registration effeicincy

  // Optical properties to be extracted from file
  std::vector<Double_t> mPhotonEnergyD;
  std::vector<Double_t> mAbsorptionLength;
  std::vector<Double_t> mRefractionIndex;
  std::vector<Double_t> mQuantumEfficiency;

  // Optical properties to be set to constants
  std::vector<Double_t> mEfficAll;
  std::vector<Float_t> mRindexAir;
  std::vector<Float_t> mAbsorAir;
  std::vector<Float_t> mRindexCathodeNext;
  std::vector<Float_t> mAbsorbCathodeNext;
  std::vector<Double_t> mEfficMet;
  std::vector<Double_t> mReflMet;

  /// Container for data points
  std::vector<HitType>* mHits = nullptr;

  /// Define the sensitive volumes of the geometry
  void defineSensitiveVolumes();

  Detector(const Detector&);

  Detector& operator=(const Detector&);

  Geometry* mGeometry = nullptr; //! Geometry

  template <typename Det>
  friend class o2::Base::DetImpl;
  ClassDefOverride(Detector, 1)
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);

std::istream& operator>>(std::istream& os, Detector& source);
} // namespace fit
} // namespace o2

#endif
