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

#ifndef ALICEO2_FT0_DETECTOR_H_
#define ALICEO2_FT0_DETECTOR_H_

#include <TGeoShape.h>
#include "SimulationDataFormat/BaseHits.h"
#include "DetectorsBase/Detector.h" // for Detector
#include "FT0Base/Geometry.h"
#include "DataFormatsFT0/HitType.h"
#include <TGeoManager.h> // for gGeoManager, TGeoManager (ptr only)

class FairModule;

class FairVolume;
class TGeoVolume;
class TGraph;

namespace o2
{
namespace ft0
{
class Geometry;
}
} // namespace o2

namespace o2
{
namespace ft0
{
// using HitType = o2::BasicXYZEHit<float>;
class Geometry;
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  enum constants {
    kAir = 1,
    kVac = 3,
    kCeramic = 4,
    kGlass = 6,
    kAl = 15,
    kOpGlass = 16,
    kOptAl = 17,
    kOpGlassCathode = 19,
    kCable = 23,
    kMCPwalls = 25
  }; // materials

  /// Name : Detector Name
  /// Active: kTRUE for active detectors (ProcessHits() will be called)
  ///         kFALSE for inactive detectors
  Detector(Bool_t Active);

  /// Default constructor
  Detector() = default;

  /// Destructor
  ~Detector() override;

  /// Initialization of the detector is done here
  void InitializeO2Detector() override;

  /// This method is called for each step during simulation (see FairMCApplication::Stepping())
  Bool_t ProcessHits(FairVolume* v) override;
  o2::ft0::HitType* AddHit(float x, float y, float z, float time, float energy, Int_t trackId, Int_t detId);

  void Register() override;

  std::vector<o2::ft0::HitType>* getHits(Int_t iColl)
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
  void ConstructOpGeometry() override;
  void SetOneMCP(TGeoVolume* stl);

  void SetCablesA(TGeoVolume* stl);
  TGeoVolume* SetCablesSize(int mod);

  // Optical properties reader: e-Energy, abs-AbsorptionLength[cm], n-refractive index
  void DefineOpticalProperties();
  Int_t ReadOptProperties(const std::string inputFilePath);
  void FillOtherOptProperties();
  Bool_t RegisterPhotoE(float energy);
  void Print(std::ostream* os) const;

  /// Reads in the content of this class in the format of Print
  /// \param istream *is The input stream
  void Read(std::istream* is);

  void DefineSim2LUTindex();
  /// Add alignable  volumes
  void addAlignableVolumes() const override;
  // Return Chip Volume UID
  /// \param id volume id
  Int_t chipVolUID(Int_t id) const
  {
    return o2::base::GeometryManager::getSensID(o2::detectors::DetID::FT0, id);
  }

 private:
  /// copy constructor (used in MT)
  Detector(const Detector& rhs);

  Int_t mIdSens1;            // Sensetive volume  in T0
  TGraph* mPMTeff = nullptr; // pmt registration effeicincy

  // Optical properties to be extracted from file
  std::vector<Double_t> mPhotonEnergyD;
  std::vector<Double_t> mAbsorptionLength;
  std::vector<Double_t> mRefractionIndex;
  std::vector<Double_t> mQuantumEfficiency;

  // Optical properties to be set to constants
  std::vector<Double_t> mEfficAll;
  std::vector<Double_t> mRindexAir;
  std::vector<Double_t> mAbsorAir;
  std::vector<Double_t> mRindexCathodeNext;
  std::vector<Double_t> mAbsorbCathodeNext;
  std::vector<Double_t> mEfficMet;
  std::vector<Double_t> mReflMet;
  std::vector<Double_t> mRindexMet;
  std::vector<Double_t> mReflBlackPaper;
  std::vector<Double_t> mAbsBlackPaper;
  std::vector<Double_t> mEffBlackPaper;
  std::vector<Double_t> mReflFrontWindow;
  std::vector<Double_t> mEffFrontWindow;
  std::vector<Double_t> mRindexFrontWindow;

  // Define the aluminium frame for the detector
  TGeoVolume* constructFrameAGeometry(); //A-side
  TGeoVolume* constructFrameCGeometry(); //C-side
  std::string cPlateShapeString();

  // BEGIN: Support structure constants
  // define some error to avoid overlaps
  static constexpr Float_t sEps = 0.05;
  static constexpr Float_t sFrameZ = 5.700;

  // PMT socket dimensions
  static constexpr Float_t sPmtSide = 5.950;
  static constexpr Float_t sPmtZ = 3.750;

  // quartz radiator socket dimensions
  static constexpr Float_t sQuartzRadiatorSide = 5.350;
  static constexpr Float_t sQuartzRadiatorZ = 1.950;
  // for the rounded socket corners
  static constexpr Float_t sCornerRadius = 0.300;

  // quartz & PMT socket transformations
  static constexpr Float_t sQuartzHeight = -sFrameZ / 2 + sQuartzRadiatorZ / 2;
  static constexpr Float_t sPmtHeight = sFrameZ / 2 - sPmtZ / 2;
  static constexpr Float_t sPmtCornerTubePos = -0.15;
  static constexpr Float_t sPmtCornerPos = 2.825;
  static constexpr Float_t sEdgeCornerPos[2] = {-6.515, -0.515};
  static constexpr Float_t sQuartzFrameOffsetX = -1.525;
  // END: Support structure constants

  /// Container for data points
  std::vector<o2::ft0::HitType>* mHits = nullptr;

  // Define volume IDs
  int mREGVolID = -1;
  int mTOPVolID = -1;
  int mMTOVolID = -1;

  /// Define the sensitive volumes of the geometry
  void defineSensitiveVolumes();

  Detector& operator=(const Detector&);

  Geometry* mGeometry = nullptr; //! Geometry

  template <typename Det>
  friend class o2::base::DetImpl;

  int mTrackIdTop;
  int mTrackIdMCPtop; // TEMPORARY

  int mSim2LUT[Geometry::Nchannels];

  Double_t mPosModuleAx[Geometry::NCellsA];
  Double_t mPosModuleAy[Geometry::NCellsA];
  Double_t mPosModuleCx[Geometry::NCellsC];
  Double_t mPosModuleCy[Geometry::NCellsC];
  Double_t mPosModuleCz[Geometry::NCellsC];
  Float_t mStartC[3] = {20., 20, 5.5};
  Float_t mStartA[3] = {20., 20., 5};
  Float_t mInStart[3] = {2.9491, 2.9491, 2.6};

  ClassDefOverride(Detector, 6);
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);

std::istream& operator>>(std::istream& os, Detector& source);
} // namespace ft0
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::ft0::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif
