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

#ifndef ALICEO2_FT0_DETECTOR_H_
#define ALICEO2_FT0_DETECTOR_H_

#include "SimulationDataFormat/BaseHits.h"
#include "DetectorsBase/Detector.h" // for Detector
#include "FT0Base/Geometry.h"
#include "DataFormatsFT0/HitType.h"

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
  std::vector<Float_t> mRindexAir;
  std::vector<Float_t> mAbsorAir;
  std::vector<Float_t> mRindexCathodeNext;
  std::vector<Float_t> mAbsorbCathodeNext;
  std::vector<Double_t> mEfficMet;
  std::vector<Double_t> mReflMet;

  /// Container for data points
  std::vector<o2::ft0::HitType>* mHits = nullptr;

  /// Define the sensitive volumes of the geometry
  void defineSensitiveVolumes();

  // Define the aluminium frame for the detector
  TGeoVolume* ConstructFrameGeometry();
  std::string frame1CompositeShapeBoolean();
  std::string frame2CompositeShapeBoolean();
  std::string frameCompositeShapeBoolean();
  std::string plateGroupCompositeShapeBoolean();
  std::string opticalFiberPlateCompositeShapeBoolean1();
  std::string opticalFiberPlateCompositeShapeBoolean2();
  std::string pmtCornerCompositeShapeBoolean();
  std::string pmtCompositeShapeBoolean();
  std::string plateBoxCompositeShapeBoolean();
  void defineTransformations();
  void defineQuartzRadiatorTransformations();
  void definePMTTransformations();
  void definePlateTransformations();
  void defineFrameTransformations();

  // define some error to avoid overlaps
  Float_t eps = 0.025;

  // frame 1 has a longer side horizontal
  Float_t frame1X = 21.500;
  Float_t frame1Y = 13.705;
  Float_t frame1PosX = 7.9278;
  Float_t frame1PosY = 9.2454;
  Float_t rect1X = 15;
  Float_t rect1Y = 1.33;
  Float_t rect2X = 2.9;
  Float_t rect2Y = 12.2;
  Float_t rect3X = 1.57;
  Float_t rect3Y = .175;
  Float_t rect4X = 5.65;
  Float_t rect4Y = 1.075;

  // frame 2 has a longer side vertical
  Float_t frame2X = 13.930;
  Float_t frame2Y = 21.475;
  Float_t frame2PosX = 10.1428;
  Float_t frame2PosY = -8.3446;
  Float_t rect5X = 1.33;
  Float_t rect5Y = 12.1;
  Float_t rect6X = .83;
  Float_t rect6Y = 3.0;
  Float_t rect7X = 13.1;
  Float_t rect7Y = 3.0;
  Float_t rect8X = 1.425;
  Float_t rect8Y = 5.5;

  // both frame boxes are the same height
  Float_t frameZ = 5.700;
  Float_t mountZ = 1.5;

  // PMT dimensions
  Float_t pmtSide = 5.950;
  Float_t pmtZ = 3.750;

  // quartz radiator dimensions
  Float_t quartzRadiatorSide = 5.350;
  Float_t quartzRadiatorZ = 1.950;

  // for the rounded corners
  Float_t cornerRadius = .300;

  // bottom plates on the frame
  Float_t plateSide = 6.000;
  Float_t basicPlateZ = 0.200;
  Float_t cablePlateZ = 0.500;
  Float_t fiberHeadX = 0.675 * 2;
  Float_t fiberHeadY = 0.275 * 2;

  // plate transformations
  Float_t opticalFiberPlateZ = 0.35;
  Float_t plateSpacing = 6.100;
  Float_t plateDisplacementDeltaY = 1.33;
  Float_t plateDisplacementX = plateSpacing + 0.3028;
  Float_t plateDisplacementY = 12.8789 - plateDisplacementDeltaY;
  Float_t plateGroupZ = -frameZ / 2 - opticalFiberPlateZ;

  // quartz & PMT transformations
  Float_t quartzHeight = -frameZ / 2 + quartzRadiatorZ / 2;
  Float_t PMTHeight = frameZ / 2 - pmtZ / 2;
  Float_t pmtCornerTubePos = -.15;
  Float_t pmtCornerPos = 2.825;
  Float_t edgeCornerPos[2] = {-6.515, -.515};
  Float_t quartzFrameOffsetX = -1.525;
  Float_t pos1X[3] = {quartzFrameOffsetX - plateSpacing, quartzFrameOffsetX, quartzFrameOffsetX + plateSpacing};
  Float_t pos1Y[4] = {3.6275, -2.4725, 2.2975, -3.8025};
  Float_t pos2X[4] = {3.69, -2.410, 2.360, -3.740};
  Float_t pos2Y[3] = {7.6875, 1.5875, -4.5125};

  Detector& operator=(const Detector&);

  Geometry* mGeometry = nullptr; //! Geometry

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
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
