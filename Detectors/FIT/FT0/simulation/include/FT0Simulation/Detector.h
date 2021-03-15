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

#include <TGeoShape.h>
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
    //   kOpAir = 7,
    kAl = 15,
    kOpGlass = 16,
    kOptAl = 17,
    kOptBlack = 18,
    kOpGlassCathode = 19,
    //   kSensAir = 22,
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

  //  Geometry* GetGeometry();

  /// Prints out the content of this class in ASCII format
  /// \param ostream *os The output stream
  void Print(std::ostream* os) const;

  /// Reads in the content of this class in the format of Print
  /// \param istream *is The input stream
  void Read(std::istream* is);

  void DefineSim2LUTindex();
  /// Create the shape of cables for a specified cell.
  /// \param  shapeName   The name of the shape.
  /// \param  cellID      The number of the cell
  /// \return The cable pattern shape.
  // TGeoShape* CreateCableShape(const std::string& shapeName, int cellID) const;

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

  // Define the aluminium frame for the detector
  TGeoVolume* constructFrameGeometry();
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
  void definePmtTransformations();
  void definePlateTransformations();
  void defineFrameTransformations();

  // BEGIN: Support structure constants
  // define some error to avoid overlaps
  static constexpr Float_t sEps = 0.05;
  // offset found to potentially remove overlaps
  static constexpr Float_t sXoffset = 0.3027999999999995;
  static constexpr Float_t sYoffset = -0.6570999999999998;

  // frame 1 has a longer side horizontal
  static constexpr Float_t sFrame1X = 21.500;
  static constexpr Float_t sFrame1Y = 13.705;
  static constexpr Float_t sFrame1PosX = 7.9278 - sXoffset;
  static constexpr Float_t sFrame1PosY = 9.2454 - sYoffset;
  static constexpr Float_t sRect1X = 15;
  static constexpr Float_t sRect1Y = 1.33;
  static constexpr Float_t sRect2X = 2.9;
  static constexpr Float_t sRect2Y = 12.2;
  static constexpr Float_t sRect3X = 1.57;
  static constexpr Float_t sRect3Y = .175;
  static constexpr Float_t sRect4X = 5.65;
  static constexpr Float_t sRect4Y = 1.075;
  // frame 2 has a longer side vertical
  static constexpr Float_t sFrame2X = 13.930;
  static constexpr Float_t sFrame2Y = 21.475;
  static constexpr Float_t sFrame2PosX = 10.1428 - sXoffset;
  static constexpr Float_t sFrame2PosY = -8.3446 - sYoffset;
  static constexpr Float_t sRect5X = 1.33;
  static constexpr Float_t sRect5Y = 12.1;

  static constexpr Float_t sRect6X = .83;
  static constexpr Float_t sRect6Y = 3.0;
  static constexpr Float_t sRect7X = 13.1;
  static constexpr Float_t sRect7Y = 3.0;
  static constexpr Float_t sRect8X = 1.425;
  static constexpr Float_t sRect8Y = 5.5;

  // both frame boxes are the same height
  static constexpr Float_t sFrameZ = 5.700;
  static constexpr Float_t sMountZ = 1.5;

  // PMT socket dimensions
  static constexpr Float_t sPmtSide = 5.950;
  static constexpr Float_t sPmtZ = 3.750;

  // quartz radiator socket dimensions
  // static constexpr Float_t sQuartzRadiatorSide = 5.350;
  // static constexpr Float_t sQuartzRadiatorZ = 1.950;
  static constexpr Float_t sQuartzRadiatorSide = 5.40;
  static constexpr Float_t sQuartzRadiatorZ = 2.0;
  // for the rounded socket corners
  static constexpr Float_t sCornerRadius = .300;

  // bottom plates on the frame
  static constexpr Float_t sPlateSide = 6.000;
  static constexpr Float_t sBasicPlateZ = 0.200;
  static constexpr Float_t sCablePlateZ = 0.500;
  static constexpr Float_t sFiberHeadX = 0.675 * 2;
  static constexpr Float_t sFiberHeadY = 0.275 * 2;

  // plate transformations
  static constexpr Float_t sOpticalFiberPlateZ = 0.35;
  static constexpr Float_t sPlateSpacing = 6.100;
  static constexpr Float_t sPlateDisplacementDeltaY = 1.33;
  static constexpr Float_t sPlateDisplacementX = sPlateSpacing + 0.3028;
  static constexpr Float_t sPlateDisplacementY = 12.8789 - sPlateDisplacementDeltaY;
  static constexpr Float_t sPlateGroupZ = -sFrameZ / 2 - sOpticalFiberPlateZ;

  // quartz & PMT socket transformations
  static constexpr Float_t sQuartzHeight = -sFrameZ / 2 + sQuartzRadiatorZ / 2;
  static constexpr Float_t sPmtHeight = sFrameZ / 2 - sPmtZ / 2;
  static constexpr Float_t sPmtCornerTubePos = -.15;
  static constexpr Float_t sPmtCornerPos = 2.825;
  static constexpr Float_t sEdgeCornerPos[2] = {-6.515, -.515};
  static constexpr Float_t sQuartzFrameOffsetX = -1.525;
  static constexpr Float_t sPos1X[3] = {sQuartzFrameOffsetX - sPlateSpacing, sQuartzFrameOffsetX, sQuartzFrameOffsetX + sPlateSpacing};
  static constexpr Float_t sPos1Y[4] = {3.6275, -2.4725, 2.2975, -3.8025};
  static constexpr Float_t sPos2X[4] = {3.69, -2.410, 2.360, -3.740};
  static constexpr Float_t sPos2Y[3] = {7.6875, 1.5875, -4.5125};
  //END: Support structure constants

  /// Container for data points
  std::vector<o2::ft0::HitType>* mHits = nullptr;

  /// Define the sensitive volumes of the geometry
  void defineSensitiveVolumes();

  Detector& operator=(const Detector&);

  Geometry* mGeometry = nullptr; //! Geometry

  template <typename Det>
  friend class o2::base::DetImpl;

  int mTrackIdTop;
  int mTrackIdMCPtop; //TEMPORARY

  int mSim2LUT[Geometry::Nchannels];

  Float_t mPosModuleAx[Geometry::NCellsA] = {-12.2, -6.1, 0, 6.1, 12.2, -12.2, -6.1, 0,
                                             6.1, 12.2, -13.3743, -7.274299999999999,
                                             7.274299999999999, 13.3743, -12.2, -6.1, 0,
                                             6.1, 12.2, -12.2, -6.1, 0, 6.1, 12.2};

  Float_t mPosModuleAy[Geometry::NCellsA] = {12.2, 12.2, 13.53, 12.2, 12.2, 6.1, 6.1,
                                             7.43, 6.1, 6.1, 0, 0, 0, 0, -6.1, -6.1,
                                             -7.43, -6.1, -6.1, -12.2, -12.2, -13.53,
                                             -12.2, -12.2};

  float mPosModuleCx[Geometry::NCellsC];
  float mPosModuleCy[Geometry::NCellsC];
  float mPosModuleCz[Geometry::NCellsC];
  Float_t mStartC[3] = {20., 20, 5.5};
  Float_t mStartA[3] = {20., 20., 5};
  Float_t mInStart[3] = {2.9491, 2.9491, 2.5};

  ClassDefOverride(Detector, 4);
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
