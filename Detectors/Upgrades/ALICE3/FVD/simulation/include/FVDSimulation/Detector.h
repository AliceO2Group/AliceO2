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

#ifndef ALICEO2_FVD_DETECTOR_H_
#define ALICEO2_FVD_DETECTOR_H_

#include "SimulationDataFormat/BaseHits.h"
#include "DetectorsBase/Detector.h"
#include "FVDBase/GeometryTGeo.h"
#include "FVDBase/FVDBaseParam.h"
#include "ITSMFTSimulation/Hit.h"
#include "Rtypes.h"
#include "TGeoManager.h"
#include "TLorentzVector.h"
#include "TVector3.h"

class FairVolume;
class TGeoVolume;

namespace o2
{
namespace fvd
{
class GeometryTGeo;
}
} // namespace o2

namespace o2
{
namespace fvd
{

class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(bool Active);

  Detector() = default;

  ~Detector() override;

  void ConstructGeometry() override;

  /// This method is an example of how to add your own point of type Hit to the clones array
  o2::itsmft::Hit* addHit(int trackID, int detID,
                          const TVector3& startPos,
                          const TVector3& endPos,
                          const TVector3& startMom,
                          double startE,
                          double endTime, double eLoss,
                          unsigned int startStatus,
                          unsigned int endStatus);

  std::vector<o2::itsmft::Hit>* getHits(Int_t iColl)
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  // Mandatory overrides
  void BeginPrimary() override { ; }
  void FinishPrimary() override { ; }
  void InitializeO2Detector() override;
  void PostTrack() override { ; }
  void PreTrack() override { ; }
  bool ProcessHits(FairVolume* v = nullptr) override;
  void EndOfEvent() override;
  void Register() override;
  void Reset() override;

  void createMaterials();
  void buildModules();

  enum EMedia {
    Scintillator,
  };

 private:
  Detector(const Detector&);
  Detector& operator=(const Detector&);

  std::vector<o2::itsmft::Hit>* mHits = nullptr;
  GeometryTGeo* mGeometryTGeo = nullptr;

  TGeoVolumeAssembly* buildModuleA();
  TGeoVolumeAssembly* buildModuleC();

  int mNumberOfRingsA;
  int mNumberOfRingsC;
  int mNumberOfSectors;
  float mDzScint;

  std::vector<float> mRingRadiiA;
  std::vector<float> mRingRadiiC;

  float mZmodA;
  float mZmodC;

  void defineSensitiveVolumes();

  /// Transient data about track passing the sensor, needed by ProcessHits()
  struct TrackData {               // this is transient
    bool mHitStarted;              //! hit creation started
    unsigned char mTrkStatusStart; //! track status flag
    TLorentzVector mPositionStart; //! position at entrance
    TLorentzVector mMomentumStart; //! momentum
    double mEnergyLoss;            //! energy loss
  } mTrackData;                    //!

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);

} // namespace fvd
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::fvd::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif
#endif
