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
//
// Design and equations: Nicola Nicassio nicola.nicassio@cern.ch
//

#ifndef ALICEO2_RICH_DETECTOR_H
#define ALICEO2_RICH_DETECTOR_H

#include "DetectorsBase/Detector.h"
#include "ITSMFTSimulation/Hit.h"

#include "RICHSimulation/RICHRing.h"
#include "RICHBase/GeometryTGeo.h"

#include <TLorentzVector.h>
#include <TString.h>

namespace o2
{
namespace rich
{

class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(bool active);
  Detector();
  ~Detector();

  void ConstructGeometry() override;

  o2::itsmft::Hit* addHit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
                          const TVector3& startMom, double startE, double endTime, double eLoss,
                          unsigned char startStatus, unsigned char endStatus);

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

  // Custom memer functions
  std::vector<o2::itsmft::Hit>* getHits(int iColl) const
  {
    if (!iColl) {
      return mHits;
    }
    return nullptr;
  }

  void configDefault();
  void configFromFile(std::string fileName = "alice3_RICH_layout.txt");
  void configToFile(std::string fileName = "alice3_RICH_layout.txt");

  void createMaterials();
  void createGeometry();
  void prepareEvenLayout();
  void prepareOddLayout();
  void createRings(TGeoVolume*);

 private:
  // Transient data about track passing the sensor
  struct TrackData {
    bool mHitStarted;              // hit creation started
    unsigned char mTrkStatusStart; // track status flag
    TLorentzVector mPositionStart; // position at entrance
    TLorentzVector mMomentumStart; // momentum
    double mEnergyLoss;            // energy loss
  } mTrackData;                    //! transient data

  GeometryTGeo* mGeometryTGeo;         //!
  std::vector<o2::itsmft::Hit>* mHits; // ITSMFT ones for the moment
  std::vector<o2::rich::Ring> mRings;

  void defineSensitiveVolumes();

 protected:
  int mNumberOfRings; // 21 for |eta|<1.82,  23 for eta|<2.00
  int mNTiles;
  // Vessel
  float mMaximumHalfLength = 350; // cm

  std::vector<double> mThetaBi;
  std::vector<double> mR0Tilt;
  std::vector<double> mZ0Tilt;
  std::vector<double> mLAerogelZ;
  std::vector<double> mTRplusG;
  std::vector<double> mMinRadialMirror;
  std::vector<double> mMaxRadialMirror;
  std::vector<double> mMaxRadialRadiator;
  std::vector<double> mVMirror1;
  std::vector<double> mVMirror2;
  std::vector<double> mVTile1;
  std::vector<double> mVTile2;
  std::vector<double> mR0Radiator;
  std::vector<double> mR0PhotoDet;
  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};
} // namespace rich
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::rich::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif
#endif