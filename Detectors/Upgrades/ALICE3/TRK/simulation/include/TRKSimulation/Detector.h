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

#ifndef ALICEO2_TRK_DETECTOR_H
#define ALICEO2_TRK_DETECTOR_H

#include "DetectorsBase/Detector.h"
#include "TRKSimulation/Hit.h"

#include "TRKSimulation/TRKLayer.h"
#include "TRKSimulation/TRKServices.h"
#include "TRKSimulation/FT3Layer.h"
#include "TRKBase/GeometryTGeo.h"

#include <TLorentzVector.h>
#include <TString.h>

namespace o2
{
namespace trk
{

class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(bool active);
  Detector();
  Detector(const Detector& other);
  ~Detector();

  // Factory method
  static o2::base::Detector* create(bool active)
  {
    return new Detector(active);
  }

  void ConstructGeometry() override;

  o2::trk::Hit* addHit(int trackID, unsigned short detID, const TVector3& startPos, const TVector3& endPos,
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

  // Custom member functions
  std::vector<o2::trk::Hit>* getHits(int iColl) const
  {
    if (!iColl) {
      return mHits;
    }
    return nullptr;
  }

  void configMLOT();
  void configFT3ScopingV3();
  void configFromFile(std::string fileName = "alice3_TRK_layout.txt");
  void configToFile(std::string fileName = "alice3_TRK_layout.txt");

  void configServices(); // To get special conf from CLI options
  void createMaterials();
  void createGeometry();

  static constexpr int kForward = 0;
  static constexpr int kBackward = 1;

 private:
  int mNumberOfVolumes;
  int mNumberOfVolumesVD;

  // Transient data about track passing the sensor
  struct TrackData {
    bool mHitStarted;               // hit creation started
    unsigned char mTrkStatusStart;  // track status flag
    TLorentzVector mPositionStart;  // position at entrance
    TLorentzVector mMomentumStart;  // momentum
    double mEnergyLoss;             // energy loss
  } mTrackData;                     //! transient data
  GeometryTGeo* mGeometryTGeo;      //!
  std::vector<o2::trk::Hit>* mHits; // Derived from ITSMFT
  std::vector<std::unique_ptr<TRKCylindricalLayer>> mLayers;
  TRKServices mServices; // Houses the services of the TRK, but not the Iris tracker

  std::vector<std::string> mFirstOrLastLayers; // Names of the first or last layers
  bool InsideFirstOrLastLayer(std::string layerName);

  void defineSensitiveVolumes();

 protected:
  std::array<std::vector<TString>, 2> mFT3LayerName; // Two sets of layer (disc) names, one per direction (forward/backward)
  std::vector<int> mSensorID;       //! layer identifiers
  std::vector<TString> mSensorName; //! layer names
  std::array<std::vector<FT3Layer>, 2> mFT3Layers; // Two sets of layers (discs), one per direction (forward/backward)

 public:
  static constexpr Int_t sNumberVDPetalCases = 4;          //! Number of VD petals
  int getNumberOfLayers() const { return mLayers.size(); } //! Number of TRK layers
  int getNumberOfFT3Layers() const
  {
    if (mFT3LayerName[kBackward].size() != mFT3LayerName[kForward].size()) {
      LOG(fatal) << "Number of layers in the two directions are different! Returning 0.";
    }
    return mFT3LayerName[kBackward].size();
  }

  void Print(FairVolume* vol, int volume, int subDetID, int layer, int stave, int halfstave, int mod, int chip, int chipID) const;

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 2);
};
} // namespace trk
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::trk::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif
#endif
