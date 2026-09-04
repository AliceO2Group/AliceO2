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

#ifndef ALICEO2_TRD_DETECTOR_H_
#define ALICEO2_TRD_DETECTOR_H_

#include <vector>
#include "DetectorsBase/Detector.h"
#include "SimulationDataFormat/BaseHits.h"
#include "CommonUtils/ShmAllocator.h"

#include "DataFormatsTRD/Hit.h"

class FairVolume;

namespace o2
{
namespace trd
{

class Geometry;
class TRsim;

class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(Bool_t active = true);
  ~Detector() override;
  void InitializeO2Detector() override;
  bool ProcessHits(FairVolume* v = nullptr) override;
  void Register() override;
  std::vector<Hit>* getHits(int iColl) const
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }
  void FinishEvent() override;
  void Reset() override;
  void EndOfEvent() override;
  void createMaterials();
  void ConstructGeometry() override;
  /// Add alignable top volumes
  void addAlignableVolumes() const override;

 private:
  /// copy constructor (used in MT)
  Detector(const Detector& rhs);

  void InitializeParams();

  // defines/sets-up the sensitive volumes
  void defineSensitiveVolumes();

  // Fills the volume-id lookup tables below; called once from InitializeO2Detector().
  void buildVolumeIdTables();

  // What a sensitive volume is, in mRegionByVolId
  enum Region : int8_t { kNotSensitive = 0,
                         kDrift = 1,
                         kAmplification = 2 };

  // addHit
  template <typename T>
  void addHit(T x, T y, T z, T locC, T locR, T locT, T tof, int charge, int trackId, int detId, bool drift = false);

  // Create TR hits
  void createTRhit(int);

  std::vector<Hit>* mHits = nullptr; ///!< Collection of TRD hits

  float mFoilDensity;
  float mGasNobleFraction;
  float mGasDensity;

  float mMaxMCStepDef;

  bool mTRon; // Switch for TR simulation
  TRsim* mTR; // Access to TR simulation

  float mWion; // Ionization potential

  Geometry* mGeom = nullptr;

  // Volume-id lookup tables, resolved once at initialisation so that ProcessHits does
  // integer indexing instead of an sscanf on a volume name at every step. Volume ids are
  // small and dense, so a flat vector beats a map here.
  std::vector<int8_t> mRegionByVolId;  //!< drift / amplification / not sensitive, by volume id
  std::vector<int8_t> mChamberByVolId; //!< chamber within the supermodule (0..29), -1 elsewhere
  std::vector<int8_t> mSectorByVolId;  //!< supermodule (0..17), -1 elsewhere

  // How far above a sensitive volume the chamber and the supermodule sit. This depends on
  // how the transport engine represents the hierarchy -- a native-Geant4 conversion flattens
  // the chamber assembly away -- so it is resolved from the tables on the first hit rather
  // than hard-coded.
  int mChamberOffset = -1; //!
  int mSectorOffset = -1;  //!

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};

template <typename T>
void Detector::addHit(T x, T y, T z, T locC, T locR, T locT, T tof, int charge, int trackId, int detId, bool drift)
{
  mHits->emplace_back(x, y, z, locC, locR, locT, tof, charge, trackId, detId, drift);
}

} // namespace trd
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::trd::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif
#endif
