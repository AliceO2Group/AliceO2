// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_DETECTOR_H_
#define ALICEO2_TRD_DETECTOR_H_

#include <TClonesArray.h>
#include <vector>
#include "DetectorsBase/Detector.h"
#include "SimulationDataFormat/BaseHits.h"

class FairVolume;

namespace o2
{
namespace trd
{
class TRDGeometry;

// define TRD hit type
using HitType = o2::BasicXYZEHit<float>;

class Detector : public o2::Base::Detector
{
 public:
  Detector() = default;

  Detector(const char* Name, bool Active);

  ~Detector() override = default;

  void Initialize() override;

  bool ProcessHits(FairVolume* v = nullptr) override;

  void Register() override;

  TClonesArray* GetCollection(int iColl) const final;

  void Reset() override;
  void EndOfEvent() override;

  void createMaterials();
  void ConstructGeometry() override;

 private:
  // defines/sets-up the sensitive volumes
  void defineSensitiveVolumes();

  // addHit
  template <typename T>
  void addHit(T x, T y, T z, T time, T energy, int trackId, int detId);

  TClonesArray* mHitCollection; ///< Collection of TRD hits

  float mFoilDensity;
  float mGasNobleFraction;
  float mGasDensity;

  TRDGeometry* mGeom = nullptr;

  ClassDefOverride(Detector, 1)
};

template <typename T>
void Detector::addHit(T x, T y, T z, T time, T energy, int trackId, int detId)
{
  TClonesArray& clref = *mHitCollection;
  Int_t size = clref.GetEntriesFast();
  // create hit in-place
  new (clref[size]) HitType(x, y, z, time, energy, trackId, detId);
}

} // end namespace trd
} // end global namespace
#endif
