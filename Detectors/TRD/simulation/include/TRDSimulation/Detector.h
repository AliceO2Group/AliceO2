// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_DETECTOR_H_
#define ALICEO2_TRD_DETECTOR_H_

#include "DetectorsBase/Detector.h"

class FairVolume;
class TClonesArray;

namespace o2
{
namespace trd
{
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

  void createMaterials();
  void ConstructGeometry() override;

 private:
  TClonesArray* mHitCollection; ///< Collection of TRD hits

  float mFoilDensity;
  float mGasNobleFraction;
  float mGasDensity;

  ClassDefOverride(Detector, 1)
};
} // end namespace trd
} // end global namespace
#endif
