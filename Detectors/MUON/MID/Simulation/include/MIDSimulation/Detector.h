// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MID_SIMULATION_DETECTOR_H
#define O2_MID_SIMULATION_DETECTOR_H

#include <vector>
#include "DetectorsBase/Detector.h"
#include "SimulationDataFormat/BaseHits.h"

namespace o2
{
namespace mid
{

using HitType = o2::BasicXYZEHit<float>;

class Detector : public o2::Base::DetImpl<Detector>
{
 public:
  Detector(bool active = true);

  void InitializeO2Detector() override;

  bool ProcessHits(FairVolume* v = nullptr) override;

  void Register() override {}

  std::vector<HitType>* getHits(int i) const
  {
    //if (i == 0) {
    //      return mHits;
    //}
    return nullptr;
  }

  void Reset() override {}

  void ConstructGeometry() override;

 private:
  void defineSensitiveVolumes();

  std::vector<HitType>* mHits = nullptr;

  template <typename Det>
  friend class o2::Base::DetImpl;
  ClassDefOverride(Detector, 1)
};

} // namespace mid
} // namespace o2

#endif
