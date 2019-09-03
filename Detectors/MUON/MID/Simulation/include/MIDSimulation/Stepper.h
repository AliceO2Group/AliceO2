// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MID_SIMULATION_STEPPER_H
#define O2_MID_SIMULATION_STEPPER_H

#include "MIDSimulation/Hit.h"
#include "TVirtualMC.h"
#include <vector>

namespace o2
{
namespace mid
{

class Stepper
{
 public:
  Stepper();
  ~Stepper();
  bool process(const TVirtualMC& vmc);

  std::vector<o2::mid::Hit>* getHits() { return mHits; }
  void setHits(std::vector<o2::mid::Hit>* ptr) { mHits = ptr; }

  void resetHits();

  void registerHits(const char* branchName);

 private:
  void resetStep();

 private:
  float mTrackEloss{0.0};
  float mTrackLength{0.0};
  std::vector<o2::mid::Hit>* mHits{nullptr};
  Point3D<float> mEntrancePoint;
};

} // namespace mid
} // namespace o2

#endif
