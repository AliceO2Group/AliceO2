// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_SIMULATION_STEPPER_H
#define O2_MCH_SIMULATION_STEPPER_H

#include "MCHSimulation/Hit.h"
#include "TVirtualMC.h"
#include <iostream>
#include <vector>
#include <array>

namespace o2
{
namespace mch
{

class Stepper
{
 public:
  Stepper();
  ~Stepper();
  void process(const TVirtualMC& vmc);

  std::vector<o2::mch::Hit>* getHits() { return mHits; }
  void setHits(std::vector<o2::mch::Hit>* ptr) { mHits = ptr; }

  void resetHits();

  void registerHits(const char* branchName);

 private:
  void resetStep();

 private:
  float mTrackEloss{0.0};
  float mTrackLength{0.0};
  std::vector<o2::mch::Hit>* mHits{nullptr};
  Point3D<float> mEntrancePoint;
};

} // namespace mch
} // namespace o2

#endif
