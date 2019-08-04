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
#include "MIDSimulation/Hit.h"
#include "MIDSimulation/Stepper.h"

namespace o2
{
namespace mid
{

class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(bool active = true);

  void InitializeO2Detector() override;

  bool ProcessHits(FairVolume* vol = nullptr) override;

  void Register() override;

  std::vector<o2::mid::Hit>* getHits(int iColl);

  void Reset() override {}

  void ConstructGeometry() override;

  void EndOfEvent() override;

 private:
  void defineSensitiveVolumes();

  bool setHits(int iColl, std::vector<o2::mid::Hit>* ptr);

  o2::mid::Stepper mStepper; //! Stepper

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};

} // namespace mid
} // namespace o2

#endif
