// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_SIMULATION_DETECTOR_H
#define O2_MCH_SIMULATION_DETECTOR_H

#include "DetectorsBase/Detector.h"
#include "MCHSimulation/Hit.h"
#include <vector>
#include <memory>

namespace o2
{
namespace mch
{

class Stepper;

class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(bool active = true);

  ~Detector() override;

  Bool_t ProcessHits(FairVolume* v = nullptr) override;

  void InitializeO2Detector() override;

  void Register() override;

  void Reset() override {}

  void ConstructGeometry() override;

  std::vector<o2::mch::Hit>* getHits(int);

  void EndOfEvent() override;

 private:
  /// copy constructor (used in MT)
  Detector(const Detector& rhs);

  void defineSensitiveVolumes();

  bool setHits(int i, std::vector<o2::mch::Hit>* ptr);

  o2::mch::Stepper* mStepper{nullptr}; //!

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};

} // namespace mch
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::mch::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif
