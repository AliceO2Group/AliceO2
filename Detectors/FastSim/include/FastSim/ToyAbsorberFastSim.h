// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FASTSIM_TOY_ABSORBER_H_
#define O2_FASTSIM_TOY_ABSORBER_H_

#include "FastSim/FastSimModel.h"

namespace o2::fastsim
{

/// A toy fast simulation model for the absorber: one particle out, continuing
/// along the incident direction with the energy exponentially attenuated over
/// the path through the envelope.
class ToyAbsorberFastSim : public FastSimModel
{
 public:
  using FastSimModel::FastSimModel;

 protected:
  std::vector<FastSimOutput> sample(const FastSimInput& input) const override;
};

} // namespace o2::fastsim

#endif // O2_FASTSIM_TOY_ABSORBER_H_
