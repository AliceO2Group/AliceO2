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

#include "FastSim/ToyAbsorberFastSim.h"

#include <cmath>

namespace o2::fastsim
{

namespace
{
/// Attenuation length of the toy transformation.
constexpr double kAbsorptionLengthCm = 60.;
} // namespace

//_____________________________________________________________________________
std::vector<FastSimOutput> ToyAbsorberFastSim::sample(const FastSimInput& input) const
{
  // A toy transformation, not a physics model: the incident particle carries on
  // in its direction with the energy attenuated over the path through the
  // envelope. A trained model returns a shower here instead.
  // Always positive: ModelTrigger only calls a model above its threshold, and
  // an exponential of a finite path cannot reach zero.
  const double kinetic = input.kineticEnergy * std::exp(-input.exitDistance / kAbsorptionLengthCm);
  const double momentum = std::sqrt(kinetic * (kinetic + 2. * input.mass));

  FastSimOutput out;
  out.pdg = input.pdg;
  out.time = input.time;
  for (int i = 0; i < 3; ++i) {
    out.position[i] =
      input.position[i] + (input.exitDistance + kSurfaceEpsilonCm) * input.direction[i];
    out.momentum[i] = momentum * input.direction[i];
  }
  return {out};
}

} // namespace o2::fastsim
