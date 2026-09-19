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

#ifndef O2_FASTSIM_MODEL_H_
#define O2_FASTSIM_MODEL_H_

/// Base class for fast simulation models.
///
/// A fast simulation model replaces the detailed transport through a region of
/// the geometry by a function from the particle that enters it to the particles
/// that leave it. `DoIt()` below is the plumbing and is the same for every
/// model; a model implements `sample()` and nothing else.

#include "G4VFastSimulationModel.hh"

#include <vector>

class G4FastStep;
class G4FastTrack;
class G4ParticleDefinition;

namespace o2::fastsim
{

/// The particle entering the envelope, plus the geometric context a model needs.
/// Units are the O2/VMC ones: cm, GeV, ns.
struct FastSimInput {
  int pdg = 0;
  double position[3] = {};   ///< global, on the envelope surface
  double direction[3] = {};  ///< unit vector
  double kineticEnergy = 0.; ///< GeV
  double mass = 0.;          ///< GeV
  double time = 0.;          ///< ns
  double exitDistance = 0.;  ///< cm from `position` to the envelope surface along `direction`
};

/// One particle leaving the envelope.
struct FastSimOutput {
  int pdg = 0;
  double position[3] = {}; ///< global; put it outside the envelope surface
  double momentum[3] = {}; ///< GeV/c
  double time = 0.;        ///< ns
};

/// A secondary created exactly on the envelope surface is located by the
/// navigator in whichever daughter owns that point, which costs two extra
/// zero-length steps before it gets out. Models should emit just beyond it.
constexpr double kSurfaceEpsilonCm = 1e-5;

class FastSimModel : public G4VFastSimulationModel
{
 public:
  FastSimModel(const G4String& name, double minEnergyGeV);

  G4bool IsApplicable(const G4ParticleDefinition& particle) override;
  G4bool ModelTrigger(const G4FastTrack& fastTrack) override;

  /// Measures the distance to the envelope surface, asks `sample()` what comes
  /// out, kills the incident particle, stacks the result and books the energy
  /// difference as a deposit.
  void DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) final;

 protected:
  /// Given the particle that entered, return everything that leaves. This is
  /// the function a trained model implements.
  virtual std::vector<FastSimOutput> sample(const FastSimInput& input) const = 0;

 private:
  double mMinEnergy = 0.; ///< internal Geant4 units; below this the detailed transport runs
};

} // namespace o2::fastsim

#endif // O2_FASTSIM_MODEL_H_
