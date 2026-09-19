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
/// that leave it. `DoIt()` is Geant4's own entry point
/// (`G4VFastSimulationModel::DoIt`); the implementation here wraps the logic
/// common to every model and delegates the physics to `sample()`. A model that
/// needs a different shape can still override `DoIt()`.

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
  double exitDistance = 0.;  ///< cm from `position` to the ENVELOPE surface along `direction`
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

/// Base class for fast simulation models.
///
/// The model is attached to regions (see G4FastSimulation.h) purely so that
/// Geant4 consults it; what it encloses is the ENVELOPE VOLUME named below,
/// which is normally the mother volume of a whole module. The two are
/// deliberately separate, because a Geant4 region in O2 can only ever be "every
/// volume of a given material" -- the VMC special cuts make every logical volume
/// a root of its own material's region, and Geant4 stops propagating a region at
/// any such daughter. So `G4FastTrack::GetEnvelopeSolid()` would hand back one
/// absorber piece rather than the absorber, and this class does not use it.
///
/// Containment and the exit distance are taken from the track's own touchable,
/// which already carries the full ancestry and the transform of every level.
class FastSimModel : public G4VFastSimulationModel
{
 public:
  FastSimModel(const G4String& name, const G4String& envelopeVolume, double minEnergyGeV);

  G4bool IsApplicable(const G4ParticleDefinition& particle) override;
  G4bool ModelTrigger(const G4FastTrack& fastTrack) override;

  /// Wraps the common logic of a fast simulation action and delegates the
  /// physics to `sample()`: it measures the distance to the envelope surface,
  /// kills the incident particle, stacks what `sample()` returned and books the
  /// energy difference as a deposit. Override it for a model that does not fit
  /// that shape.
  void DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) override;

 protected:
  /// Given the particle that entered, return everything that leaves. This is
  /// the function a trained model implements.
  virtual std::vector<FastSimOutput> sample(const FastSimInput& input) const = 0;

 private:
  /// The track's ancestry level at which the envelope volume sits, or -1 when
  /// the track is not inside it at all.
  int envelopeDepth(const G4Track* track) const;

  G4String mEnvelope;     ///< logical volume the model encloses
  double mMinEnergy = 0.; ///< internal Geant4 units; below this the detailed transport runs
  mutable bool mWarned = false;
};

} // namespace o2::fastsim

#endif // O2_FASTSIM_MODEL_H_
