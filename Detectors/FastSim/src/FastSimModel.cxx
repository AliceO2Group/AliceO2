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

#include "FastSim/FastSimModel.h"

#include <fairlogger/Logger.h>

#include <G4DynamicParticle.hh>
#include <G4FastStep.hh>
#include <G4FastTrack.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>
#include <G4VSolid.hh>

#include <algorithm>

namespace o2::fastsim
{

//_____________________________________________________________________________
FastSimModel::FastSimModel(const G4String& name, double minEnergyGeV)
  : G4VFastSimulationModel(name), mMinEnergy(minEnergyGeV * CLHEP::GeV)
{
}

//_____________________________________________________________________________
G4bool FastSimModel::IsApplicable(const G4ParticleDefinition&)
{
  // Which particles a model sees is decided by the `setParticles` selection,
  // not here.
  return true;
}

//_____________________________________________________________________________
G4bool FastSimModel::ModelTrigger(const G4FastTrack& fastTrack)
{
  // Below the threshold the detailed transport is cheap and a surrogate would
  // be extrapolating.
  return fastTrack.GetPrimaryTrack()->GetKineticEnergy() > mMinEnergy;
}

//_____________________________________________________________________________
void FastSimModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep)
{
  const G4Track* track = fastTrack.GetPrimaryTrack();
  const G4ThreeVector& position = track->GetPosition();
  const G4ThreeVector& direction = track->GetMomentumDirection();

  FastSimInput input;
  input.pdg = track->GetDefinition()->GetPDGEncoding();
  input.position[0] = position.x() / CLHEP::cm;
  input.position[1] = position.y() / CLHEP::cm;
  input.position[2] = position.z() / CLHEP::cm;
  input.direction[0] = direction.x();
  input.direction[1] = direction.y();
  input.direction[2] = direction.z();
  input.kineticEnergy = track->GetKineticEnergy() / CLHEP::GeV;
  input.mass = track->GetDefinition()->GetPDGMass() / CLHEP::GeV;
  input.time = track->GetGlobalTime() / CLHEP::ns;
  input.exitDistance = fastTrack.GetEnvelopeSolid()->DistanceToOut(
                         fastTrack.GetPrimaryTrackLocalPosition(),
                         fastTrack.GetPrimaryTrackLocalDirection()) /
                       CLHEP::cm;

  const std::vector<FastSimOutput> outgoing = sample(input);

  fastStep.KillPrimaryTrack();
  fastStep.ProposePrimaryTrackPathLength(input.exitDistance * CLHEP::cm);

  // NOTE: a fast step defaults to AvoidHitInvocation, so Geant4 does not call
  // the sensitive detector and TVirtualMCApplication::Stepping() is not invoked
  // for it. For a passive envelope that is what we want -- there are no hits to
  // lose, and the steps disappearing from the step log is the saving. A model
  // covering a region that scores would add
  //   fastStep.ProposeSteppingControl(NormalCondition);
  // here.

  double outgoingKineticEnergy = 0.;
  fastStep.SetNumberOfSecondaryTracks(outgoing.size());
  for (const auto& out : outgoing) {
    const G4ParticleDefinition* definition =
      G4ParticleTable::GetParticleTable()->FindParticle(out.pdg);
    if (definition == nullptr) {
      LOG(error) << "fast simulation: model " << GetName() << " returned unknown pdg " << out.pdg
                 << "; particle dropped";
      continue;
    }
    const G4ThreeVector momentum(out.momentum[0] * CLHEP::GeV, out.momentum[1] * CLHEP::GeV,
                                 out.momentum[2] * CLHEP::GeV);
    G4DynamicParticle particle(definition, momentum);
    outgoingKineticEnergy += particle.GetKineticEnergy();
    fastStep.CreateSecondaryTrack(particle,
                                  G4ThreeVector(out.position[0] * CLHEP::cm,
                                                out.position[1] * CLHEP::cm,
                                                out.position[2] * CLHEP::cm),
                                  out.time * CLHEP::ns, /*localCoordinates=*/false);
  }

  // Whatever did not come out stayed in.
  fastStep.ProposeTotalEnergyDeposited(
    std::max(0., track->GetKineticEnergy() - outgoingKineticEnergy));
}

} // namespace o2::fastsim
