#include <iostream>

#include <G4Track.hh>
#include <G4PathFinder.hh>
#include <G4FieldTrack.hh>
#include <G4FieldTrackUpdator.hh>
#include <G4FastTrack.hh>
#include <G4FastStep.hh>
#include <G4SystemOfUnits.hh>

#include "FastSimBase/DirectTransport.h"

using namespace o2::base;

DirectTransport::DirectTransport(const G4String& name)
  : G4VFastSimulationModel(name), mDidDirectStep(false), mLastStep(1.)
{}

DirectTransport::DirectTransport(const G4String& name, G4Region* region)
  : G4VFastSimulationModel(name, region), mDidDirectStep(false), mLastStep(1.)
{}

G4bool DirectTransport::IsApplicable(const G4ParticleDefinition& aParticleType)
{
  return true;
}

G4bool DirectTransport::ModelTrigger(const G4FastTrack &)
{
  if(!mDidDirectStep || (mDidDirectStep && mLastStep > 0.01)) {
    std::cout << "IsApplicable, last step: " << mLastStep << std::endl;
    return true;
  }
  std::cout << "IsApplicable: NO" << std::endl;
  mDidDirectStep = false;
  return false;
}

void DirectTransport::DoIt(const G4FastTrack& aFastTrack, G4FastStep& aFastStep)
{
  auto track = *aFastTrack.GetPrimaryTrack();

  std::cout << track.GetPosition() << std::endl;

  auto pathFinder = G4PathFinder::GetInstance();
  std::cout << "Do direct transport" << std::endl;

  G4FieldTrack aFieldTrack('0');
  G4FieldTrackUpdator::Update( &aFieldTrack, &track );

  G4double retSafety = -1.0;
  ELimited retStepLimited;
  G4FieldTrack endTrack('a');
  G4double currentMinimumStep = DBL_MAX;

  mLastStep = pathFinder->ComputeStep(aFieldTrack, currentMinimumStep, 0,
                          aFastTrack.GetPrimaryTrack()->GetCurrentStepNumber(),
                          retSafety, retStepLimited, endTrack,
                          aFastTrack.GetPrimaryTrack()->GetVolume());

  //G4ThreeVector addStep(0.001, 0.001, 0.001);

  //addStep += endTrack.GetPosition();

  aFastStep.ProposePrimaryTrackFinalPosition(endTrack.GetPosition());

  std::cout << endTrack.GetPosition() << std::endl;
  std::cout << mLastStep << std::endl;
  mDidDirectStep = true;
}
