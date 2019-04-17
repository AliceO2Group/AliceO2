#ifndef ALICEO2_FASTSIM_BASE_DIRECTTRANSPORT_H_
#define ALICEO2_FASTSIM_BASE_DIRECTTRANSPORT_H_

#include <G4VFastSimulationModel.hh>

namespace o2
{
namespace base
{

/// This fast sim model just pushes a track through a FairModule repspecting
/// external fields as well
class DirectTransport : public G4VFastSimulationModel
{
  public:

    DirectTransport(const G4String& name);
    DirectTransport(const G4String& name, G4Region* region);

    G4bool IsApplicable(const G4ParticleDefinition& aParticleType) final;
    G4bool ModelTrigger(const G4FastTrack& aFastTrack) final;
    void DoIt(const G4FastTrack& aFastTrack, G4FastStep& aFastStep) final;

  private:
    DirectTransport(const DirectTransport&);
    DirectTransport& operator=(const DirectTransport&);

  private:
    G4bool mDidDirectStep;
    G4double mLastStep;
};

} // namespace base
} // namespace o2

#endif /* ALICEO2_FASTSIM_BASE_DIRECTTRANSPORT_H_ */
