// configures a trigger class
//   usage: o2sim --trigger external --extTrgFile multiphi_trigger.C
// options:                          --extTrgFunc multiphi_trigger(2)

/// \author R+Preghenella - January 2020

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "Generators/Trigger.h"
#include "TClonesArray.h"
#include "TParticle.h"
#endif

#include "FairLogger.h"

/** class definition **/

class MultiPhiTrigger : public o2::eventgen::Trigger
{
 public:
  // constructor, destructor
  MultiPhiTrigger() = default;
  ~MultiPhiTrigger() = default;

  // method to override
  Bool_t fired(TClonesArray* particles) override;

  // setters
  void setNumberOfPhis(Int_t val) { mNumberOfPhis = val; };

 private:
  Int_t mMaxNumberOfAttempts = 1000000;
  Int_t mWarnNumberOfAttempts = 10000;
  Int_t mNumberOfAttempts = 0;
  Int_t mNumberOfPhis = 2;
};

/** class implementation **/

Bool_t MultiPhiTrigger::fired(TClonesArray* particles)
{
  // loop over generated particles
  Int_t nParticles = particles->GetEntries();
  TParticle* particle = nullptr;
  Int_t nPhis = 0;
  for (Int_t iparticle = 0; iparticle < nParticles; iparticle++) {
    particle = (TParticle*)particles->At(iparticle);
    if (!particle)
      continue;
    // count phi mesons at mid rapidity with pt > 500 MeV/c
    if (particle->GetPdgCode() == 333 && fabs(particle->Y()) < 0.5 && particle->Pt() > 0.5)
      nPhis++;
  }

  // increment number of attempts
  mNumberOfAttempts++;

  // trigger fired
  if (nPhis >= mNumberOfPhis) {
    LOG(INFO) << "MultiPhiTrigger fired: " << nPhis << " phi mesons";
    mNumberOfAttempts = 0;
    return kTRUE;
  }

  // warning message
  if (mNumberOfAttempts % mWarnNumberOfAttempts == 0)
    LOG(WARNING) << "MultiPhiTrigger did not fire yet: " << mNumberOfAttempts << " number of attempts";

  // fatal message
  if (mNumberOfAttempts >= mMaxNumberOfAttempts)
    LOG(FATAL) << "MultiPhiTrigger did not fire after " << mMaxNumberOfAttempts;

  // trigger did not fire
  return kFALSE;
};

/** main function **/

o2::eventgen::Trigger*
  multiphi_trigger(Int_t numberOfPhis = 2)
{
  auto trigger = new MultiPhiTrigger();
  trigger->setNumberOfPhis(numberOfPhis);
  return trigger;
}
