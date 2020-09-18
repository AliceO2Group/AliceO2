// multiplicity trigger
//
//   usage: o2sim --trigger external --configKeyValues 'TriggerExternal.fileName=trigger_multiplicity.C;TriggerExternal.funcName="trigger_multiplicity(-0.8, 0.8, 100)"'
//

/// \author R+Preghenella - February 2020

#include "Generators/Trigger.h"
#include "TParticle.h"
#include "TParticlePDG.h"

o2::eventgen::Trigger
  trigger_multiplicity(double etaMin = -0.8, double etaMax = 0.8, int minNch = 100)
{
  auto trigger = [etaMin, etaMax, minNch](const std::vector<TParticle>& particles) -> bool {
    int nch = 0;
    for (const auto& particle : particles) {
      if (particle.GetStatusCode() != 1)
        continue;
      if (!particle.GetPDG())
        continue;
      if (particle.GetPDG()->Charge() == 0)
        continue;
      if (particle.Eta() < etaMin || particle.Eta() > etaMax)
        continue;
      nch++;
    }
    bool fired = nch >= minNch;
    return fired;
  };

  return trigger;
}
