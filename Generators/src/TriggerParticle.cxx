// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - September 2019

#include "Generators/TriggerParticle.h"
#include "TClonesArray.h"
#include "TParticle.h"

namespace o2
{
namespace eventgen
{

Trigger TriggerParticle(const TriggerParticleParam& param)
{
  LOG(INFO) << "Init trigger \'particle\' with following parameters";
  LOG(INFO) << param;
  return [&param](const std::vector<TParticle>& particles) -> bool {
    for (const auto& particle : particles) {
      if (particle.GetPdgCode() != param.pdg) {
        continue;
      }
      if (particle.Pt() < param.ptMin || particle.Pt() > param.ptMax) {
        continue;
      }
      if (particle.Eta() < param.etaMin || particle.Eta() > param.etaMax) {
        continue;
      }
      if (particle.Phi() < param.phiMin || particle.Phi() > param.phiMax) {
        continue;
      }
      if (particle.Y() < param.yMin || particle.Y() > param.yMax) {
        continue;
      }
      return true; /** trigger fired **/
    }
    return false; /** trigger did not fire **/
  };
}

} /* namespace eventgen */
} /* namespace o2 */
