// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_SIMCONFIG_G4PARAM_H_
#define O2_SIMCONFIG_G4PARAM_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace conf
{

// enumerating the possible G4 physics settings
enum class EG4Physics {
  kFTFP_BERT_optical = 0,         /* just ordinary */
  kFTFP_BERT_optical_biasing = 1, /* with biasing enabled */
  kFTFP_INCLXX_optical = 2        /* special INCL++ version */
};

// parameters to influence the G4 engine
struct G4Params : public o2::conf::ConfigurableParamHelper<G4Params> {
  EG4Physics physicsmode = EG4Physics::kFTFP_BERT_optical; // physics mode with which to configure G4
  std::string const& getPhysicsConfigString() const;

  O2ParamDef(G4Params, "G4");
};

} // namespace conf
} // namespace o2

#endif /* O2_SIMCONFIG_G4PARAM_H_ */
