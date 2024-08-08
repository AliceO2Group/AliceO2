// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_SIMCONFIG_G4PARAM_H_
#define O2_SIMCONFIG_G4PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace conf
{

// enumerating the possible G4 physics settings
enum class EG4Physics {
  kFTFP_BERT_optical = 0,             /* just ordinary */
  kFTFP_BERT_optical_biasing = 1,     /* with biasing enabled */
  kFTFP_INCLXX_optical = 2,           /* special INCL++ version */
  kFTFP_BERT_HP_optical = 3,          /* enable low energy neutron transport */
  kFTFP_BERT_EMV_optical = 4,         /* just ordinary with faster electromagnetic physics */
  kFTFP_BERT_EMV_optical_biasing = 5, /* with biasing enabled with faster electromagnetic physics */
  kFTFP_INCLXX_EMV_optical = 6,       /* special INCL++ version */
  kFTFP_BERT_EMV_HP_optical = 7,      /* enable low energy neutron transport */
  kUSER = 8                           /* allows to give own string combination */
};

// parameters to influence the G4 engine
struct G4Params : public o2::conf::ConfigurableParamHelper<G4Params> {
  EG4Physics physicsmode = EG4Physics::kFTFP_BERT_EMV_optical; // default physics mode with which to configure G4

  std::string configMacroFile = ""; // a user provided g4Config.in file (otherwise standard one fill be taken)
  std::string userPhysicsList = ""; // possibility to directly give physics list as string

  std::string const& getPhysicsConfigString() const;

  O2ParamDef(G4Params, "G4");
};

} // namespace conf
} // namespace o2

#endif /* O2_SIMCONFIG_G4PARAM_H_ */
