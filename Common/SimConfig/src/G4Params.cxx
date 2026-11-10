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

#include "SimConfig/G4Params.h"
O2ParamImpl(o2::conf::G4Params);

namespace o2
{
namespace conf
{

namespace
{
static const std::string confstrings[8] = {"FTFP_BERT+optical",
                                           "FTFP_BERT+optical+biasing",
                                           "FTFP_INCLXX+optical",
                                           "FTFP_BERT_HP+optical",
                                           "FTFP_BERT_EMV+optical",
                                           "FTFP_BERT_EMV+optical+biasing",
                                           "FTFP_INCLXX_EMV+optical",
                                           "FTFP_BERT_HP_EMV+optical"};
}

std::string G4Params::getPhysicsConfigString() const
{
  std::string conf = physicsmode == o2::conf::EG4Physics::kUSER ? userPhysicsList : confstrings[(int)physicsmode];
  if (useG4HepEm && conf.find("_HEP") == std::string::npos) {
    // geant4_vmc selects G4HepEm with the suffix _HEP on the hadronic list, eg. FTFP_BERT_EMV_HEP+optical
    const auto pos = conf.find('+');
    conf.insert(pos == std::string::npos ? conf.size() : pos, "_HEP");
  }
  return conf;
}

} // namespace conf
} // namespace o2
