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

#ifndef ALICEO2_EVENTGEN_EVENTDECAYERPARAM_H_
#define ALICEO2_EVENTGEN_EVENTDECAYERPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string>

namespace o2
{
namespace eventgen
{

/// Settings of the EventDecayer, which decays generator particles with Pythia8 before transport
struct EventDecayerParam : public o2::conf::ConfigurableParamHelper<EventDecayerParam> {
  std::string config[8] = {"", "", "", "", "", "", "", ""};                          // Pythia8 config files, an empty slot takes DecayerPythia8.config
  double tau0Max = 10.;                                                              // species with a larger tau0 in mm/c stay undecayed (<= 0 switches it off)
  std::string skipPdg = "115 225 335 10221 10331 20223 20333 102132 102134 9010221"; // PDG codes never decayed, matched on |pdg|
  std::string g4ParticleTable = "";                                                  // Geant4 particle table from o2-sim-g4-particle-table
  int seed = 0;                                                                      // Pythia8 random seed (0 derives it from gRandom)
  bool verbose = false;                                                              // list every Pythia8 decay
  bool showChanged = false;                                                          // show changed particle data at init
  O2ParamDef(EventDecayerParam, "EventDecayer");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_EVENTDECAYERPARAM_H_
