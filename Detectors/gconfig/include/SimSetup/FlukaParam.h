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

/// \author A+Morsch

#include <string>
#ifndef ALICEO2_EVENTGEN_FLUKAPARAM_H_
#define ALICEO2_EVENTGEN_FLUKAPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
/**
 ** A parameter class/struct holding values for
 ** FLUKA Transport Code
 **/
struct FlukaParam : public o2::conf::ConfigurableParamHelper<FlukaParam> {
  bool activationSimulation = false; // whether FLUKA is used for activation studies
  bool lowNeutron = false;           // switch for low energy neutron transport
  bool userStepping = true;          // switch for hit scoring
  float activationHadronCut = 0.003; // hadron kinetic energy cut for activation studies
  std::string scoringFile = "";      // input file for user scoring options

  O2ParamDef(FlukaParam, "FlukaParam");
};
} // end namespace o2

#endif // ALICEO2_EVENTGEN_FLUKAPARAM_H_
