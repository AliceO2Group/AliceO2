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

/// \author M. Giacalone - October 2024

#ifndef ALICEO2_EVENTGEN_GENERATORHYBRIDPARAM_H_
#define ALICEO2_EVENTGEN_GENERATORHYBRIDPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to keep the settings of
 ** the Hybrid event generator and
 ** allow the user to modify them
 **/

struct GeneratorHybridParam : public o2::conf::ConfigurableParamHelper<GeneratorHybridParam> {
  std::string configFile = "";    // JSON configuration file for the generators
  bool randomize = false;         // randomize the order of the generators, if not generator using fractions
  int num_workers = 1;            // number of threads available for asyn/parallel event generation
  bool switchExtToHybrid = false; // force external generator to be executed as hybrid mode, useful for Hyperloop MCGEN
  O2ParamDef(GeneratorHybridParam, "GeneratorHybrid");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_GENERATORHYBRIDPARAM_H_
