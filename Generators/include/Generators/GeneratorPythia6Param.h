// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - January 2020

#ifndef ALICEO2_EVENTGEN_GENERATORPYTHIA6PARAM_H_
#define ALICEO2_EVENTGEN_GENERATORPYTHIA6PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to keep the settings of
 ** the pythia6 event-generator
 ** allow the user to modify them 
 **/
struct GeneratorPythia6Param : public o2::conf::ConfigurableParamHelper<GeneratorPythia6Param> {
  std::string config;
  int tune = 350;
  std::string frame = "CMS";
  std::string beam = "p";
  std::string target = "p";
  double win = 14000.;
  O2ParamDef(GeneratorPythia6Param, "Pythia6");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_GENERATORPYTHIA6PARAM_H_
