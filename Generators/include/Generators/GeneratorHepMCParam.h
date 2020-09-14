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

#ifndef ALICEO2_EVENTGEN_GENERATORHEPMCPARAM_H_
#define ALICEO2_EVENTGEN_GENERATORHEPMCPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string>

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to keep the settings of
 ** the HepMC event generator and
 ** allow the user to modify them 
 **/

struct GeneratorHepMCParam : public o2::conf::ConfigurableParamHelper<GeneratorHepMCParam> {
  std::string fileName = "";
  int version = 2;
  O2ParamDef(GeneratorHepMCParam, "HepMC");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_GENERATORHEPMCPARAM_H_
