// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CONF_DIGIPARAMS_H_
#define O2_CONF_DIGIPARAMS_H_

// Global parameters for digitization

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"
#include <string>

namespace o2
{
namespace conf
{
// Global parameters for digitization

struct DigiParams : public o2::conf::ConfigurableParamHelper<DigiParams> {

  std::string ccdb = "http://ccdb-test.cern.ch:8080"; // URL for CCDB acces

  O2ParamDef(DigiParams, "DigiParams");
};

} // namespace conf
} // namespace o2

#endif
