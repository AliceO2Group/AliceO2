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

#ifndef O2_CONF_DIGIPARAMS_H_
#define O2_CONF_DIGIPARAMS_H_

// Global parameters for digitization

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "CommonUtils/NameConf.h"
#include <string>

namespace o2
{
namespace conf
{

// Global parameters for digitization
struct DigiParams : public o2::conf::ConfigurableParamHelper<DigiParams> {

  std::string ccdb = o2::base::NameConf::getCCDBServer();
  std::string digitizationgeometry = "";              // with with geometry file to digitize -> leave empty as this needs to be filled by the digitizer workflow
  std::string grpfile = "";                           // which GRP file to use --> leave empty as this needs to be filled by the digitizer workflow
  bool mctruth = true;                                // whether to create labels

  O2ParamDef(DigiParams, "DigiParams");
};

} // namespace conf
} // namespace o2

#endif
