// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author ruben.shahoyan@cern.ch
/// \brief params for ConfigurableParam

#ifndef COMMON_CONFIGURABLE_KEYVAL_PARAM_H_
#define COMMON_CONFIGURABLE_KEYVAL_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace conf
{
struct KeyValParam : public o2::conf::ConfigurableParamHelper<KeyValParam> {
  std::string input_dir = "none";
  std::string output_dir = "none";

  O2ParamDef(KeyValParam, "keyval");
};
} // namespace conf
} // namespace o2

#endif
