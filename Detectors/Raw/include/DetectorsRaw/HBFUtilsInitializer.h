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

// @brief Aux.class initialize HBFUtils
// @author ruben.shahoyan@cern.ch

#ifndef _O2_HBFUTILS_INITIALIZER_
#define _O2_HBFUTILS_INITIALIZER_

#include <vector>

namespace o2
{

namespace framework
{
class ConfigContext;
class DataProcessorSpec;
class ConfigParamSpec;
using WorkflowSpec = std::vector<DataProcessorSpec>;
} // namespace framework

namespace raw
{

struct HBFUtilsInitializer {
  static constexpr char HBFConfOpt[] = "hbfutils-config";

  HBFUtilsInitializer(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& wf);

  static void addConfigOption(std::vector<o2::framework::ConfigParamSpec>& opts);
};

} // namespace raw
} // namespace o2

#endif
