// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "RawFileReaderWorkflow.h"
#include <string>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"conf", o2::framework::VariantType::String, "", {"configuration file to init from (obligatory)"}});
  workflowOptions.push_back(ConfigParamSpec{"loop", o2::framework::VariantType::Int, 1, {"loop N times (infinite for N<0)"}});
  workflowOptions.push_back(ConfigParamSpec{"message-per-tf", o2::framework::VariantType::Bool, false, {"send TF of each link as a single FMQ message rather than multipart with message per HB"}});
  workflowOptions.push_back(ConfigParamSpec{"output-per-link", o2::framework::VariantType::Bool, false, {"send message per Link rather than per FMQ output route"}});
  workflowOptions.push_back(ConfigParamSpec{"delay", o2::framework::VariantType::Float, 0.f, {"delay in seconds between consecutive TFs sending"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto inifile = configcontext.options().get<std::string>("conf");
  auto loop = configcontext.options().get<int>("loop");
  auto tfAsMessage = configcontext.options().get<bool>("message-per-tf");
  auto outPerRoute = !configcontext.options().get<bool>("output-per-link");
  uint32_t delay_us = uint32_t(1e6 * configcontext.options().get<int>("delay")); // delay in microseconds
  return std::move(o2::raw::getRawFileReaderWorkflow(inifile, tfAsMessage, outPerRoute, loop, delay_us));
}
