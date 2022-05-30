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

#include <string>
#include <vector>
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/InputSpec.h"
#include "CommonUtils/NameConf.h"
#include "CTFWorkflow/CTFWriterSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"onlyDet", VariantType::String, std::string{DetID::NONE}, {"comma separated list of detectors to accept. Overrides skipDet"}});
  options.push_back(ConfigParamSpec{"skipDet", VariantType::String, std::string{DetID::NONE}, {"comma separate list of detectors to skip"}});
  options.push_back(ConfigParamSpec{"grpfile", VariantType::String, o2::base::NameConf::getGRPFileName(), {"name of the grp file"}});
  options.push_back(ConfigParamSpec{"no-grp", VariantType::Bool, false, {"do not read GRP file"}});
  options.push_back(ConfigParamSpec{"output-type", VariantType::String, "ctf", {"output types: ctf (per TF) or dict (create dictionaries) or both or none"}});
  options.push_back(ConfigParamSpec{"ctf-writer-verbosity", VariantType::Int, 0, {"verbosity level (0: summary per detector, 1: summary per block"}});
  options.push_back(ConfigParamSpec{"report-data-size-interval", VariantType::Int, 200, {"report sizes per detector for every N-th timeframe"}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
  std::swap(workflowOptions, options);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:CTF|ctf).*[W,w]riter.*"));
}

// ------------------------------------------------------------------
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  DetID::mask_t dets = 0;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  long run = 0;
  bool doCTF = true, doDict = false, dictPerDet = false;
  size_t szMin = 0, szMax = 0;
  std::string outType{}; // RS FIXME once global/local options clash is solved, --output-type will become device option
  if (!configcontext.helpOnCommandLine()) {
    bool noGRP = configcontext.options().get<bool>("no-grp");
    auto onlyDet = configcontext.options().get<std::string>("onlyDet");
    if (!noGRP) {
      std::unique_ptr<o2::parameters::GRPObject> grp(o2::parameters::GRPObject::loadFrom(configcontext.options().get<std::string>("grpfile")));
      dets = grp->getDetsReadOut(onlyDet, configcontext.options().get<std::string>("skipDet"));
      run = grp->getRun();
    } else {
      dets.set(); // by default read all
      auto mskOnly = DetID::getMask(configcontext.options().get<std::string>("onlyDet"));
      auto mskSkip = DetID::getMask(configcontext.options().get<std::string>("skipDet"));
      if (mskOnly.any()) {
        dets &= mskOnly;
      } else {
        dets ^= mskSkip;
      }
      run = 0;
    }
    if (dets.none()) {
      throw std::invalid_argument("Invalid workflow: no detectors found");
    }
    outType = configcontext.options().get<std::string>("output-type");
  }
  WorkflowSpec specs{o2::ctf::getCTFWriterSpec(dets, run, outType,
                                               configcontext.options().get<int>("ctf-writer-verbosity"),
                                               configcontext.options().get<int>("report-data-size-interval"))};
  return std::move(specs);
}
