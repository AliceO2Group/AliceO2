// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <string>
#include <vector>
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputSpec.h"
#include "DetectorsCommonDataFormats/NameConf.h"
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
  options.push_back(ConfigParamSpec{"dict-per-det", VariantType::Bool, false, {"create dictionary file per detector"}});
  options.push_back(ConfigParamSpec{"grpfile", VariantType::String, o2::base::NameConf::getGRPFileName(), {"name of the grp file"}});
  options.push_back(ConfigParamSpec{"no-grp", VariantType::Bool, false, {"do not read GRP file"}});
  options.push_back(ConfigParamSpec{"min-file-size", VariantType::Int64, 0l, {"accumulate CTFs until given file size reached"}});
  options.push_back(ConfigParamSpec{"max-file-size", VariantType::Int64, 0l, {"if > 0, avoid exceeding given file size in accumulation mode"}});
  options.push_back(ConfigParamSpec{"output-type", VariantType::String, "ctf", {"output types: ctf (per TF) or dict (create dictionaries) or both or none"}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  DetID::mask_t dets;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  long run = 0;
  bool doCTF = true, doDict = false, dictPerDet = false;
  size_t szMin = 0, szMax = 0;

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
    auto outmode = configcontext.options().get<std::string>("output-type");
    dictPerDet = configcontext.options().get<bool>("dict-per-det");
    if (outmode == "ctf") {
      doCTF = true;
      doDict = false;
    } else if (outmode == "dict") {
      doCTF = false;
      doDict = true;
    } else if (outmode == "both") {
      doCTF = true;
      doDict = true;
    } else if (outmode == "none") {
      doCTF = false;
      doDict = false;
    } else {
      throw std::invalid_argument("Invalid output-type");
    }
    szMin = configcontext.options().get<int64_t>("min-file-size");
    szMax = configcontext.options().get<int64_t>("max-file-size");
  }
  WorkflowSpec specs{o2::ctf::getCTFWriterSpec(dets, run, doCTF, doDict, dictPerDet, szMin, szMax)};
  return std::move(specs);
}
