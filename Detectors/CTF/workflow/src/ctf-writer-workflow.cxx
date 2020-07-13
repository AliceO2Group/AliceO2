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
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  DetID::mask_t dets;
  long run = 0;
  if (!configcontext.helpOnCommandLine()) {
    std::unique_ptr<o2::parameters::GRPObject> grp(o2::parameters::GRPObject::loadFrom(configcontext.options().get<std::string>("grpfile")));
    dets = grp->getDetsReadOut(configcontext.options().get<std::string>("onlyDet"), configcontext.options().get<std::string>("skipDet"));
    run = grp->getRun();
  }
  WorkflowSpec specs{o2::ctf::getCTFWriterSpec(dets, run)};
  return std::move(specs);
}
