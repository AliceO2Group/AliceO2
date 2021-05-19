// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "AODProducerWorkflow/AODProducerWorkflowSpec.h"
#include "Framework/CompletionPolicy.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DetectorsCommonDataFormats/DetID.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  bool useMC = true;

  GID::mask_t src = GID::getSourcesMask("ITS,MFT,TPC,ITS-TPC,ITS-TPC-TOF,TPC-TOF");
  src |= GID::getSourceMask(GID::FT0);
  GID::mask_t dummy, srcClus = GID::includesDet(DetID::TOF, src) ? GID::getSourceMask(GID::TOF) : dummy;

  WorkflowSpec specs;

  specs.emplace_back(o2::aodproducer::getAODProducerWorkflowSpec(src));

  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcClus, src, src, useMC, srcClus);
  o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, specs, useMC);

  return std::move(specs);
}
