// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ConfigParamSpec.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using namespace o2::dataformats;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable visualization of MC data"}},
    {"track-types", VariantType::String, std::string{GlobalTrackID::ALL}, {"comma-separated list of track sources to read"}},
    {"cluster-types", VariantType::String, std::string{GlobalTrackID::ALL}, {"comma-separated list of cluster sources to read"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable reading root files, essentially making this workflow void, but needed for compatibility"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  bool useMC = !cfgc.options().get<bool>("disable-mc");
  GlobalTrackID::mask_t srcTrk = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("track-types"));
  GlobalTrackID::mask_t srcCl = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("cluster-types"));
  InputHelper::addInputSpecs(cfgc, specs, srcCl, srcTrk, srcTrk, useMC);

  return std::move(specs);
}
