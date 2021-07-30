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

#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"do not use MC info"}},
    {"read-tpc-only-as-well", o2::framework::VariantType::Bool, false, {"read also TPC only tracks"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  auto useMC = !cfgc.options().get<bool>("disable-mc");
  auto readTPConly = cfgc.options().get<bool>("read-tpc-only-as-well");
  WorkflowSpec wf;
  // Update the (declared) parameters if changed from the command line
  wf.emplace_back(o2::tof::getTOFMatchedReaderSpec(useMC));
  wf.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
  if (readTPConly) {
    wf.emplace_back(o2::tpc::getTPCTrackReaderSpec(useMC));
  }
  return wf;
}
