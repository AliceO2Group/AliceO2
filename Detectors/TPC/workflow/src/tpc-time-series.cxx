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

/// \file   tpc-time-series.cxx
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#include "TPCWorkflow/TPCTimeSeriesSpec.h"
#include "TPCWorkflow/TPCTimeSeriesWriterSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "GPUDebugStreamer.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"disable-root-output", VariantType::Bool, false, {"disable root-files output writers"}},
    {"material-type", VariantType::Int, 2, {"Type for the material budget during track propagation: 0=None, 1=Geo, 2=LUT"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  WorkflowSpec workflow;
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  const bool disableWriter = config.options().get<bool>("disable-root-output");
  auto materialType = static_cast<o2::base::Propagator::MatCorrType>(config.options().get<int>("material-type"));
  workflow.emplace_back(o2::tpc::getTPCTimeSeriesSpec(disableWriter, materialType));
  if (!disableWriter) {
    workflow.emplace_back(o2::tpc::getTPCTimeSeriesWriterSpec());
  }
  return workflow;
}
