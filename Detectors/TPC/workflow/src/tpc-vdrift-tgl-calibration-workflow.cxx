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

#include "Framework/DataProcessorSpec.h"
#include "TPCWorkflow/TPCVDriftTglCalibSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"nbins-tgl", o2::framework::VariantType::Int, 20, {"number of bins in tgL"}},
    {"max-tgl-its", o2::framework::VariantType::Float, 1.f, {"max range for tgL of ITS tracks"}},
    {"nbins-dtgl", o2::framework::VariantType::Int, 100, {"number of bins in tgL_ITS - tgl_TPC"}},
    {"max-dtgl-itstpc", o2::framework::VariantType::Float, 0.2f, {"max range for tgL_ITS - tgl_TPC"}},
    {"min-entries-per-slot", o2::framework::VariantType::Int, 10000, {"mininal number of entries per slot"}},
    {"time-slot-length", o2::framework::VariantType::Int, 50000, {"time slot length"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  specs.emplace_back(o2::tpc::getTPCVDriftTglCalibSpec(configcontext.options().get<int>("nbins-tgl"),
                                                       configcontext.options().get<float>("max-tgl-its"),
                                                       configcontext.options().get<int>("nbins-dtgl"),
                                                       configcontext.options().get<float>("max-dtgl-itstpc"),
                                                       configcontext.options().get<int>("time-slot-length"),
                                                       configcontext.options().get<int>("min-entries-per-slot")));
  return specs;
}
