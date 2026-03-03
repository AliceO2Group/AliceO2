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

#include "FT0EventsPerBcSpec.h"
#include "Framework/Lifetime.h"
#include <limits>

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& cfgc)
{
  using namespace o2::framework;
  using o2::calibration::FT0EventsPerBcProcessor;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", "FT0", "DIGITSBC", Lifetime::Timeframe);
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                false,                          // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "EventsPerBc"}, Lifetime::Timeframe);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "EventsPerBc"}, Lifetime::Timeframe);
  DataProcessorSpec dataProcessorSpec{
    "FT0EventsPerBcProcessor",
    inputs,
    outputs,
    AlgorithmSpec(adaptFromTask<FT0EventsPerBcProcessor>(ccdbRequest)),
    Options{
      {"save-to-file", VariantType::Bool, false, {"Save calibration object to local file"}},
      {"slot-len-sec", VariantType::UInt32, 3600u, {"Duration of each slot in seconds"}},
      {"one-object-per-run", VariantType::Bool, false, {"If set, workflow creates only one calibration object per run"}},
      {"min-entries-number", VariantType::UInt32, 5000u, {"Minimum number of entries required for a slot to be valid"}},
      {"min-ampl-side-a", VariantType::Int, 0, {"Amplitude threshold for Side A events"}},
      {"min-ampl-side-c", VariantType::Int, 0, {"Amplitude threshold for Side C events"}},
      {"min-sum-of-ampl", VariantType::Int, 0, {"Amplitude threshold for sum of A-side and C-side amplitudes"}}}};

  WorkflowSpec workflow;
  workflow.emplace_back(dataProcessorSpec);
  return workflow;
}
