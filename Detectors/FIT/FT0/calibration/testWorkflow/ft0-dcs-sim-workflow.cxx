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

// // we need to add workflow options before including Framework/runDataProcessing
// void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
// {
//   // option allowing to set parameters
// }

// ------------------------------------------------------------------

#include "DCStestWorkflow/DCSRandomDataGeneratorSpec.h"
#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  std::vector<o2::dcs::test::HintType> dphints;
  // for testing, we use less DPs than the official ones
  dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_A/MCP_A[1..2]/actual/iMon", 250, 350});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int>{"FT0/PM/channel[000..001]/actual/ADC_ZERO", 150, 300});
  // Official list
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_A/MCP_A[1..5]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_A/MCP_B[1..5]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_A/MCP_C[1..2]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_A/MCP_C[4..5]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_A/MCP_D[1..5]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_A/MCP_E[1..5]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_C/MCP_A[2..5]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_C/MCP_B[1..6]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_C/MCP_C[1..2]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_C/MCP_C[5..6]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_C/MCP_D[1..2]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_C/MCP_D[5..6]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_C/MCP_E[1..6]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/FT0_C/MCP_F[2..5]/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<float>{"FT0/HV/MCP_LC/actual/iMon", 250, 350});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<int>{"FT0/PM/channel[000..211]/actual/ADC_ZERO", 150, 300});

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "FT0"));
  return specs;
}
