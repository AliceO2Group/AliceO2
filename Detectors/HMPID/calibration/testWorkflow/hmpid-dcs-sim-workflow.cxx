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


#include <fmt/format.h>

#include "Framework/ConfigParamSpec.h"


#include "DCStestWorkflow/DCSRandomDataGeneratorSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"max-chambers", VariantType::Int, 0, {"max chamber number to use DCS variables, 0-6"}},
  };

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  const auto maxChambers = std::min(config.options().get<int>("max-chambers"), 6);

  std::vector<o2::dcs::test::HintType> dphints;
  // ===| CH4 PRESSURE values (mbar) |============================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_CH4_PRESSURE", 980., 1040.});
//EF is this the correct one?: 

//  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMP_DET/HMP_MP1/HMP_MP1_GAS/HMP_MP1_GAS_PMWPC.actual.value", 980., 1040.});


  // ==| Temperature C6F14 IN/OUT / RADIATORS  (C) |=================================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_C6F14_RADIATOR0_IN_TEMPERATURE", 25., 27.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_C6F14_RADIATOR0_OUT_TEMPERATURE", 25., 27.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_C6F14_RADIATOR1_IN_TEMPERATURE", 25., 27.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_C6F14_RADIATOR1_OUT_TEMPERATURE", 25., 27.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_C6F14_RADIATOR2_IN_TEMPERATURE", 25., 27.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_C6F14_RADIATOR2_OUT_TEMPERATURE", 25., 27.});

  // ===| HV / SECTORS (V) |=========================================================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_HV_RADIATOR0_SECTOR0", 2400., 2500.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_HV_RADIATOR0_SECTOR1", 2400., 2500.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_HV_RADIATOR1_SECTOR0", 2400., 2500.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_HV_RADIATOR1_SECTOR1", 2400., 2500.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_HV_RADIATOR2_SECTOR0", 2400., 2500.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMPID_HV_RADIATOR2_SECTOR1", 2400., 2500.});



  WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "HMPID"));
  return specs;
}
