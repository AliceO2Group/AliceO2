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
    {"max-sectors", VariantType::Int, 0, {"max sector number to use for HV sensors, 0-17"}},
  };

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  const auto maxSectors = std::min(config.options().get<int>("max-sectors"), 17);

  std::vector<o2::dcs::test::HintType> dphints;
  // ===| Gas sensors and gas chromatograph values |============================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_ARGON", 0, 100.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_CO2", 0, 100.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_N2", 0, 100.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_NEON", 0, 100.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_O2", 0, 100.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_WATER", 0, 100.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_An_L1Sr141_H2O", 0, 1000.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_An_L1Sr141_O2", 0, 1000.});

  // ==| Temperature sensors |==================================================
  // only the ones inside the TPC
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_351_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_376_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_415_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_447_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_477_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_488_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_537_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_575_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_589_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_629_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_664_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_695_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_735_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_757_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_797_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_831_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_851_TEMPERATURE", 20., 30.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_PT_895_TEMPERATURE", 20., 30.});

  // ===| HV supplies |=========================================================
  //
  // ---| A-Side voltages |-----------------------------------------------------
  //
  // ---| bottom electrodes |---
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_A[00..{:02}]_I_G[1..4]B_U", maxSectors), 0, 800.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_A[00..{:02}]_O[1..3]_G[1..4]B_U", maxSectors), 0, 800.});

  // ---| top electrodes |---
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_A[00..{:02}]_I_G[1..4]T_U", maxSectors), 0, 800.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_A[00..{:02}]_O[1..3]_G[1..4]T_U", maxSectors), 0, 800.});

  // ---| C-Side voltages |-----------------------------------------------------
  // ---| bottom electrodes |---
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_C[00..{:02}]_I_G[1..4]B_U", maxSectors), 0, 800.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_C[00..{:02}]_O[1..3]_G[1..4]B_U", maxSectors), 0, 800.});

  // ---| top electrodes |---
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_C[00..{:02}]_I_G[1..4]T_U", maxSectors), 0, 800.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_C[00..{:02}]_O[1..3]_G[1..4]T_U", maxSectors), 0, 800.});

  // ---| A-Side currents |-----------------------------------------------------
  //
  // ---| bottom electrodes |---
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_A[00..{:02}]_I_G[1..4]B_I", maxSectors), 0, 800.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_A[00..{:02}]_O[1..3]_G[1..4]B_I", maxSectors), 0, 800.});

  // ---| top electrodes |---
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_A[00..{:02}]_I_G[1..4]T_I", maxSectors), 0, 800.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_A[00..{:02}]_O[1..3]_G[1..4]T_I", maxSectors), 0, 800.});

  // ---| C-Side currents |-----------------------------------------------------
  // ---| bottom electrodes |---
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_C[00..{:02}]_I_G[1..4]B_I", maxSectors), 0, 800.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_C[00..{:02}]_O[1..3]_G[1..4]B_I", maxSectors), 0, 800.});

  // ---| top electrodes |---
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_C[00..{:02}]_I_G[1..4]T_I", maxSectors), 0, 800.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("TPC_HV_C[00..{:02}]_O[1..3]_G[1..4]T_I", maxSectors), 0, 800.});

  WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "TPC"));
  return specs;
}
