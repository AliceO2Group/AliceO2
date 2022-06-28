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

/// \file DCSDPHints.cxx
/// \brief DCS data point configuration for the TPC
///
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include "DetectorsDCS/DCSDataPointHint.h"
#include "TPCdcs/DCSDPHints.h"

std::vector<o2::dcs::test::HintType> o2::tpc::dcs::getTPCDCSDPHints(const int maxSectors)
{
  std::vector<o2::dcs::test::HintType> dphints;
  // ===| Gas sensors and gas chromatograph values |============================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_ARGON", 0., 100.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_CO2", 0., 20.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_N2", 4., 6.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_NEON", 85., 95.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"TPC_GC_O2", 0.02, 0.035});
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

  // ---| Stack status |--------------------------------------------------------
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{fmt::format("TPC_HV_A[00..{:02}]_I_STATUS", maxSectors), 0, 29});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{fmt::format("TPC_HV_A[00..{:02}]_O[1..3]_STATUS", maxSectors), 0, 29});

  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{fmt::format("TPC_HV_C[00..{:02}]_I_STATUS", maxSectors), 0, 29});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{fmt::format("TPC_HV_C[00..{:02}]_O[1..3]_STATUS", maxSectors), 0, 29});

  return dphints;
}
