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

#include "DCStestWorkflow/DCSRandomDataGeneratorSpec.h"
#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  std::vector<o2::dcs::test::HintType> dphints;
  // for ZDC
  // (as far as my knowledge goes about the DCS dp right now!)
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNA_POS.actual.position", 100, 200});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPA_POS.actual.position", 100, 200});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNC_POS.actual.position", 100, 200});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPC_POS.actual.position", 100, 200});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNA_HV0.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNA_HV1.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNA_HV2.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNA_HV3.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNA_HV4.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPA_HV0.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPA_HV1.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPA_HV2.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPA_HV3.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPA_HV4.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNC_HV0.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNC_HV1.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNC_HV2.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNC_HV3.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNC_HV4.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPC_HV0.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPC_HV1.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPC_HV2.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPC_HV3.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPC_HV4.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZEM_HV0.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZEM_HV1.actual.vMon", 900, 2000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNA_HV0_D[1..2]", 100, 500});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZNC_HV0_D[1..2]", 100, 500});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPA_HV0_D[1..2]", 100, 500});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ZDC_ZPC_HV0_D[1..2]", 100, 500});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"ZDC_CONFIG_[00..32]", 0, 8});
  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "ZDC"));
  return specs;
}
