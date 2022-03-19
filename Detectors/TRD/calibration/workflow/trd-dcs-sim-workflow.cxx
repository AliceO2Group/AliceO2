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
  // for TRD
  // for test, we use less DPs than official ones
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gasCO2", 0, 50.}); // adding a data point of type double with the name trd_gasCO2 which takes values between 0 and 50
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gasH2O", 0, 500.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gasO2", 0, 100.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gaschromatographCO2", 0, 100.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gaschromatographN2", 0, 100.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gaschromatographXe", 0, 100.});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_hvAnodeImon[00..02]", 0, 50.});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_hvAnodeUmon[00..02]", 0, 50.});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_hvDriftImon[00..02]", 0, 50.});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_hvDriftUmon[00..02]", 0, 50.});
  // dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"trd_fedChamberStatus", 0, 255});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"trd_runNo", 0, 255});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"trd_runType", 0, 255});
  // for TRD, official list, not yet complete
  /*
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gasCO2", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gasH2O", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gasO2", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gaschromatographCO2", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gaschromatographN2", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_gaschromatographXe", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_hvAnodeImon[00..539]", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_hvAnodeUmon[00..539]", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_hvDriftImon[00..539]", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"trd_hvDriftUmon[00..539]", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"trd_fedChamberStatus", 0, 255});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"trd_runNo", 0, 255});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"trd_runType", 0, 255});
  */
  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "TRD"));
  return specs;
}
