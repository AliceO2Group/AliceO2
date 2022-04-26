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
#include "MIDConditions/DCSNamer.h"

/**
 * DPL workflow which generates fake random MID DCS data points.
 *
 * Data points are generated for HV (currents and voltages).
 */
o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  std::vector<o2::dcs::test::HintType> dphints;

  for (auto a : o2::mid::dcs::aliases({o2::mid::dcs::MeasurementType::HV_V})) {
    dphints.emplace_back(o2::dcs::test::DataPointHint<double>{a, 9200, 9800});
  }

  for (auto a : o2::mid::dcs::aliases({o2::mid::dcs::MeasurementType::HV_I})) {
    dphints.emplace_back(o2::dcs::test::DataPointHint<double>{a, 2, 10});
  }

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "MID"));
  return specs;
}
