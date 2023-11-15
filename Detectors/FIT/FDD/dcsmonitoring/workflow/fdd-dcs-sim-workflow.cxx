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

/// \file fdd-dcs-sim-workflow.cxx
/// \brief Simulate DCS data for FDD
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "DCStestWorkflow/DCSRandomDataGeneratorSpec.h"
#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  std::vector<o2::dcs::test::HintType> dphints;
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/SIDE_A/HV_A9/[I,V]MON", 250, 350});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/SIDE_C/HV_C[9,32]/[I,V]MON", 250, 350});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/SIDE_C/LAYER0/PMT_0_[0..3]/[I,V]MON", 250, 350});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/SIDE_C/LAYER1/PMT_1_[0..3]/[I,V]MON", 250, 350});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/SIDE_A/LAYER2/PMT_2_[0..3]/[I,V]MON", 250, 350});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/SIDE_A/LAYER3/PMT_3_[0..3]/[I,V]MON", 250, 350});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint>{"FDD/PM/SIDE_A/PMT_A_9/ADC[0,1]_BASELINE", 30, 150});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint>{"FDD/PM/SIDE_C/PMT_C_[9,32]/ADC[0,1]_BASELINE", 30, 150});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint>{"FDD/PM/SIDE_C/LAYER0/PMT_0_[0..3]/ADC[0,1]_BASELINE", 30, 150});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint>{"FDD/PM/SIDE_C/LAYER1/PMT_1_[0..3]/ADC[0,1]_BASELINE", 30, 150});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint>{"FDD/PM/SIDE_A/LAYER2/PMT_2_[0..3]/ADC[0,1]_BASELINE", 30, 150});
  dphints.emplace_back(o2::dcs::test::DataPointHint<uint>{"FDD/PM/SIDE_A/LAYER3/PMT_3_[0..3]/ADC[0,1]_BASELINE", 30, 150});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/Trigger1_Central/CNT_RATE", 0, 5000000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/Trigger2_SemiCentral/CNT_RATE", 0, 5000000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/Trigger3_Vertex/CNT_RATE", 0, 5000000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/Trigger4_OrC/CNT_RATE", 0, 5000000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/Trigger5_OrA/CNT_RATE", 0, 5000000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/Background/[0..9]/CNT_RATE", 0, 50000});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"FDD/Background/[A,B,C,D,E,F,G,H]/CNT_RATE", 0, 50000});

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "FDD"));
  return specs;
}
