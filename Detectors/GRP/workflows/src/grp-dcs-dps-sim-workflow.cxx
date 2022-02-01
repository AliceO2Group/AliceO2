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
  // for GRP
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"L3Current", 29.9, 30.1});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"DipoleCurrent", 11.9, 12.1});
  dphints.emplace_back(o2::dcs::test::DataPointHint<bool>{"L3Polarity", 0, 1});
  dphints.emplace_back(o2::dcs::test::DataPointHint<bool>{"DipolePolarity", 0, 1});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"CavernTemperature", 16, 28});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"CavernAtmosPressure", 0.95, 1.05});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"SurfaceAtmosPressure", 0.95, 1.05});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"CavernAtmosPressure2", 0.95, 1.05});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"LHC_IntensityBeam[1..2]_totalIntensity", +1.811e+14, +1.906e+14});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ALI_Background[1..3]", +1.839e+14, +1.909e+14});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"ALI_Lumi_Total_Inst", +2.589e+00, +2.618e+00});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"BPTX_deltaT_B1_B2", +4.096e-02, +5.006e-02});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"BPTX_deltaTRMS_B1_B2", +2.800e-02, +2.800e-02});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"BPTX_Phase_B[1..2]", +1.142e+00, +1.181e+00});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"BPTX_PhaseRMS_B[1..2]", +2.271e-03, +3.942e-03});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"BPTX_Phase_Shift_B[1..2]", +1.835e-02, -2.068e-02});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"LHC_CollimatorPos_TCLIA_4R2_lvdt_gap_downstream", +5.592e+01, +5.592e+01});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"LHC_CollimatorPos_TCLIA_4R2_lvdt_gap_upstream", +5.588e+01, +5.588e+01});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"LHC_CollimatorPos_TCLIA_4R2_lvdt_left_downstream", +2.797e+01, +2.797e+01});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"LHC_CollimatorPos_TCLIA_4R2_lvdt_left_upstream", +2.799e+01, +2.799e+01});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"LHC_CollimatorPos_TCLIA_4R2_lvdt_right_downstream", -2.797e+01, -2.797e+01});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"LHC_CollimatorPos_TCLIA_4R2_lvdt_right_upstream", -2.799e+01, -2.799e+01});
  dphints.emplace_back(o2::dcs::test::DataPointHint<std::string>{"ALI_Lumi_Source_Name", "FT0ORA", "FT0ORC"});
  dphints.emplace_back(o2::dcs::test::DataPointHint<std::string>{"BEAM_MODE", "RAMP", "STABLE BEAMS"});
  dphints.emplace_back(o2::dcs::test::DataPointHint<std::string>{"MACHINE_MODE", "PP PHYSICS", "ION PHYSICS"});
  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "GRP"));
  return specs;
}
