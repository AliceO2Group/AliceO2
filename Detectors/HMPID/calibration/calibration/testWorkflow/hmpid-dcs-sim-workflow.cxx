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

//https://github.com/AliceO2Group/AliceO2/blob/ef01b5f61f4832f253a20b573cbbcbe9a96b7593/Detectors/DCS/testWorkflow/src/DCSRandomDataGeneratorSpec.cxx#L169-L178
// max-timeframes and delta-fraction-

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
  const auto maxChambers = std::min(config.options().get<int>("max-chambers"), 6); // o2-hmpid-dcs-sim-workflow --max-chambers 6

  std::vector<o2::dcs::test::HintType> dphints;

  // ==| Environment Pressure  (mBar) |=================================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMP_DET/HMP_ENV/HMP_ENV_PENV.actual.value", 980., 1040.});


  // ==|(CH4) Chamber Pressures  (mBar?) |=================================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_GAS/HMP_MP[0..{}]_GAS_PMWPC.actual.value",maxChambers,maxChambers,maxChambers), 980., 1040.});	

  //==| Temperature C6F14 IN/OUT / RADIATORS  (C) |=================================
  int iRad = 3; 

  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_LIQ_LOOP.actual.sensors.Rad[0..{}]In_Temp",maxChambers,maxChambers,iRad),25., 27.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_LIQ_LOOP.actual.sensors.Rad[0..{}]Out_Temp",maxChambers,maxChambers,iRad),25., 27.});	

  // ===| HV / SECTORS (V) |=========================================================	      
  int iSec = 6; 

  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_PW/HMP_MP[0..{}]_SEC[0..{}]/HMP_MP[0..{}]_SEC[0..{}]_HV.actual.vMon",maxChambers,maxChambers,maxChambers,iSec,maxChambers,iSec), 2400., 2500.});


  // string for DPs of Refractive Index Parameters =============================================================
  // EF: dont know ranges for IRs yet, using 2400 2500 as temp
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].waveLenght"), 2400., 2500.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].argonReference"), 2400., 2500.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].argonCell"), 2400., 2500.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].c6f14Cell"), 2400., 2500.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].c6f14Reference"), 2400., 2500.});


  o2::framework::WorkflowSpec specs;

  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "HMP"));
  return specs;
}

/*

EnvPressure

HMP_DET/HMP_ENV/HMP_ENV_PENV.actual.value


Chamber Pressure 

HMP_DET/HMP_MP[0..6]/HMP_MP[0..6]_GAS/HMP_MP[0..6]_GAS_PMWPC.actual.value


Temp-in

HMP_DET/HMP_MP[0..6]/HMP_MP[0..6]_LIQ_LOOP.actual.sensors.Rad[0..3]In_Temp


HV

HMP_DET/HMP_MP[0..6]/HMP_MP[0..6]_PW/HMP_MP[0..6]_SEC[0..6]/
HMP_MP[0..6]_SEC[0..6]_HV.actual.vMon


IR

HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].waveLenght

*/




