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

// ==| Environment Pressure  (mBar) |=================================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMP_DET/HMP_ENV/HMP_ENV_PENV.actual.value", 980., 1040.});


      for(int iCh = 0; iCh < 7; iCh++)
      {
    	// ==| Chamber Pressures  (mBar?) |=================================
	dphints.emplace_back(o2::dcs::test::DataPointHint<double>{Form("HMP_DET/HMP_MP%i/HMP_MP%i_GAS/HMP_MP%i_GAS_PMWPC.actual.value",iCh,iCh,iCh), 980., 1040.});

	    // ==| Temperature C6F14 IN/OUT / RADIATORS  (C) |=================================
           for(int iRad = 0; iRad < 3; iRad++)
           {  
		   		 
		dphints.emplace_back(o2::dcs::test::DataPointHint<double>{Form("HMP_DET/HMP_MP%i/HMP_MP%i_LIQ_LOOP.actual.sensors.Rad%iIn_Temp",iCh,iCh,iRad),25., 27.});
		dphints.emplace_back(o2::dcs::test::DataPointHint<double>{Form("HMP_DET/HMP_MP%i/HMP_MP%i_LIQ_LOOP.actual.sensors.Rad%iOut_Temp",iCh,iCh,iRad),25., 27.});

           }
	      
	   // ===| HV / SECTORS (V) |=========================================================	      
           for(int iSec = 0; iSec < 6; iSec++)
           {  
		dphints.emplace_back(o2::dcs::test::DataPointHint<double>{Form("HMP_DET/HMP_MP%i/HMP_MP%i_PW/HMP_MP%i_SEC%i/HMP_MP%i_SEC%i_HV.actual.vMon",iCh,iCh,iCh,iSec,iCh,iSec), 2400., 2500.});
           } 
      }

  
  
     // string for DPs of Refractive Index Parameters =============================================================
     // EF: dont know ranges for IRs yet, using 2400 2500 as temp
  
      for(int i = 0; i < 30; i++)
      {
        dphints.emplace_back(o2::dcs::test::DataPointHint<double>{Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.waveLenght",i), 2400., 2500.});
        dphints.emplace_back(o2::dcs::test::DataPointHint<double>{Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.argonReference",i), 2400., 2500.});
        dphints.emplace_back(o2::dcs::test::DataPointHint<double>{Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.argonCell",i), 2400., 2500.});
        dphints.emplace_back(o2::dcs::test::DataPointHint<double>{Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.c6f14Cell",i), 2400., 2500.});
        dphints.emplace_back(o2::dcs::test::DataPointHint<double>{Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.c6f14Reference",i), 2400., 2500.}); 
      }  








 

  

  WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "HMPID"));
  return specs;
}
