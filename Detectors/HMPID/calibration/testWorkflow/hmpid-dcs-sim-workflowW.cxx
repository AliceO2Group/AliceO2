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
#include <iostream>
#include "Framework/ConfigParamSpec.h"


#include "DCStestWorkflow/DCSRandomDataGeneratorSpec.h"

using namespace o2::framework;


#include "Framework/runDataProcessing.h"

void t( )
{
  
 	int maxChambers = 6; iRad = 3; iSec = 6;
	// ==| Environment Pressure  (mBar) |=================================
	std::cout << "HMP_DET/HMP_ENV/HMP_ENV_PENV.actual.value";//, 980., 1040.});
std::cout << fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_GAS/HMP_MP[0..{}]_GAS_PMWPC.actual.value",maxChambers,maxChambers,maxChambers);

	

	std::cout <<fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_LIQ_LOOP.actual.sensors.Rad[0..{}]In_Temp",maxChambers,maxChambers,iRad);
	

	      
	auto iSec = 6; 

	std::cout <<fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_PW/HMP_MP[0..{}]_SEC[0..{}]/HMP_MP[0..{}]_SEC[0..{}]_HV.actual.vMon",maxChambers,maxChambers,maxChambers,iSec,maxChambers,iSec),);


	// string for DPs of Refractive Index Parameters =============================================================
	// EF: dont know ranges for IRs yet, using 2400 2500 as temp
	std::cout << fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].waveLenght");


 

}

