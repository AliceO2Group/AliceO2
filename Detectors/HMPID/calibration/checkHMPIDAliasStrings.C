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

#include <vector>
#include <string>
#include "TFile.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DataPointIdentifier.h"

#include <unordered_map>
#include <chrono>

using DPID = o2::dcs::DataPointIdentifier;
void processHMPID(std::string alias);
void processIR(std::string alias);
auto indexOfIR= 69;
int subStringToInt(std::string inputString, std::size_t startIndex);
int makeHMPIDCCDBEntryForDCS(const std::string url = "http://localhost:8080")
{
// check if IR or other HMPID specifciation
			 
  
  // string for DPs of pressures, HV and temperatures =============================================================

      for(int iCh = 0; iCh < 1; iCh++)
      {

	   auto alias = (std::string) Form("HMP_DET/HMP_MP%i/HMP_MP%i_GAS/HMP_MP%i_GAS_PMWPC.actual.value",iCh,iCh,iCh);
           auto detector_id = alias.substr(0, 7);
	   std::cout << detector_id << std::endl;
	   //processHMPID(Form("HMP_DET/HMP_MP%i/HMP_MP%i_GAS/HMP_MP%i_GAS_PMWPC.actual.value",iCh,iCh,iCh));
           for(int iRad = 0; iRad < 1; iRad++)
           {  
		//processHMPID(Form("HMP_DET/HMP_MP%i/HMP_MP%i_LIQ_LOOP.actual.sensors.Rad%iOut_Temp",iCh,iCh,iRad));
           }        
           for(int iSec = 0; iSec < 1; iSec++)
           {  
		//processHMPID(Form("HMP_DET/HMP_MP%i/HMP_MP%i_PW/HMP_MP%i_SEC%i/HMP_MP%i_SEC%i_HV.actual.vMon",iCh,iCh,iCh,iSec,iCh,iSec));
           } 
      }

  
     // string for DPs of Refractive Index Parameters =============================================================
    
      std::string wLen = "waveLenght", argRef = "argonReference",
      argCell = "argonCell", c6f14Cell= "c6f14Cell", c6f14Ref = "c6f14Reference"; 

      std::string temp;	
      for(int i = 29; i < 30; i++)
      {
	temp = Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.",i);
        processIR(temp+wLen);
        processIR(temp+argRef);
        processIR(temp+argCell);
       processIR(temp+c6f14Cell);
       processIR(temp+c6f14Ref);
 
      }  
  return 0;
}

void processHMPID(std::string alias)
{
   
constexpr auto HMPID_ID{"HMP_DET"sv};
			 constexpr auto IR_ID{"HMP_DET/HMP_INFR"sv};

			// HMPID-temp, HV, pressure IDs (HMPID_ID{"HMP_DET"sv};)
			 constexpr auto TEMP_OUT_ID{"Out_Temp"sv};
			 constexpr auto TEMP_IN_ID{"In_Temp"sv};
			 constexpr auto HV_ID{"vMon"sv};
			 constexpr auto ENV_PRESS_ID{"PENV.actual.value"sv};
			 constexpr auto CH_PRESS_ID{"PMWPC.actual.value"sv};

// HMPID-IR IDs (IR_ID{"HMP_DET/HMP_INFR"sv})
			 constexpr auto WAVE_LEN_ID{"waveLenght"sv}; // 0-9 
			 constexpr auto REF_ID{"Reference"sv}; // argonReference and freonRef
			 constexpr auto ARGON_CELL_ID{"argonCell"sv}; // argon Cell reference 
			 constexpr auto FREON_CELL_ID{"c6f14Cell"sv}; // fron Cell Reference

			 constexpr auto ARGON_REF_ID{"argonReference"sv}; // argonReference 
			 constexpr auto FREON_REF_ID{"c6f14Reference"sv}; // freonReference

 if ( alias.substr(alias.length()-7) == TEMP_IN_ID ) {
      LOG(info) << "Temperature_in DP: {}"<< alias;

    } else if (alias.substr(alias.length()-8) == TEMP_OUT_ID) {
      LOG(info) << "Temperature_out DP: {}"<< alias;

    } else if (alias.substr(alias.length()-4) == HV_ID) {
      LOG(info) << "HV DP: {}"<< alias;

    } else if (alias.substr(alias.length()-17) == ENV_PRESS_ID ) {
      LOG(info) << "Environment Pressure DP: {}"<< alias;

    } else if (alias.substr(alias.length()-18) == CH_PRESS_ID) {
      LOG(info) << "Chamber Pressure DP: {}"<< alias;
    
    } else {
      LOG(debug) << "Unknown data point: {}"<< alias;
    }	    
}

 void processIR(std::string alias)
{
constexpr auto HMPID_ID{"HMP_DET"sv};
			 constexpr auto IR_ID{"HMP_DET/HMP_INFR"sv};



// HMPID-IR IDs (IR_ID{"HMP_DET/HMP_INFR"sv})
			 constexpr auto WAVE_LEN_ID{"waveLenght"sv}; // 0-9 
			 constexpr auto REF_ID{"Reference"sv}; // argonReference and freonRef
			 constexpr auto ARGON_CELL_ID{"argonCell"sv}; // argon Cell reference 
			 constexpr auto FREON_CELL_ID{"c6f14Cell"sv}; // fron Cell Reference

			 constexpr auto ARGON_REF_ID{"argonReference"sv}; // argonReference 
			 constexpr auto FREON_REF_ID{"c6f14Reference"sv}; // freonReference

     auto specify_id = alias.substr(alias.length()-9);
	std::cout << alias << std::endl;
	std::cout << specify_id << std::endl;
	auto numIR = subStringToInt(alias, indexOfIR );
	auto numIR_2nd =  subStringToInt(alias, indexOfIR+1);

	if (numIR_2nd!=-1 )
	  {numIR = numIR*10 +numIR_2nd;}
       	if(numIR < 30 && numIR >0) 				 
	  {
	if(alias.substr(alias.length()-10) == WAVE_LEN_ID) {
      		LOG(info) << "WAVE_LEN_ID DP: "<< alias;
	}  else if(specify_id == FREON_CELL_ID) { 
      		LOG(info) << "FREON_CELL_ID DP: "<< alias;
	}  else if(specify_id == ARGON_CELL_ID) { 
      		LOG(info) << "ARGON_CELL_ID DP: "<< alias;
	}
	else if(specify_id == REF_ID) { 
		if( alias.substr(alias.length()-14) ==  ARGON_REF_ID){
      			LOG(info) << "ARGON_REF_ID DP: "<< alias;
		} else if( alias.substr(alias.length()-14) ==  FREON_REF_ID){
      			LOG(info) << "FREON_REF_ID DP: "<< alias;
		}      LOG(debug) << "Unknown data point: "<< alias;
	} else LOG(debug) << "Datapoint not found: "<< alias;
	std::cout << "==================" << std::endl;
	  }
	 else LOG(debug) << "Datapoint index out of range: "<< numIR;
}

 int subStringToInt(std::string inputString, std::size_t startIndex)
{ 
  	char stringPos = inputString.at(startIndex);
	int charInt = ((int)stringPos) - ((int)'0');
	if(charInt < 10 && charInt >= 0) return charInt;
  	else return -1;
}
