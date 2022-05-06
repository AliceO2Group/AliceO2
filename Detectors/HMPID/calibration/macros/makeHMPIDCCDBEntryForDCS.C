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

int makeHMPIDCCDBEntryForDCS(const std::string url = "http://localhost:8080")
{

  //  std::string url(argv[0]);
  // macro to populate CCDB for HMPID with the configuration for DCS
  std::unordered_map<DPID, std::string> dpid2DataDesc;
  std::vector<std::string> aliases; // vector of strings that will hold DataPoints identifiers
  
  // string for DPs of pressures, HV and temperatures =============================================================
  std::vector<std::string> tempInString, tempOutString, chamberPressureString, highVoltageString;

   aliases.push_back("HMP_DET/HMP_ENV/HMP_ENV_PENV.actual.value"); // environment pressure 
      for(int iCh = 0; iCh < 7; iCh++)
      {
           chamberPressureString.push_back( Form("HMP_DET/HMP_MP%i/HMP_MP%i_GAS/HMP_MP%i_GAS_PMWPC.actual.value",iCh,iCh,iCh));
           for(int iRad = 0; iRad < 3; iRad++)
           {  
               tempOutString.push_back(Form("HMP_DET/HMP_MP%i/HMP_MP%i_LIQ_LOOP.actual.sensors.Rad%iOut_Temp",iCh,iCh,iRad)); 
               tempInString.push_back(Form("HMP_DET/HMP_MP%i/HMP_MP%i_LIQ_LOOP.actual.sensors.Rad%iIn_Temp",iCh,iCh,iRad)); 
           }        
           for(int iSec = 0; iSec < 6; iSec++)
           {  
               highVoltageString.push_back(Form("HMP_DET/HMP_MP%i/HMP_MP%i_PW/HMP_MP%i_SEC%i/HMP_MP%i_SEC%i_HV.actual.vMon",iCh,iCh,iCh,iSec,iCh,iSec)); 
           } 
      }
      aliases.insert(aliases.end(), chamberPressureString.begin(), chamberPressureString.end()); 
      aliases.insert(aliases.end(), tempOutString.begin(), tempOutString.end()); 
      aliases.insert(aliases.end(), tempInString.begin(), tempInString.end()); 
      aliases.insert(aliases.end(), highVoltageString.begin(), highVoltageString.end()); 
  
  
     // string for DPs of Refractive Index Parameters =============================================================
      std::vector<std::string> waveLenghtString, argonReferenceString, argonCellString, c6f14CellString, c6f14ReferenceString;
      std::string wLen = "waveLenght", argRef = "argonReference",
      argCell = "argonCell", c6f14Cell= "c6f14Cell", c6f14Ref = "c6f14Reference"; 

      std::string temp;	
      for(int i = 0; i < 30; i++)
      {
	temp = Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.",i);
        waveLenghtString.push_back(temp+wLen);
        argonReferenceString.push_back(temp+argRef);
        argonCellString.push_back(temp+argCell);
        c6f14CellString.push_back(temp+c6f14Cell);
        c6f14ReferenceString.push_back(temp+c6f14Ref); 
      }  
      aliases.insert(aliases.end(), waveLenghtString.begin(), waveLenghtString.end()); 
      aliases.insert(aliases.end(), argonReferenceString.begin(), argonReferenceString.end()); 
      aliases.insert(aliases.end(), argonCellString.begin(), argonCellString.end()); 
      aliases.insert(aliases.end(), c6f14CellString.begin(), c6f14CellString.end()); 
      aliases.insert(aliases.end(), c6f14ReferenceString.begin(), c6f14ReferenceString.end()); 

  std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);

  DPID dpidtmp;
  for (size_t i = 0; i < expaliases.size(); ++i) {
    DPID::FILL(dpidtmp, expaliases[i], o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = "HMPIDDATAPOINTS";
  }

  o2::ccdb::CcdbApi api;
  api.init(url); // or http://localhost:8080 for a local installation
  std::map<std::string, std::string> md;
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  api.storeAsTFileAny(&dpid2DataDesc, "HMPID/Config/DCSDPconfig", md, ts, 99999999999999);

  return 0;
}


