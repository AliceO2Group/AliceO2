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

#ifndef O2_HMPID_DATAPROCESSORSPEC_H
#define O2_HMPID_DATAPROCESSORSPEC_H

/// @file   DCSTOFDataProcessorSpec.h
/// @brief  TOF Processor for DCS Data Points


#include <unistd.h>
#include "Framework/Task.h"
#include "HMPIDCalibration/HMPIDDCSProcessor.h"
#include "Framework/DataProcessorSpec.h"
#include "CCDB/BasicCCDBManager.h"

using namespace o2::framework;

namespace o2
{
namespace hmpid
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;
using namespace o2::ccdb;
using CcdbManager = o2::ccdb::BasicCCDBManager;
using clbUtils = o2::calibration::Utils;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

class HMPIDDCSDataProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  
  
  
  // is for-loop ok, or should use something like
  // https://github.com/AliceO2Group/AliceO2/blob/f76eb23c3fa8d81c5bfea9071d047ef548059fd4/Detectors/ITSMFT/MFT/condition/testWorkflow/MFTDCSDataProcessorSpec.h#L92-L115
  void aliasString()
  {
   aliases.push_back("HMP_DET/HMP_ENV/HMP_ENV_PENV.actual.value"); // environment pressure 
      for(int iCh = 0; iCh < 6; iCh++)
      {
           chamberPressureString.push_back( Form("HMP_DET/HMP_MP%i/HMP_MP%i_GAS/HMP_MP%i_GAS_PMWPC.actual.value",iCh,iCh,iCh));
           for(int iRad = 0; iRad < 3; iRad++)
           {  
               tempOutString.push_back(Form("HMP_DET/HMP_MP%i/HMP_MP%i_LIQ_LOOP.actual.sensors.Rad%iOut_Temp",iCh,iCh,iRad)); 
               tempInString.push_back(Form("HMP_DET/HMP_MP%i/HMP_MP%i_LIQ_LOOP.actual.sensors.Rad%iIn_Temp",iCh,iCh,iRad)); 
           }        
           for(int iSec = 0; iSec < 3; iSec++)
           {  
               highVoltageString.push_back(Form("HMP_DET/HMP_MP%i/HMP_MP%i_PW/HMP_MP%i_SEC%i/HMP_MP%i_SEC%i_HV.actual.vMon",iCh,iCh,iCh,iSec,iCh,iSec)); 
           } 
      }
      aliases.insert(aliases.end(), chamberPressureString.begin(), chamberPressureString.end()); 
      aliases.insert(aliases.end(), tempOutString.begin(), tempOutString.end()); 
      aliases.insert(aliases.end(), tempInString.begin(), tempInString.end()); 
      aliases.insert(aliases.end(), highVoltageString.begin(), highVoltageString.end()); 
      
      for(int i = 0; i < 30; i++)
      {
        waveLenghtString.push_back(Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.waveLenght",i));
        argonReferenceString.push_back(Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.argonReference",i));
        argonCellString.push_back(Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.argonCell",i));
        c6f14CellString.push_back(Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.c6f14Cell",i));
        c6f14ReferenceString.push_back(Form("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure%i.c6f14Reference",i)); 
      }  
      aliases.insert(aliases.end(), waveLenghtString.begin(), waveLenghtString.end()); 
      aliases.insert(aliases.end(), argonReferenceString.begin(), argonReferenceString.end()); 
      aliases.insert(aliases.end(), argonCellString.begin(), argonCellString.end()); 
      aliases.insert(aliases.end(), c6f14CellString.begin(), c6f14CellString.end()); 
      aliases.insert(aliases.end(), c6f14ReferenceString.begin(), c6f14ReferenceString.end()); 
  }


 private:
 
  std::vector<std::string> aliases;
  std::vector<std::string> tempInString, tempOutString, chamberPressureString, highVoltageString;
  std::vector<std::string> waveLenghtString, argonReferenceString, argonCellString, c6f14CellString, c6f14ReferenceString;

 
  std::unique_ptr<HMPIDDCSProcessor> mProcessor;
  HighResClock::time_point mTimer;
  int64_t mDPsUpdateInterval;

  // fill CCDB with ChargeThres (arQthre)   
  void sendChargeThresOutput(DataAllocator& output);

  // fill CCDB with RefIndex (arrMean)   
  void sendRefIndexOutput(DataAllocator& output);


}; // end class
} // namespace hmpid

namespace framework
{
DataProcessorSpec getHMPIDDCSDataProcessorSpec(); 
} // namespace framework
} // namespace o2

#endif



