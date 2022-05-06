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

//using namespace o2::framework; // no

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
  
  void aliasString()
  {
 
      
    std::vector<std::string> aliases;

    // ==| Environment Pressure  (mBar) |=================================
    aliases.push_back("HMP_DET/HMP_ENV/HMP_ENV_PENV.actual.value");
    
    int maxChambers = 7;
    // ==|(CH4) Chamber Pressures  (mBar?) |=================================
    aliases.push_back(fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_GAS/HMP_MP[0..{}]_GAS_PMWPC.actual.value",maxChambers,maxChambers,maxChambers));	

    //==| Temperature C6F14 IN/OUT / RADIATORS  (C) |=================================
    int iRad = 3; 

    aliases.push_back(fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_LIQ_LOOP.actual.sensors.Rad[0..{}]In_Temp",maxChambers,maxChambers,iRad));
    aliases.push_back(fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_LIQ_LOOP.actual.sensors.Rad[0..{}]Out_Temp",maxChambers,maxChambers,iRad));	

    // ===| HV / SECTORS (V) |=========================================================	      
    int iSec = 6; 
    aliases.push_back(o2::dcs::test::DataPointHint<double>{fmt::format("HMP_DET/HMP_MP[0..{}]/HMP_MP[0..{}]_PW/HMP_MP[0..{}]_SEC[0..{}]/HMP_MP[0..{}]_SEC[0..{}]_HV.actual.vMon",maxChambers,maxChambers,maxChambers,iSec,maxChambers,iSec), 2400., 2500.});


    // string for DPs of Refractive Index Parameters =============================================================
    aliases.push_back(fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].waveLenght"));
    aliases.push_back(fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].argonReference"));
    aliases.push_back(fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].argonCell"));
    aliases.push_back(fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].c6f14Cell"));
    aliases.push_back(fmt::format("HMP_DET/HMP_INFR/HMP_INFR_TRANPLANT/HMP_INFR_TRANPLANT_MEASURE.mesure[00..29].c6f14Reference"));     
  }
  
  // fill CCDB with ChargeThres (arQthre)   
  void sendChargeThresOutput(DataAllocator& output);

  // fill CCDB with RefIndex (arrMean)   
  void sendRefIndexOutput(DataAllocator& output);
 	

 private:
 
  std::vector<std::string> aliases;
  std::vector<std::string> tempInString, tempOutString, chamberPressureString, highVoltageString;
  std::vector<std::string> waveLenghtString, argonReferenceString, argonCellString, c6f14CellString, c6f14ReferenceString;

 
  std::unique_ptr<HMPIDDCSProcessor> mProcessor;
  HighResClock::time_point mTimer;
  int64_t mDPsUpdateInterval;


}; // end class
} // namespace hmpid

namespace framework
{
DataProcessorSpec getHMPIDDCSDataProcessorSpec(); 
} // namespace framework
} // namespace o2

#endif



