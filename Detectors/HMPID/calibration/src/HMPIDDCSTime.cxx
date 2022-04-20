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

// help-class for HMPIDDCSProcessor,
// to evaluate timeStamps of DPCOM-vectors 

#include "HMPIDCalibration/HMPIDDCSTime.h"
#include <algorithm>
#include <iterator>
#include <vector>
#include <array>
#include <memory>
#include <deque> 



#include "DetectorsDCS/DataPointCompositeObject.h"

#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h" 
#include "CommonUtils/MemFileHelper.h"



using namespace o2::dcs;

// https://github.com/AliceO2Group/AliceO2/blob/dev/DataFormats/Detectors/TPC/include/DataFormatsTPC/DCS.h 

using DPCOM = o2::dcs::DataPointCompositeObject;

namespace o2::hmpid {
          using TimeStampType = uint64_t;

  // return timestamp of first fetched datapoint for a given ID (Tin/Tout, Environment pressure, HV, chamber pressure)	
  TimeStampType HMPIDDCSTime::getMinTime(const std::vector<DPCOM> dps)
  {		
      //constexpr auto max = std::numeric_limits<uint64_t>::max(); in TPC, not necessary here ? 
	  
      // initialize firstTime to maximal possible value of uint64_t: 
      TimeStampType firstTime = std::numeric_limits<uint64_t>::max(); 
      for (const auto& dp : dps) {           
     
	// TOF/TPC: unordered map, checks size of dvect for dps with ids that matches ID
        //const auto time = size_condition ? dp.data.get_epoch_time() : max;

	// check if time of DP is earlier than previously latest fetched DP: 
	const auto time = dp.data.get_epoch_time();
        firstTime = std::min(firstTime, time);
      }
      
      return firstTime;
  }

    // return timestamp of last fetched datapoint for a given ID (Tin/Tout, Environment pressure, HV, chamber pressure)		
    TimeStampType HMPIDDCSTime::getMaxTime(const std::vector<DPCOM> dps)
    {
      //constexpr auto min = 0; in TPC, not necessary here ? 
	    
      // initialize lastTime to 0    
      TimeStampType lastTime = 0;
      for (const auto& dp : dps) {
	
	// TOF/TPC: unordered map, checks size of dvect for dps with ids that matches ID
        // const auto time = size_condition ? dp.data.get_epoch_time() : 0;
	
	// check if time of DP is greater (i.e. later) than previously latest fetched DP: 
        const auto time = dp.data.get_epoch_time(); 	
        lastTime = std::max(lastTime, time);
      }
      return lastTime;
    }

// iterates through 1d-array of DPCOM-vectors, for arNmean[42] startTimeTemp endTimeTemp
TimeStampType HMPIDDCSTime::getMinTimeArr(const std::vector<DPCOM> dataArray[])
  {
    // initialize firstTime to first entry in first vector array
    TimeStampType firstTime = (dataArray[0]).at(0).data.get_epoch_time(); 
    // (dataArray[0]) = vector at element num 1; .at(0) = first DPCOM vector element  
    // .data.get_epoch_time gets time_stamp
	
    // iterate through array of vectors, pass each array-element (i.e. vector of DPCOMs)
    // to getMinTime-function
    for(int i = 0; i < dataArray->size(); i++){
        firstTime = std::min(getMinTime(dataArray[i]), firstTime);
    }
    return firstTime;
  }
      
   // iterates through 1d-array of DPCOM-vectors,  for arNmean[42] startTimeTemp endTimeTemp
  TimeStampType HMPIDDCSTime::getMaxTimeArr(const std::vector<DPCOM> dataArray[])
  {
    // initialize lastTime to first entry of first vector in array
    TimeStampType lastTime = (dataArray[0]).at(0).data.get_epoch_time(); 
    // (dataArray[0]) = vector at element num 1; .at(0) = first DPCOM vector element  
    // .data.get_epoch_time gets time_stamp

    // iterate through array of vectors, pass each array-element (i.e. vector of DPCOMs)
    // to getMaxTime-function
    for(int i = 0; i < dataArray->size(); i++){
        lastTime = std::max(getMaxTime(dataArray[i]), lastTime);
    }
    return lastTime;
  } 
}

