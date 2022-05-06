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

#ifndef HMPIDDCSTIME_H
#define HMPIDDCSTIME_H

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

using DPCOM = o2::dcs::DataPointCompositeObject;


namespace o2::hmpid
{
  
    class HMPIDDCSTime{
     

      public: 
          using TimeStampType = uint64_t;

          struct TimeRange {
              uint64_t first{};
              uint64_t last{};
          };

          const auto& getTimeTemperature() const { return mTimeTemperature; } 
          const auto& getTimeHighVoltage() const { return mTimeHighVoltage; }
          const auto& getTimeEnvPressure() const { return mTimeEnvPressure; }
          const auto& getTimeChamberPressure() const { return mTimeChamberPressure; }



          static TimeStampType getMinTimeArr(const std::vector<DPCOM> dataArray[]);
          
          static TimeStampType getMaxTimeArr(const std::vector<DPCOM> dataArray[]);

          static TimeStampType getMinTime(const std::vector<DPCOM> data);

          static TimeStampType getMaxTime(const std::vector<DPCOM> data);

          void sortAndClean(std::vector<DPCOM>& data);


          void clear(std::vector<DPCOM>& data) { data.clear(); }

          /// return value at the last valid time stamp
          ///
          /// values are valid unitl the next time stamp
          /*const T& getValueForTime(const TimeStampType timeStamp) const
          {
            const auto i = std::upper_bound(data.begin(), data.end(), DPType{timeStamp, {}});
            return (i == data.begin()) ? (*i).value : (*(i - 1)).value;
          }; */ 


	void doClear(std::vector<DPCOM>& data);
        void doSortAndClean(std::vector<DPCOM>& data);

	private: 
		TimeRange mTimeTemperature; ///< Time range for temperature values
		TimeRange mTimeHighVoltage; ///< Time range for high voltage values
		TimeRange mTimeEnvPressure;         ///< Time range for environment-pressure values 
		TimeRange mTimeChamberPressure;         ///< Time range for chamber-pressure values 


      ClassDefNV(HMPIDDCSTime,0);

    };// end class 
} // end o2::hmpid
#endif 

