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




/*

 // is it enough to make a 
 std::unordered_map<DPID,HMPIDDCSinfo> mHMPIDDCS;
  containing all the DPIDs and a struct of first and last value of timestamps?
 
 
  //TOFCalibration/TOFDCSProcessor.h
      const std::unordered_map<DPID, TOFDCSinfo>& getTOFDPsInfo() const { return mTOFDCS; }
      std::unordered_map<DPID, TOFDCSinfo> mTOFDCS;                // this is the object that will go to the CCDB

      const CcdbObjectInfo& getccdbDPsInfo() const { return mccdbDPsInfo; }
      CcdbObjectInfo mccdbDPsInfo;

   //TOFCalibration/TOFDCSProcessor.cxx::updateDPsCCDB
       o2::calibration::Utils::prepareCCDBobjectInfo(mTOFDCS, mccdbDPsInfo, "TOF/Calib/DCSDPs", md, mStartValidity, 3 * 24L * 3600000);

   ///testWorkflow/TOFDCSDataProcessorSpec.h
       void sendDPsoutput(DataAllocator& output)
            const auto& payload = mProcessor->getTOFDPsInfo();
            auto& info = mProcessor->getccdbDPsInfo();
            auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
=======================================================================================
  //TOFCalibration/TOFDCSProcessor.h
  const std::bitset<Geo::NCHANNELS>& getLVStatus() const { return mFeac; }
  std::bitset<Geo::NCHANNELS> mFeac;    // bitset with feac status per channel
          //  static constexpr int NCHANNELS = NSTRIPS * NPADS;
          
  const CcdbObjectInfo& getccdbLVInfo() const { return mccdbLVInfo; }
  CcdbObjectInfo mccdbLVInfo;
  
  ///testWorkflow/TOFDCSDataProcessorSpec.h
  void sendLVandHVoutput(DataAllocator& output)
      const auto& payload = mProcessor->getLVStatus();
      auto& info = mProcessor->getccdbLVInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
*/ 
#include <unistd.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/AliasExpander.h"
#include "HMPIDCalibration/HMPIDDCSProcessor.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

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

class HMPIDDataProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
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

