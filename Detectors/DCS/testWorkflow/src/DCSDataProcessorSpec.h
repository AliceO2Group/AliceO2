// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_DCS_DATAPROCESSOR_H
#define O2_DCS_DATAPROCESSOR_H

/// @file   DataGeneratorSpec.h
/// @brief  Dummy data generator

#include <unistd.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DCSProcessor.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

namespace o2
{
namespace dcs
{

using namespace o2::dcs;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

class DCSDataProcessor : public o2::framework::Task
{
 public:
  enum Detectors {
    kTest,
    // kTPC,  // commented out for test, when we use only 1 "Test" detector
    kNdetectors
  };

  void init(o2::framework::InitContext& ic) final
  {

    // stopping all stopwatches, since they start counting from the moment they are created
    for (int idet = 0; idet < kNdetectors; idet++) {
      mDeltaProcessingDetLoop[idet].Stop();
      mDeltaProcessingDetLoop[idet].Reset();
    }

    std::vector<DPID> pidVect;

    DPID dpidtmp;
    DeliveryType typechar = RAW_CHAR;
    std::string dpAliaschar = "TestChar_0";
    DPID::FILL(dpidtmp, dpAliaschar, typechar);
    pidVect.push_back(dpidtmp);

    //std::vector<int> vectDet{kTest, kTPC}; // only one detector for now
    std::vector<int> vectDet{kTest};
    mDetectorPid[dpidtmp] = vectDet;

    DeliveryType typeint = RAW_INT;
    for (int i = 0; i < 50000; i++) {
      std::string dpAliasint = "TestInt_" + std::to_string(i);
      DPID::FILL(dpidtmp, dpAliasint, typeint);
      pidVect.push_back(dpidtmp);
      mDetectorPid[dpidtmp] = vectDet;
    }

    DeliveryType typedouble = RAW_DOUBLE;
    for (int i = 0; i < 4; i++) {
      std::string dpAliasdouble = "TestDouble_" + std::to_string(i);
      DPID::FILL(dpidtmp, dpAliasdouble, typedouble);
      pidVect.push_back(dpidtmp);
      mDetectorPid[dpidtmp] = vectDet;
    }

    DeliveryType typestring = RAW_STRING;
    std::string dpAliasstring0 = "TestString_0";
    DPID::FILL(dpidtmp, dpAliasstring0, typestring);
    pidVect.push_back(dpidtmp);
    mDetectorPid[dpidtmp] = vectDet;

    for (int idet = 0; idet < kNdetectors; idet++) {
      mDCSprocVect[idet].init(pidVect);
      mDCSprocVect[idet].setMaxCyclesNoFullMap(ic.options().get<int64_t>("max-cycles-no-full-map"));
      mDCSprocVect[idet].setName("Test1Det");
    }
    mProcessFullDeltaMap = ic.options().get<bool>("process-full-delta-map");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfid = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    for (int idet = 0; idet < kNdetectors; idet++) {
      mDCSprocVect[idet].setTF(tfid);
    }

    TStopwatch s;
    LOG(DEBUG) << "TF: " << tfid << " -->  receiving binary data...";
    mReceiveBinaryData.Start(mFirstTF);
    auto rawchar = pc.inputs().get<const char*>("input");
    mReceiveBinaryData.Stop();
    LOG(DEBUG) << "TF: " << tfid << " -->  ...binary data received: realTime = "
               << mReceiveBinaryData.RealTime() << ", cpuTime = "
               << mReceiveBinaryData.CpuTime();
    LOG(DEBUG) << "TF: " << tfid << " -->  receiving (delta) binary data...";
    mDeltaReceiveBinaryData.Start(mFirstTF);
    auto rawcharDelta = pc.inputs().get<const char*>("inputDelta");
    mDeltaReceiveBinaryData.Stop();
    LOG(DEBUG) << "TF: " << tfid << " -->  ...binary (delta) data received: realTime = "
               << mDeltaReceiveBinaryData.RealTime()
               << ", cpuTime = " << mDeltaReceiveBinaryData.CpuTime();

    // full map
    const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().get("input").header);
    auto sz = dh->payloadSize;
    int nDPs = sz / sizeof(DPCOM);
    std::unordered_map<DPID, DPVAL> dcsmap;
    DPCOM dptmp;
    LOG(DEBUG) << "TF: " << tfid << " -->  building unordered_map...";
    mBuildingUnorderedMap.Start(mFirstTF);
    for (int i = 0; i < nDPs; i++) {
      memcpy(&dptmp, rawchar + i * sizeof(DPCOM), sizeof(DPCOM));
      dcsmap[dptmp.id] = dptmp.data;
      LOG(DEBUG) << "Reading from generator: i = " << i << ", DPCOM = " << dptmp;
      LOG(DEBUG) << "Reading from generator: i = " << i << ", DPID = " << dptmp.id;
    }
    mBuildingUnorderedMap.Stop();
    LOG(DEBUG) << "TF: " << tfid << " -->  ...unordered_map built = "
               << mBuildingUnorderedMap.RealTime() << ", cpuTime = " << mBuildingUnorderedMap.CpuTime();

    // delta map
    const auto* dhDelta = o2::header::get<o2::header::DataHeader*>(pc.inputs().get("inputDelta").header);
    auto szDelta = dhDelta->payloadSize;
    int nDPsDelta = szDelta / sizeof(DPCOM);
    std::unordered_map<DPID, DPVAL> dcsmapDelta;
    LOG(DEBUG) << "TF: " << tfid << " -->  building (delta) unordered_map...";
    mDeltaBuildingUnorderedMap.Start(mFirstTF);
    for (int i = 0; i < nDPsDelta; i++) {
      memcpy(&dptmp, rawcharDelta + i * sizeof(DPCOM), sizeof(DPCOM));
      dcsmapDelta[dptmp.id] = dptmp.data;
      LOG(DEBUG) << "Reading from generator: i = " << i << ", DPCOM = " << dptmp;
      LOG(DEBUG) << "Reading from generator: i = " << i << ", DPID = " << dptmp.id;
    }
    mDeltaBuildingUnorderedMap.Stop();
    LOG(DEBUG) << "TF: " << tfid << " -->  ...unordered_map (delta) built = "
               << mDeltaBuildingUnorderedMap.RealTime() << ", cpuTime = "
               << mDeltaBuildingUnorderedMap.CpuTime();

    if (tfid % 6000 == 0) {
      LOG(INFO) << "Number of DPs received = " << nDPs;
      for (int idet = 0; idet < kNdetectors; idet++) {
        LOG(DEBUG) << "TF: " << tfid << " -->  starting processing...";
        mProcessing[idet].Start(mResetStopwatchProcessing);
        mDCSprocVect[idet].processMap(dcsmap, false);
        mProcessing[idet].Stop();
        LOG(DEBUG) << "TF: " << tfid << " -->  ...processing done: realTime = "
                   << mProcessing[idet].RealTime() << ", cpuTime = "
                   << mProcessing[idet].CpuTime();
      }
      mResetStopwatchProcessing = false; // from now on, we sum up the processing time
      mTFsProcessing++;
    } else {
      LOG(INFO) << "Number of DPs received (delta map) = " << nDPsDelta;
      if (mProcessFullDeltaMap) {
        for (int idet = 0; idet < kNdetectors; idet++) {
          LOG(DEBUG) << "TF: " << tfid << " -->  starting (delta) processing...";
          mDeltaProcessing[idet].Start(mResetStopwatchDeltaProcessing);
          mDCSprocVect[idet].processMap(dcsmapDelta, true);
          mDeltaProcessing[idet].Stop();
          LOG(DEBUG) << "TF: " << tfid << " -->  ...processing (delta) done: realTime = "
                     << mDeltaProcessing[idet].RealTime()
                     << ", cpuTime = " << mDeltaProcessing[idet].CpuTime();
        }
        mResetStopwatchDeltaProcessing = false; // from now on, we sum up the processing time
        mTFsDeltaProcessing++;
      } else {

        // processing per DP found in the map, to be done in case of a delta map

        LOG(DEBUG) << "TF: " << tfid << " -->  starting (delta) processing in detector loop...";
        for (const auto& dpcom : dcsmapDelta) {
          std::vector<int> detVect = mDetectorPid[dpcom.first];
          for (int idet = 0; idet < detVect.size(); idet++) {
            mDeltaProcessingDetLoop[idet].Start(mResetStopwatchDeltaProcessingDetLoop);
            mDCSprocVect[idet].processDP(dpcom);
            mDeltaProcessingDetLoop[idet].Stop();
          }
          mResetStopwatchDeltaProcessingDetLoop = false; // from now on, we sum up the processing time
        }
        for (int idet = 0; idet < kNdetectors; idet++) {
          LOG(DEBUG) << "TF: " << tfid << " -->  ...processing (delta) in detector loop done: realTime = "
                     << mDeltaProcessingDetLoop[idet].RealTime() << ", cpuTime = "
                     << mDeltaProcessingDetLoop[idet].CpuTime();
        }
        // now preparing CCDB object
        for (int idet = 0; idet < kNdetectors; idet++) {
          std::map<std::string, std::string> md;
          mDCSprocVect[idet].prepareCCDBobject(mDCSprocVect[idet].getCCDBSimpleMovingAverage(),
                                               mDCSprocVect[idet].getCCDBSimpleMovingAverageInfo(),
                                               mDCSprocVect[idet].getName() + "/TestDCS/SimpleMovingAverageDPs",
                                               tfid, md);
        }
        mTFsDeltaProcessingDetLoop++;
      }
    }
    sendOutput(pc.outputs());
    mFirstTF = false;
    mTFs++;
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(INFO) << "\n\nTIMING SUMMARY:\n";
    LOG(INFO) << "Number of processed TF: " << mTFs;
    LOG(INFO) << "Number of processed TF, processing full map: " << mTFsProcessing;
    LOG(INFO) << "Number of processed TF, processing delta map: " << mTFsDeltaProcessing;
    LOG(INFO) << "Number of processed TF, processing delta map per DP: " << mTFsDeltaProcessingDetLoop;
    LOG(INFO) << "Receiving binary data --> realTime = "
              << mReceiveBinaryData.RealTime() / mTFs << ", cpuTime = "
              << mReceiveBinaryData.CpuTime() / mTFs;
    LOG(INFO) << "Receiving binary data (delta) --> realTime = "
              << mDeltaReceiveBinaryData.RealTime() / mTFs << ", cpuTime = "
              << mDeltaReceiveBinaryData.CpuTime() / mTFs;
    LOG(INFO) << "Building unordered_map --> realTime = "
              << mBuildingUnorderedMap.RealTime() / mTFs << ", cpuTime = "
              << mBuildingUnorderedMap.CpuTime() / mTFs;
    LOG(INFO) << "Building unordered_map (delta) --> realTime = "
              << mDeltaBuildingUnorderedMap.RealTime() / mTFs << ", cpuTime = "
              << mDeltaBuildingUnorderedMap.CpuTime() / mTFs;
    for (int i = 0; i < kNdetectors; i++) {
      LOG(INFO) << " --> : Detector " << i;
      if (mTFsProcessing != 0) {
        LOG(INFO) << "Processing full map (average over " << mTFsProcessing << " TFs) --> realTime = "
                  << mProcessing[i].RealTime() / mTFsProcessing << ", cpuTime = "
                  << mProcessing[i].CpuTime() / mTFsProcessing;
      } else {
        LOG(INFO) << "Full DCS map was never processed";
      }
      if (mTFsDeltaProcessing != 0) {
        LOG(INFO) << "Processing full delta map (average over " << mTFsDeltaProcessing << " TFs) --> realTime = "
                  << mDeltaProcessing[i].RealTime() / mTFsDeltaProcessing << ", cpuTime = "
                  << mDeltaProcessing[i].CpuTime() / mTFsDeltaProcessing;
      } else {
        LOG(INFO) << "Full delta DCS map was never processed";
      }
      if (mTFsDeltaProcessingDetLoop != 0) {
        LOG(INFO) << "Processing delta map per DP (average over " << mTFsDeltaProcessingDetLoop
                  << " TFs) --> realTime = "
                  << mDeltaProcessingDetLoop[i].RealTime() / mTFsDeltaProcessingDetLoop << ", cpuTime = "
                  << mDeltaProcessingDetLoop[i].CpuTime() / mTFsDeltaProcessingDetLoop;
      } else {
        LOG(INFO) << "Delta map was never process DP by DP";
      }
    }
  }

 private:
  std::unordered_map<DPID, std::vector<int>> mDetectorPid;
  std::array<DCSProcessor, kNdetectors> mDCSprocVect;
  TStopwatch mReceiveBinaryData;
  TStopwatch mDeltaReceiveBinaryData;
  TStopwatch mBuildingUnorderedMap;
  TStopwatch mDeltaBuildingUnorderedMap;
  TStopwatch mProcessing[kNdetectors];
  TStopwatch mDeltaProcessing[kNdetectors];
  TStopwatch mDeltaProcessingDetLoop[kNdetectors];
  bool mProcessFullDeltaMap = false;
  bool mFirstTF = true;
  uint64_t mTFs = 0;
  uint64_t mTFsProcessing = 0;
  uint64_t mTFsDeltaProcessing = 0;
  uint64_t mTFsDeltaProcessingDetLoop = 0;
  bool mResetStopwatchProcessing = true;
  bool mResetStopwatchDeltaProcessing = true;
  bool mResetStopwatchDeltaProcessingDetLoop = true;

  //________________________________________________________________
  void sendOutput(framework::DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // copied from LHCClockCalibratorSpec.cxx
    using clbUtils = o2::calibration::Utils;
    for (int idet = 0; idet < kNdetectors; idet++) {
      const auto& payload = mDCSprocVect[idet].getCCDBSimpleMovingAverage();
      auto& info = mDCSprocVect[idet].getCCDBSimpleMovingAverageInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

      output.snapshot(framework::Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, 0}, *image.get());
      output.snapshot(framework::Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, 0}, info);
    }
  }
}; // end class
} // namespace dcs

namespace framework
{

DataProcessorSpec getDCSDataProcessorSpec()
{

  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload});
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo});

  return DataProcessorSpec{
    "dcs-data-processor",
    Inputs{{"input", "DCS", "DATAPOINTS"}, {"inputDelta", "DCS", "DATAPOINTSdelta"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::dcs::DCSDataProcessor>()},
    Options{
      {"max-cycles-no-full-map", VariantType::Int64, 6000ll, {"max num of cycles between the sending of 2 full maps"}},
      {"process-full-delta-map", VariantType::Bool, false, {"to process the delta map as a whole instead of per DP"}}}};
}

} // namespace framework
} // namespace o2

#endif
