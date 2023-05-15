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

/// @file   CalDetMergerPublisherSpec.cxx
/// @brief  TPC CalDet merger and CCDB publisher
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
/// @author David Silvermyr

#include <bitset>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

#include <fmt/format.h>

#include "TMemFile.h"
#include "TFile.h"

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"

#include "Headers/DataHeader.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "TPCBase/CDBInterface.h"
#include "TPCBase/CalDet.h"
#include "TPCWorkflow/CalibRawPartInfo.h"
#include "TPCWorkflow/CalDetMergerPublisherSpec.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;
using namespace o2::tpc;
using clbUtils = o2::calibration::Utils;
using o2::header::gDataOriginTPC;

class CalDetMergerPublisherSpec : public o2::framework::Task
{
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

 public:
  CalDetMergerPublisherSpec(uint32_t lanes, bool skipCCDB, bool dumpAfterComplete = false) : mLanesToExpect(lanes), mCalibInfos(lanes), mSkipCCDB(skipCCDB), mPublishAfterComplete(dumpAfterComplete) {}

  void init(o2::framework::InitContext& ic) final
  {
    mForceQuit = ic.options().get<bool>("force-quit");
    mDirectFileDump = ic.options().get<bool>("direct-file-dump");
    mCheckCalibInfos = ic.options().get<bool>("check-calib-infos");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    int nSlots = pc.inputs().getNofParts(0);
    assert(pc.inputs().getNofParts(1) == nSlots);

    mRunNumber = processing_helpers::getRunNumber(pc);

    for (int isl = 0; isl < nSlots; isl++) {
      const auto calibInfo = pc.inputs().get<CalibRawPartInfo>("clbInfo", isl);
      const auto type = calibInfo.calibType;
      const auto pld = pc.inputs().get<gsl::span<char>>("clbPayload", isl); // this is actually an image of TMemFile
      const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().get("clbInfo", isl));
      const auto subSpec = dh->subSpecification;
      const int lane = subSpec >> 4;
      const int calibType = subSpec & 0xf;
      mCalibInfos[lane] = calibInfo;

      // const auto& path = wrp->getPath();
      TMemFile f("file", (char*)&pld[0], pld.size(), "READ");
      if (!f.IsZombie()) {
        auto calDetMap = f.Get<CalDetMap>("data");
        if (calDetMap) {
          if (mMergedCalDetsMap.size() == 0) {
            mCalDetMapType = CDBType(type);
            for (auto& [key, obj] : *calDetMap) {
              mMergedCalDetsMap[key] = obj;
            }
          } else {
            if (int(mCalDetMapType) != type) {
              LOGP(fatal, "received CalDetMap of different type for merging, previous: {}, present{}", CDBTypeMap.at(mCalDetMapType), CDBTypeMap.at(CDBType(type)));
            }
            for (auto& [key, obj] : *calDetMap) {
              mMergedCalDetsMap[key] += obj;
            }
          }
        }

        auto calDet = f.Get<o2::tpc::CalDet<float>>("data");
        if (calDet) {
          if (mMergedCalDets.find(type) == mMergedCalDets.end()) {
            mMergedCalDets[type] = *calDet;
          } else {
            mMergedCalDets[type] += *calDet;
          }
        }
      }
      f.Close();

      LOGP(info, "getting slot {}, subspec {:#8x}, lane {}, type {} ({}), firstTF {}, cycle {}", isl, subSpec, lane, calibType, type, calibInfo.tfIDInfo.tfCounter, calibInfo.publishCycle);
      // if (mReceivedLanes.test(lane)) {
      // LOGP(warning, "lane {} received multiple times", lane);
      // }
      mReceivedLanes.set(lane);
    }

    if (mReceivedLanes.count() == mLanesToExpect) {
      LOGP(info, "data of all lanes received");
      if (mPublishAfterComplete) {
        LOGP(info, "publishing after all data was received");
        sendOutput(pc.outputs());

        // reset calibration objects
        mMergedCalDetsMap.clear();
        for (auto& [type, object] : mMergedCalDets) {
          object = 0;
        }
      }
      mReceivedLanes.reset();
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");

    if (mReceivedLanes.count() == mLanesToExpect) {
      sendOutput(ec.outputs());
    } else {
      LOGP(info, "Received lanes {} does not match expected lanes {}, object already sent", mReceivedLanes.count(), mLanesToExpect);
    }
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  using dataType = o2::tpc::CalDet<float>;
  using CalDetMap = std::unordered_map<std::string, dataType>;
  std::bitset<128> mReceivedLanes;                  ///< counter for received lanes
  std::unordered_map<int, dataType> mMergedCalDets; ///< calibration data to merge
  std::vector<CalibRawPartInfo> mCalibInfos;        ///< calibration info of all partially sent data sets
  CalDetMap mMergedCalDetsMap;                      ///< calibration data to merge; Map
  CDBType mCalDetMapType;                           ///< calibration type of CalDetMap object
  uint64_t mRunNumber{0};                           ///< processed run number
  uint32_t mLanesToExpect{0};                       ///< number of expected lanes sending data
  bool mForceQuit{false};                           ///< for quit after processing finished
  bool mDirectFileDump{false};                      ///< directly dump the calibration data to file
  bool mPublishAfterComplete{false};                ///< dump calibration directly after data from all lanes received
  bool mSkipCCDB{false};                            ///< skip sending of calibration data
  bool mCheckCalibInfos{false};                     ///< check calib infos

  //____________________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    if (mCheckCalibInfos) {
      if (std::adjacent_find(mCalibInfos.begin(), mCalibInfos.end(), std::not_equal_to<>()) != mCalibInfos.end()) {
        LOGP(warning, "Different calib info found");
      }
    }

    // perhaps should be changed to time of the run
    const auto now = std::chrono::system_clock::now();
    const long timeStart = mCalibInfos[0].tfIDInfo.creation + mCalibInfos[0].publishCycle;
    const long timeEnd = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP;

    std::map<std::string, std::string> md;

    if (mMergedCalDetsMap.size() > 0) {
      o2::ccdb::CcdbObjectInfo w;
      auto image = o2::ccdb::CcdbApi::createObjectImage(&mMergedCalDetsMap, &w);

      w.setPath(CDBTypeMap.at(mCalDetMapType));
      w.setStartValidityTimestamp(timeStart);
      w.setEndValidityTimestamp(timeEnd);

      md = w.getMetaData();
      md[o2::base::NameConf::CCDBRunTag.data()] = std::to_string(mRunNumber);
      w.setMetaData(md);

      LOGP(info, "Sending object {}/{} of size {} bytes, valid for {} : {}", w.getPath(), w.getFileName(), image->size(), w.getStartValidityTimestamp(), w.getEndValidityTimestamp());

      o2::header::DataHeader::SubSpecificationType subSpec{(o2::header::DataHeader::SubSpecificationType)mCalDetMapType};
      output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "TPC_CALIB", subSpec}, *image.get());
      output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "TPC_CALIB", subSpec}, w);
    }

    for (auto& [type, object] : mMergedCalDets) {
      o2::ccdb::CcdbObjectInfo w;
      auto image = o2::ccdb::CcdbApi::createObjectImage(&object, &w);

      w.setPath(CDBTypeMap.at(CDBType(type)));
      w.setStartValidityTimestamp(timeStart);
      w.setEndValidityTimestamp(timeEnd);

      md = w.getMetaData();
      md[o2::base::NameConf::CCDBRunTag.data()] = std::to_string(mRunNumber);
      w.setMetaData(md);

      LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();

      o2::header::DataHeader::SubSpecificationType subSpec{(o2::header::DataHeader::SubSpecificationType)type};
      output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "TPC_CALIB", subSpec}, *image.get());
      output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "TPC_CALIB", subSpec}, w);
    }

    dumpCalibData();
  }

  //____________________________________________________________________________
  void dumpCalibData()
  {
    if (mDirectFileDump) {
      LOGP(info, "Dumping output to file");
      std::string fileName = "merged_CalDet.root";
      if (mMergedCalDetsMap.size()) {
        const auto& cdbType = CDBTypeMap.at(mCalDetMapType);
        const auto name = cdbType.substr(cdbType.rfind("/") + 1);
        fileName = fmt::format("merged_{}_{}_{}.root", name, mCalibInfos[0].tfIDInfo.tfCounter, mCalibInfos[0].publishCycle);
      }
      TFile f(fileName.data(), "recreate");
      for (auto& [key, object] : mMergedCalDetsMap) {
        f.WriteObject(&object, object.getName().data());
      }
      for (auto& [type, object] : mMergedCalDets) {
        f.WriteObject(&object, object.getName().data());
      }
    }
  }
};

o2::framework::DataProcessorSpec o2::tpc::getCalDetMergerPublisherSpec(uint32_t lanes, bool skipCCDB, bool dumpAfterComplete)
{
  std::vector<OutputSpec> outputs;
  if (!skipCCDB) {
    outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "TPC_CALIB"}, Lifetime::Sporadic);
    outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "TPC_CALIB"}, Lifetime::Sporadic);
  }

  std::vector<InputSpec> inputs;
  inputs.emplace_back("clbPayload", ConcreteDataTypeMatcher{gDataOriginTPC, "CLBPART"});
  inputs.emplace_back("clbInfo", ConcreteDataTypeMatcher{gDataOriginTPC, "CLBPARTINFO"});

  const std::string id = "calib-tpc-caldet-merger-publisher";

  return DataProcessorSpec{
    id.data(),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CalDetMergerPublisherSpec>(lanes, skipCCDB, dumpAfterComplete)},
    Options{
      {"force-quit", VariantType::Bool, false, {"force quit after max-events have been reached"}},
      {"direct-file-dump", VariantType::Bool, false, {"directly dump calibration to file"}},
      {"check-calib-infos", VariantType::Bool, false, {"make consistency check of calib infos"}},
    } // end Options
  };  // end DataProcessorSpec
}
