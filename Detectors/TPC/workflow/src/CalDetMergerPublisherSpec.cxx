// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CalDetMergerPublisherSpec.cxx
/// @brief  TPC CalDet merger and CCDB publisher
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

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
#include "TPCWorkflow/CalDetMergerPublisherSpec.h"

using namespace o2::framework;
using namespace o2::tpc;
using clbUtils = o2::calibration::Utils;

class CalDetMergerPublisherSpec : public o2::framework::Task
{
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

 public:
  CalDetMergerPublisherSpec(bool skipCCDB) : mSkipCCDB(skipCCDB) {}

  void init(o2::framework::InitContext& ic) final
  {
    mForceQuit = ic.options().get<bool>("force-quit");
    mDirectFileDump = ic.options().get<bool>("direct-file-dump");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    int nSlots = pc.inputs().getNofParts(0);
    assert(pc.inputs().getNofParts(1) == nSlots);

    LOGP(info, "CalDetMergerPublisherSpec run");
    for (int isl = 0; isl < nSlots; isl++) {
      const auto type = pc.inputs().get<int>("clbInfo", isl);
      const auto pld = pc.inputs().get<gsl::span<char>>("clbPayload", isl); // this is actually an image of TMemFile

      //const auto& path = wrp->getPath();
      TMemFile f("file", (char*)&pld[0], pld.size(), "READ");
      if (!f.IsZombie()) {
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

      LOGP(info, "getting slot {}", isl);
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "CalDetMergerPublisherSpec endOfStream");

    dumpCalibData();
    sendOutput(ec.outputs());
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  using dataType = o2::tpc::CalDet<float>;
  std::unordered_map<int, dataType> mMergedCalDets; ///< calibration data to merge
  bool mForceQuit{false};                           ///< for quit after processing finished
  bool mDirectFileDump{false};                      ///< directly dump the calibration data to file
  bool mCalibDumped{false};                         ///< if calibration object already dumped
  bool mSkipCCDB{false};                            ///< skip sending of calibration data

  //____________________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    //CDBStorage::MetaData_t md;

    // perhaps should be changed to time of the run
    const auto now = std::chrono::system_clock::now();
    const long timeStart = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    const long timeEnd = 99999999999999;

    for (auto& [type, object] : mMergedCalDets) {
      o2::ccdb::CcdbObjectInfo w;
      auto image = o2::ccdb::CcdbApi::createObjectImage(&object, &w);

      w.setPath(CDBTypeMap.at(CDBType(type)));
      w.setStartValidityTimestamp(timeStart);
      w.setEndValidityTimestamp(timeEnd);

      LOG(INFO) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();

      o2::header::DataHeader::SubSpecificationType subSpec{(o2::header::DataHeader::SubSpecificationType)type};
      output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, subSpec}, *image.get());
      output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, subSpec}, w);
    }
  }

  //____________________________________________________________________________
  void dumpCalibData()
  {
    if (mDirectFileDump && !mCalibDumped) {
      LOGP(info, "CalDetMergerPublisherSpec Dumping output");
      TFile f("merged_CalDet.root", "recreate");
      for (auto& [type, object] : mMergedCalDets) {
        f.WriteObject(&object, object.getName().data());
      }
      mCalibDumped = true;
    }
  }
};

o2::framework::DataProcessorSpec o2::tpc::getCalDetMergerPublisherSpec(bool skipCCDB)
{
  std::vector<OutputSpec> outputs;
  if (!skipCCDB) {
    outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload});
    outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo});
  }

  std::vector<InputSpec> inputs;
  inputs.emplace_back("clbPayload", ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, "TPCCLBPART"});
  inputs.emplace_back("clbInfo", ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, "TPCCLBPARTINFO"});

  const std::string id = "calib-tpc-caldet-merger-publisher";

  return DataProcessorSpec{
    id.data(),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CalDetMergerPublisherSpec>(skipCCDB)},
    Options{
      {"force-quit", VariantType::Bool, false, {"force quit after max-events have been reached"}},
      {"direct-file-dump", VariantType::Bool, false, {"directly dump calibration to file"}},
    } // end Options
  };  // end DataProcessorSpec
}
