// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_TPCCALIBPEDESTALSPEC_H
#define O2_CALIBRATION_TPCCALIBPEDESTALSPEC_H

/// @file   TPCCalibPedestalSpec.h
/// @brief  TPC Pedestal calibration processor

#include <vector>
#include <string>
#include <chrono>

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"

#include "Headers/DataHeader.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"

#include "TPCBase/CDBInterface.h"
#include "TPCCalibration/CalibPedestal.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCWorkflow/CalibProcessingHelper.h"

using namespace o2::framework;
using namespace o2::tpc;
using clbUtils = o2::calibration::Utils;

namespace o2
{
namespace calibration
{

class TPCCalibPedestalDevice : public o2::framework::Task
{
 public:
  TPCCalibPedestalDevice(bool skipCalib) : mSkipCalib(skipCalib) {}

  void init(o2::framework::InitContext& ic) final
  {
    // set up ADC value filling
    mCalibPedestal.init(); // initialize configuration via configKeyValues
    mRawReader.createReader("");
    mRawReader.setADCDataCallback([this](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> int {
      const int timeBins = mCalibPedestal.update(padROCPos, cru, data);
      mCalibPedestal.setNumberOfProcessedTimeBins(std::max(mCalibPedestal.getNumberOfProcessedTimeBins(), size_t(timeBins)));
      return timeBins;
    });

    mMaxEvents = static_cast<uint32_t>(ic.options().get<int>("max-events"));
    mUseOldSubspec = ic.options().get<bool>("use-old-subspec");
    mForceQuit = ic.options().get<bool>("force-quit");
    mDirectFileDump = ic.options().get<bool>("direct-file-dump");
    if (mUseOldSubspec) {
      LOGP(info, "Using old subspecification (CruId << 16) | ((LinkId + 1) << (CruEndPoint == 1 ? 8 : 0))");
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // in case the maximum number of events was reached don't do further processing
    if (mReadyToQuit) {
      return;
    }

    auto& reader = mRawReader.getReaders()[0];
    calib_processing_helper::processRawData(pc.inputs(), reader, mUseOldSubspec);

    mCalibPedestal.incrementNEvents();
    LOGP(info, "Number of processed events: {} ({})", mCalibPedestal.getNumberOfProcessedEvents(), mMaxEvents);

    if ((mCalibPedestal.getNumberOfProcessedEvents() >= mMaxEvents) && !mCalibDumped) {
      LOGP(info, "Maximm number of events reached ({}), no more processing will be done", mMaxEvents);
      mReadyToQuit = true;
      dumpCalibData();
      if (mForceQuit) {
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
      } else {
        //pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      }
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");
    dumpCalibData();
    if (!mSkipCalib) {
      sendOutput(ec.outputs());
    }
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  CalibPedestal mCalibPedestal;
  rawreader::RawReaderCRUManager mRawReader;
  uint32_t mMaxEvents{100};
  bool mReadyToQuit{false};
  bool mCalibDumped{false};
  bool mUseOldSubspec{false};
  bool mForceQuit{false};
  bool mDirectFileDump{false};
  bool mSkipCalib{false};

  //____________________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    CDBStorage::MetaData_t md;

    // perhaps should be changed to time of the run
    const auto now = std::chrono::system_clock::now();
    long timeStart = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    long timeEnd = 99999999999999;

    std::array<const CalDet<float>*, 2> data = {&mCalibPedestal.getPedestal(), &mCalibPedestal.getNoise()};
    std::array<CDBType, 2> dataType = {CDBType::CalPedestal, CDBType::CalNoise};

    for (size_t i = 0; i < data.size(); ++i) {
      auto cal = data[i];
      o2::ccdb::CcdbObjectInfo w;
      auto image = o2::ccdb::CcdbApi::createObjectImage(cal, &w);

      w.setPath(CDBTypeMap.at(dataType[i]));
      w.setStartValidityTimestamp(timeStart);
      w.setEndValidityTimestamp(timeEnd);

      LOG(INFO) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();

      header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)i};
      output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, subSpec}, *image.get());
      output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, subSpec}, w);
    }
  }

  //____________________________________________________________________________
  void dumpCalibData()
  {
    if (mDirectFileDump && !mCalibDumped) {
      LOGP(info, "Dumping output");
      mCalibPedestal.analyse();
      mCalibPedestal.dumpToFile("pedestals.root");
      mCalibDumped = true;
    }
  }
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTPCCalibPedestalSpec(const std::string inputSpec, bool skipCalib)
{
  using device = o2::calibration::TPCCalibPedestalDevice;

  std::vector<OutputSpec> outputs;
  if (!skipCalib) {
    outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload});
    outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo});
  }

  return DataProcessorSpec{
    "calib-tpc-pedestal",
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{adaptFromTask<device>(skipCalib)},
    Options{
      {"max-events", VariantType::Int, 100, {"maximum number of events to process"}},
      {"use-old-subspec", VariantType::Bool, false, {"use old subsecifiation definition"}},
      {"force-quit", VariantType::Bool, false, {"force quit after max-events have been reached"}},
      {"direct-file-dump", VariantType::Bool, false, {"directly dump calibration to file"}},
    } // end Options
  };  // end DataProcessorSpec
}

} // namespace framework
} // namespace o2

#endif
