// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_TPCCALIBRAWSPEC_H
#define O2_CALIBRATION_TPCCALIBRAWSPEC_H

/// @file   TPCCalibPawSpec.h
/// @brief  TPC Raw Pedestal Pulser calibration processor

#include <vector>
#include <string>
#include <chrono>
#include <fmt/format.h>

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"

#include "CommonUtils/MemFileHelper.h"
#include "Headers/DataHeader.h"

#include "TPCBase/CDBInterface.h"
#include "TPCCalibration/CalibPedestal.h"
#include "TPCCalibration/CalibPulser.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCWorkflow/CalibProcessingHelper.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2
{
namespace tpc
{

template <class T = CalibPedestal>
class TPCCalibRawDevice : public o2::framework::Task
{
 public:
  TPCCalibRawDevice(uint32_t lane, const std::vector<int>& sectors, uint32_t publishAfterTFs) : mLane{lane}, mSectors(sectors), mPublishAfter(publishAfterTFs) {}

  void init(o2::framework::InitContext& ic) final
  {
    // set up ADC value filling
    // TODO: clean up to not go via RawReaderCRUManager
    mCalib.init(); // initialize configuration via configKeyValues
    mRawReader.createReader("");

    mRawReader.setADCDataCallback([this](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> int {
      const int timeBins = mCalib.update(padROCPos, cru, data);
      mCalib.setNumberOfProcessedTimeBins(std::max(mCalib.getNumberOfProcessedTimeBins(), size_t(timeBins)));
      return timeBins;
    });

    mRawReader.setLinkZSCallback([this](int cru, int rowInSector, int padInRow, int timeBin, float adcValue) -> bool {
      CRU cruID(cru);
      mCalib.updateROC(cruID.roc(), rowInSector - (rowInSector > 62) * 63, padInRow, timeBin, adcValue);
      return true;
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
    calib_processing_helper::processRawData(pc.inputs(), reader, mUseOldSubspec, mSectors);

    mCalib.incrementNEvents();
    const auto nTFs = mCalib.getNumberOfProcessedEvents();
    LOGP(info, "Number of processed TFs: {} ({})", nTFs, mMaxEvents);

    if ((mPublishAfter && (nTFs % mPublishAfter) == 0)) {
      LOGP(info, "Publishing after {} TFs", nTFs);
      mCalib.analyse();
      dumpCalibData();
      sendOutput(pc.outputs());
    }

    if (mMaxEvents && (nTFs >= mMaxEvents) && !mCalibDumped) {
      LOGP(info, "Maximm number of TFs reached ({}), no more processing will be done", mMaxEvents);
      mReadyToQuit = true;
      mCalib.analyse();
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
    mCalib.analyse();
    dumpCalibData();
    sendOutput(ec.outputs());
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  T mCalib;
  rawreader::RawReaderCRUManager mRawReader;
  uint32_t mMaxEvents{0};      ///< maximum number of events to process
  uint32_t mPublishAfter{0};   ///< number of events after which to dump the calibration
  uint32_t mLane{0};           ///< lane number of processor
  std::vector<int> mSectors{}; ///< sectors to process in this instance
  bool mReadyToQuit{false};    ///< if processor is ready to quit
  bool mCalibDumped{false};    ///< if calibration object already dumped
  bool mUseOldSubspec{false};  ///< use the old subspec definition
  bool mForceQuit{false};      ///< for quit after processing finished
  bool mDirectFileDump{false}; ///< directly dump the calibration data to file

  //____________________________________________________________________________
  void sendOutput(DataAllocator& output)
  {

    if constexpr (std::is_same_v<T, CalibPedestal>) {
      std::array<const CalDet<float>*, 2> data = {&mCalib.getPedestal(), &mCalib.getNoise()};
      std::array<CDBType, 2> dataType = {CDBType::CalPedestal, CDBType::CalNoise};

      for (size_t i = 0; i < data.size(); ++i) {
        auto cal = data[i];
        auto image = o2::utils::MemFileHelper::createFileImage(cal, typeid(*cal), cal->getName(), "data");
        int type = int(dataType[i]);

        header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)((mLane << 4) + i)};
        output.snapshot(Output{gDataOriginTPC, "CLBPART", subSpec}, *image.get());
        output.snapshot(Output{gDataOriginTPC, "CLBPARTINFO", subSpec}, type);
      }
    } else if constexpr (std::is_same_v<T, CalibPulser>) {
      std::unordered_map<std::string, CalDet<float>> pulserCalib;
      pulserCalib["T0"] = mCalib.getT0();
      pulserCalib["Width"] = mCalib.getWidth();
      pulserCalib["Qtot"] = mCalib.getQtot();
      int type = int(CDBType::CalPulser);
      auto image = o2::utils::MemFileHelper::createFileImage(&pulserCalib, typeid(pulserCalib), "CalibPulser", "data");

      header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)((mLane << 4))};
      output.snapshot(Output{gDataOriginTPC, "CLBPART", subSpec}, *image.get());
      output.snapshot(Output{gDataOriginTPC, "CLBPARTINFO", subSpec}, type);
    }
  }

  //____________________________________________________________________________
  void dumpCalibData()
  {
    if (mDirectFileDump && !mCalibDumped) {
      LOGP(info, "Dumping output");
      std::string desc = "pedestals";
      if constexpr (std::is_same_v<T, CalibPulser>) {
        desc = "pulser";
      }
      mCalib.dumpToFile(fmt::format("{}-{:02}.root", desc, mLane));
      mCalibDumped = true;
    }
  }
};

template <class T = CalibPedestal>
DataProcessorSpec getTPCCalibRawSpec(const std::string inputSpec, uint32_t ilane = 0, std::vector<int> sectors = {}, uint32_t publishAfterTFs = 0)
{
  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, "CLBPART"});
  outputs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, "CLBPARTINFO"});

  std::string desc = "pedestal";
  if constexpr (std::is_same_v<T, CalibPulser>) {
    desc = "pulser";
  }
  const auto id = fmt::format("calib-tpc-{}-{:02}", desc, ilane);

  return DataProcessorSpec{
    id.data(),
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{adaptFromTask<TPCCalibRawDevice<T>>(ilane, sectors, publishAfterTFs)},
    Options{
      {"max-events", VariantType::Int, 0, {"maximum number of events to process"}},
      {"use-old-subspec", VariantType::Bool, false, {"use old subsecifiation definition"}},
      {"force-quit", VariantType::Bool, false, {"force quit after max-events have been reached"}},
      {"direct-file-dump", VariantType::Bool, false, {"directly dump calibration to file"}},
    } // end Options
  };  // end DataProcessorSpec
}

} // namespace tpc
} // namespace o2

#endif
