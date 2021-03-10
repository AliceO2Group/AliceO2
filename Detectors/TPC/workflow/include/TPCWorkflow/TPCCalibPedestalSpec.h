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
#include <fmt/format.h>

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"

#include "CommonUtils/MemFileHelper.h"
#include "Headers/DataHeader.h"
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
namespace tpc
{

class TPCCalibPedestalDevice : public o2::framework::Task
{
 public:
  TPCCalibPedestalDevice(int lane, const std::vector<int>& sectors) : mLane{lane}, mSectors(sectors) {}

  void init(o2::framework::InitContext& ic) final
  {
    // set up ADC value filling
    // TODO: clean up to not go via RawReaderCRUManager
    mCalibPedestal.init(); // initialize configuration via configKeyValues
    mRawReader.createReader("");

    mRawReader.setADCDataCallback([this](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> int {
      const int timeBins = mCalibPedestal.update(padROCPos, cru, data);
      mCalibPedestal.setNumberOfProcessedTimeBins(std::max(mCalibPedestal.getNumberOfProcessedTimeBins(), size_t(timeBins)));
      return timeBins;
    });

    mRawReader.setLinkZSCallback([this](int cru, int rowInSector, int padInRow, int timeBin, float adcValue) -> bool {
      CRU cruID(cru);
      mCalibPedestal.updateROC(cruID.roc(), rowInSector - (rowInSector > 62) * 63, padInRow, timeBin, adcValue);
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
    sendOutput(ec.outputs());
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  CalibPedestal mCalibPedestal;
  rawreader::RawReaderCRUManager mRawReader;
  uint32_t mMaxEvents{100};    ///< maximum number of events to process
  int mLane{0};                ///< lane number of processor
  std::vector<int> mSectors{}; ///< sectors to process in this instance
  bool mReadyToQuit{false};    ///< if processor is ready to quit
  bool mCalibDumped{false};    ///< if calibration object already dumped
  bool mUseOldSubspec{false};  ///< use the old subspec definition
  bool mForceQuit{false};      ///< for quit after processing finished
  bool mDirectFileDump{false}; ///< directly dump the calibration data to file

  //____________________________________________________________________________
  void sendOutput(DataAllocator& output)
  {

    std::array<const CalDet<float>*, 2> data = {&mCalibPedestal.getPedestal(), &mCalibPedestal.getNoise()};
    std::array<CDBType, 2> dataType = {CDBType::CalPedestal, CDBType::CalNoise};

    for (size_t i = 0; i < data.size(); ++i) {
      auto cal = data[i];
      auto image = o2::utils::MemFileHelper::createFileImage(cal, typeid(*cal), cal->getName(), "data");
      int type = int(dataType[i]);

      header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)i};
      output.snapshot(Output{clbUtils::gDataOriginCLB, "TPCCLBPART", subSpec}, *image.get());
      output.snapshot(Output{clbUtils::gDataOriginCLB, "TPCCLBPARTINFO", subSpec}, type);
    }
  }

  //____________________________________________________________________________
  void dumpCalibData()
  {
    if (mDirectFileDump && !mCalibDumped) {
      LOGP(info, "Dumping output");
      mCalibPedestal.analyse();
      mCalibPedestal.dumpToFile(fmt::format("pedestals_{:02}.root", mLane));
      mCalibDumped = true;
    }
  }
};

DataProcessorSpec getTPCCalibPedestalSpec(const std::string inputSpec, int ilane = 0, std::vector<int> sectors = {})
{
  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, "TPCCLBPART"});
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, "TPCCLBPARTINFO"});

  const auto id = fmt::format("calib-tpc-pedestal-{:02}", ilane);
  return DataProcessorSpec{
    id.data(),
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{adaptFromTask<TPCCalibPedestalDevice>(ilane, sectors)},
    Options{
      {"max-events", VariantType::Int, 100, {"maximum number of events to process"}},
      {"use-old-subspec", VariantType::Bool, false, {"use old subsecifiation definition"}},
      {"force-quit", VariantType::Bool, false, {"force quit after max-events have been reached"}},
      {"direct-file-dump", VariantType::Bool, false, {"directly dump calibration to file"}},
    } // end Options
  };  // end DataProcessorSpec
}

} // namespace tpc
} // namespace o2

#endif
