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

#include <vector>
#include <string>
#include "fmt/format.h"

#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"

#include "DataFormatsTPC/TPCSectorHeader.h"
#include "Headers/DataHeader.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"

#include "TPCQC/Clusters.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/DigitDump.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCWorkflow/CalibProcessingHelper.h"
using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tpc
{

class TPCDigitDumpDevice : public o2::framework::Task
{
 public:
  TPCDigitDumpDevice(const std::vector<int>& sectors) : mSectors(sectors) {}

  void init(o2::framework::InitContext& ic) final
  {
    // parse command line arguments
    mMaxEvents = static_cast<uint32_t>(ic.options().get<int>("max-events"));
    mUseOldSubspec = ic.options().get<bool>("use-old-subspec");
    const bool createOccupancyMaps = ic.options().get<bool>("create-occupancy-maps");
    mForceQuit = ic.options().get<bool>("force-quit");
    mCheckDuplicates = ic.options().get<bool>("check-for-duplicates");
    mRemoveDuplicates = ic.options().get<bool>("remove-duplicates");

    if (mUseOldSubspec) {
      LOGP(info, "Using old subspecification (CruId << 16) | ((LinkId + 1) << (CruEndPoint == 1 ? 8 : 0))");
    }

    // set up ADC value filling
    mRawReader.createReader("");
    mDigitDump.init();
    mDigitDump.setInMemoryOnly();
    const auto pedestalFile = ic.options().get<std::string>("pedestal-file");
    if (pedestalFile.length()) {
      LOGP(info, "Setting pedestal file: {}", pedestalFile);
      mDigitDump.setPedestalAndNoiseFile(pedestalFile);
    }

    // set up cluster qc if requested
    if (createOccupancyMaps) {
      mClusterQC = std::make_unique<qc::Clusters>();
    }

    mRawReader.setADCDataCallback([this](const PadROCPos& padROCPos, const CRU& cru, const gsl::span<const uint32_t> data) -> int {
      const int timeBins = mDigitDump.update(padROCPos, cru, data);
      mDigitDump.setNumberOfProcessedTimeBins(std::max(mDigitDump.getNumberOfProcessedTimeBins(), size_t(timeBins)));
      return timeBins;
    });

    mRawReader.setLinkZSCallback([this](int cru, int rowInSector, int padInRow, int timeBin, float adcValue) -> bool {
      CRU cruID(cru);
      const PadRegionInfo& regionInfo = Mapper::instance().getPadRegionInfo(cruID.region());
      mDigitDump.updateCRU(cruID, rowInSector - regionInfo.getGlobalRowOffset(), padInRow, timeBin, adcValue);
      if (mClusterQC) {
        mClusterQC->fillADCValue(cru, rowInSector, padInRow, timeBin, adcValue);
      }
      return true;
    });
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // in case the maximum number of events was reached don't do further processing
    if (mReadyToQuit) {
      return;
    }

    auto& reader = mRawReader.getReaders()[0];
    mActiveSectors = calib_processing_helper::processRawData(pc.inputs(), reader, mUseOldSubspec);

    mDigitDump.incrementNEvents();
    LOGP(info, "Number of processed events: {} ({})", mDigitDump.getNumberOfProcessedEvents(), mMaxEvents);

    snapshotDigits(pc.outputs());

    if (mMaxEvents && (mDigitDump.getNumberOfProcessedEvents() >= mMaxEvents)) {
      LOGP(info, "Maximm number of events reached ({}), no more processing will be done", mMaxEvents);
      mReadyToQuit = true;
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
    if (mActiveSectors) {
      snapshotDigits(ec.outputs());
    }
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);

    if (mClusterQC) {
      dumpClusterQC();
    }
  }

 private:
  DigitDump mDigitDump;
  std::unique_ptr<qc::Clusters> mClusterQC;
  rawreader::RawReaderCRUManager mRawReader;
  uint32_t mMaxEvents{0};
  bool mReadyToQuit{false};
  bool mCalibDumped{false};
  bool mUseOldSubspec{false};
  bool mForceQuit{false};
  bool mCheckDuplicates{false};
  bool mRemoveDuplicates{false};
  uint64_t mActiveSectors{0};  ///< bit mask of active sectors
  std::vector<int> mSectors{}; ///< tpc sector configuration

  //____________________________________________________________________________
  void snapshotDigits(DataAllocator& output)
  {
    if (mCheckDuplicates || mRemoveDuplicates) {
      // iplicityly sorts
      mDigitDump.checkDuplicates(mRemoveDuplicates);
    } else {
      mDigitDump.sortDigits();
    }
    for (auto isector : mSectors) {
      o2::tpc::TPCSectorHeader header{isector};
      header.activeSectors = mActiveSectors;
      // digit for now are transported per sector, not per lane
      output.snapshot(Output{"TPC", "DIGITS", static_cast<SubSpecificationType>(isector), Lifetime::Timeframe, header},
                      mDigitDump.getDigits(isector));
    }
    mDigitDump.clearDigits();
    mActiveSectors = 0;
  }

  //____________________________________________________________________________
  void dumpClusterQC()
  {
    mClusterQC->analyse();
    mClusterQC->dumpToFile("ClusterQC.root");
  }
};

DataProcessorSpec getRawToDigitsSpec(int channel, const std::string inputSpec, std::vector<int> const& tpcSectors)
{
  using device = o2::tpc::TPCDigitDumpDevice;

  std::vector<OutputSpec> outputs;
  for (auto isector : tpcSectors) {
    outputs.emplace_back("TPC", "DIGITS", static_cast<SubSpecificationType>(isector), Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    fmt::format("tpc-raw-to-digits-{}", channel),
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{adaptFromTask<device>(tpcSectors)},
    Options{
      {"max-events", VariantType::Int, 0, {"maximum number of events to process"}},
      {"use-old-subspec", VariantType::Bool, false, {"use old subsecifiation definition"}},
      {"force-quit", VariantType::Bool, false, {"force quit after max-events have been reached"}},
      {"pedestal-file", VariantType::String, "", {"file with pedestals and noise for zero suppression"}},
      {"create-occupancy-maps", VariantType::Bool, false, {"create occupancy maps and store them to local root file for debugging"}},
      {"check-for-duplicates", VariantType::Bool, false, {"check if duplicate digits exist and only report them"}},
      {"remove-duplicates", VariantType::Bool, false, {"check if duplicate digits exist and remove them"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace tpc
} // namespace o2
