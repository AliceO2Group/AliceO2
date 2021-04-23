// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <memory>
#include <vector>
#include <string>
#include <algorithm>
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

#include "TPCBase/RDHUtils.h"
#include "TPCBase/Mapper.h"
#include "TPCReconstruction/KrBoxClusterFinder.h"
#include "TPCReconstruction/KrCluster.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCWorkflow/CalibProcessingHelper.h"
#include "TPCWorkflow/KryptonClustererSpec.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tpc
{

class KrBoxClusterFinderDevice : public o2::framework::Task
{
 public:
  KrBoxClusterFinderDevice(int lane, const std::vector<int>& sectors) : mClusters(36), mLane{lane}, mSectors(sectors), mClusterFinder{std::make_unique<KrBoxClusterFinder>()} {}

  void init(o2::framework::InitContext& ic) final
  {
    // set up ADC value filling
    mRawReader.createReader("");

    mRawReader.setLinkZSCallback([this](int cru, int rowInSector, int padInRow, int timeBin, float adcValue) -> bool {
      const int sector = cru / 10;
      if ((mLastSector > -1) && (sector != mLastSector)) {
        LOGP(debug, "analysing sector {} ({})", mLastSector, sector);
        mClusterFinder->findLocalMaxima(true);
        LOGP(info, "found {} clusters in sector {}", mClusterFinder->getClusters().size(), mLastSector);
        std::swap(mClusters[mLastSector], mClusterFinder->getClusters());
        mClusterFinder->resetADCMap();
        mClusterFinder->resetClusters();
      }

      mClusterFinder->fillADCValue(cru, rowInSector, padInRow, timeBin, adcValue);

      mLastSector = sector;
      return true;
    });

    mMaxEvents = static_cast<uint32_t>(ic.options().get<int>("max-events"));
    mForceQuit = ic.options().get<bool>("force-quit");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // in case the maximum number of events was reached don't do further processing
    if (mReadyToQuit) {
      return;
    }

    std::for_each(mClusters.begin(), mClusters.end(), [](auto& cl) { cl.clear(); });

    auto& reader = mRawReader.getReaders()[0];
    mActiveSectors = calib_processing_helper::processRawData(pc.inputs(), reader, false, mSectors);

    // analyse the final sector
    if (mLastSector > -1) {
      LOGP(debug, "analysing sector {}", mLastSector);
      mClusterFinder->findLocalMaxima(true);
      if (mClusterFinder->getClusters().size()) {
        LOGP(info, "found {} clusters in sector {}", mClusterFinder->getClusters().size(), mLastSector);
        std::swap(mClusters[mLastSector], mClusterFinder->getClusters());
      }
    }
    mClusterFinder->resetADCMap();
    mClusterFinder->resetClusters();

    ++mProcessedTFs;
    LOGP(info, "Number of processed time frames: {} ({})", mProcessedTFs, mMaxEvents);

    snapshotClusters(pc.outputs());

    // TODO: is this still needed?
    if (mMaxEvents && (mProcessedTFs >= mMaxEvents)) {
      LOGP(info, "Maximm number of time frames reached ({}), no more processing will be done", mMaxEvents);
      mReadyToQuit = true;
      if (mForceQuit) {
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
      } else {
        pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      }
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");
    if (mActiveSectors) {
      snapshotClusters(ec.outputs());
    }
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  std::vector<std::vector<KrCluster>> mClusters;
  std::unique_ptr<KrBoxClusterFinder> mClusterFinder;
  int mLastSector{-1};
  rawreader::RawReaderCRUManager mRawReader;
  int mLane{0};                ///< lane number of processor
  std::vector<int> mSectors{}; ///< sectors to process in this instance
  uint32_t mMaxEvents{0};
  uint32_t mProcessedTFs{0};
  bool mReadyToQuit{false};
  bool mCalibDumped{false};
  bool mForceQuit{false};
  uint64_t mActiveSectors{0}; ///< bit mask of active sectors

  //____________________________________________________________________________
  void snapshotClusters(DataAllocator& output)
  {
    for (const int sector : mSectors) {
      o2::tpc::TPCSectorHeader header{sector};
      header.activeSectors = mActiveSectors;
      // digit for now are transported per sector, not per lane
      output.snapshot(Output{"TPC", "KRCLUSTERS", static_cast<SubSpecificationType>(sector), Lifetime::Timeframe, header},
                      mClusters[sector]);
    }
    mActiveSectors = 0;
  }
};

o2::framework::DataProcessorSpec getKryptonClustererSpec(const std::string inputSpec, int ilane, std::vector<int> const& sectors)
{
  using device = o2::tpc::KrBoxClusterFinderDevice;

  std::vector<OutputSpec> outputs;
  for (auto isector : sectors) {
    outputs.emplace_back("TPC", "KRCLUSTERS", static_cast<SubSpecificationType>(isector), Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    fmt::format("tpc-krypton-clusterer-{}", ilane),
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ilane, sectors)},
    Options{
      {"max-events", VariantType::Int, 0, {"maximum number of events to process"}},
      {"force-quit", VariantType::Bool, false, {"force quit after max-events have been reached"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace tpc
} // namespace o2
