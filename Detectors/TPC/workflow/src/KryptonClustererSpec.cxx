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

#include <memory>

#include "Framework/Task.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Headers/DataHeader.h"

#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/KrCluster.h"
#include "TPCReconstruction/KrBoxClusterFinder.h"
#include "TPCWorkflow/KryptonClustererSpec.h"

using namespace o2::framework;
using namespace o2::header;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tpc
{

class KrBoxClusterFinderDevice : public o2::framework::Task
{
 public:
  KrBoxClusterFinderDevice() : mClusterFinder{std::make_unique<KrBoxClusterFinder>()} {}

  void init(o2::framework::InitContext& ic) final
  {
    mClusterFinder->init();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    for (auto const& inputRef : InputRecordWalker(pc.inputs())) {
      auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(inputRef);
      if (sectorHeader == nullptr) {
        LOGP(error, "sector header missing on header stack for input on ", inputRef.spec->binding);
        continue;
      }

      const int sector = sectorHeader->sector();
      auto inDigits = pc.inputs().get<gsl::span<o2::tpc::Digit>>(inputRef);

      mClusterFinder->loopOverSector(inDigits, sector);

      snapshotClusters(pc.outputs(), mClusterFinder->getClusters(), sector);

      LOGP(info, "processed sector {} with {} digits and {} reconstructed clusters", sector, inDigits.size(), mClusterFinder->getClusters().size());

      mClusterFinder->resetClusters();
    }

    ++mProcessedTFs;
    LOGP(info, "Number of processed time frames: {}", mProcessedTFs);
  }

 private:
  std::unique_ptr<KrBoxClusterFinder> mClusterFinder;
  uint32_t mProcessedTFs{0};

  //____________________________________________________________________________
  void snapshotClusters(DataAllocator& output, const std::vector<o2::tpc::KrCluster>& clusters, int sector)
  {
    o2::tpc::TPCSectorHeader header{sector};
    header.activeSectors = (0x1 << sector);
    output.snapshot(Output{gDataOriginTPC, "KRCLUSTERS", static_cast<SubSpecificationType>(sector), header}, clusters);
  }
};

o2::framework::DataProcessorSpec getKryptonClustererSpec()
{
  using device = o2::tpc::KrBoxClusterFinderDevice;

  std::vector<InputSpec> inputs{
    InputSpec{"digits", gDataOriginTPC, "DIGITS", 0, Lifetime::Timeframe},
  };

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(gDataOriginTPC, "KRCLUSTERS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-krypton-clusterer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{} // end Options
  };          // end DataProcessorSpec
}
} // namespace tpc
} // namespace o2
