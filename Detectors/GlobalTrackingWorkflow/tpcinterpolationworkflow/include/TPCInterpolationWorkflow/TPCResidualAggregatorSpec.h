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

#ifndef O2_TPC_RESIDUALAGGREGATORSPEC_H
#define O2_TPC_RESIDUALAGGREGATORSPEC_H

/// \file   TPCResidualAggregatorSpec.h
/// \brief DPL device for collecting and binning TPC cluster residuals
/// \author Ole Schmidt

#include "DetectorsCalibration/Utils.h"
#include "SpacePoints/TrackResiduals.h"
#include "SpacePoints/ResidualAggregator.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class ResidualAggregatorDevice : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    int minEnt = ic.options().get<int>("min-entries");
    long slotLength = ic.options().get<long>("tf-per-slot");
    if (slotLength == -1) {
      slotLength = std::numeric_limits<long>::max();
    }
    int updateInterval = ic.options().get<int>("updateInterval");
    int delay = ic.options().get<int>("max-delay");
    mAggregator = std::make_unique<o2::tpc::ResidualAggregator>(minEnt);
    // TODO mAggregator should get an option to set the binning externally (expose TrackResiduals::setBinning methods to user? as command line option?)
    mAggregator->setSlotLength(slotLength);
    mAggregator->setMaxSlotsDelay(delay);
    mAggregator->setCheckIntervalInfiniteSlot(updateInterval);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    auto data = pc.inputs().get<gsl::span<o2::tpc::TrackResiduals::UnbinnedResid>>("input");
    LOG(debug) << "Processing TF " << tfcounter << " with " << data.size() << " unbinned residuals";
    mAggregator->process(tfcounter, data);
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(info) << "Finalizing calibration for end of stream";
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mAggregator->checkSlotsToFinalize(INFINITE_TF);
    mAggregator.reset(); // must invoke destructor manually here, otherwise we get a segfault
  }

 private:
  std::unique_ptr<o2::tpc::ResidualAggregator> mAggregator;
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTPCResidualAggregatorSpec()
{
  return DataProcessorSpec{
    "residual-aggregator",
    Inputs{{"input", "GLO", "UNBINNEDRES"}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<o2::calibration::ResidualAggregatorDevice>()},
    Options{
      {"tf-per-slot", VariantType::Int, 6'000, {"number of TFs per calibration time slot (put -1 for infinite slot length)"}},
      {"updateInterval", VariantType::Int, 6'000, {"update interval in number of TFs in case slot length is infinite"}},
      {"max-delay", VariantType::Int, 10, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 0, {"minimum number of entries on average per voxel"}}}};
}

} // namespace framework
} // namespace o2

#endif // O2_TPC_RESIDUALAGGREGATORSPEC_H
