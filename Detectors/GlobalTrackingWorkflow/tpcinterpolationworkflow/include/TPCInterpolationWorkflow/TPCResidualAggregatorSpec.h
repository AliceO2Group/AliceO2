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
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/RawDeviceService.h"
#include <fairmq/Device.h>

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class ResidualAggregatorDevice : public o2::framework::Task
{
 public:
  ResidualAggregatorDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    int minEnt = ic.options().get<int>("min-entries");
    auto slotLength = ic.options().get<uint32_t>("tf-per-slot");
    if (slotLength == 0) {
      slotLength = o2::calibration::INFINITE_TF;
    }
    auto updateInterval = ic.options().get<uint32_t>("updateInterval");
    auto delay = ic.options().get<uint32_t>("max-delay");

    std::string outputDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("output-dir"));
    std::string metaFileDir = ic.options().get<std::string>("meta-output-dir");
    bool storeMetaFile = false;
    if (metaFileDir != "/dev/null") {
      metaFileDir = o2::utils::Str::rectifyDirectory(metaFileDir);
      storeMetaFile = true;
    }
    mAggregator = std::make_unique<o2::tpc::ResidualAggregator>(minEnt);
    mAggregator->setOutputDir(outputDir);
    if (storeMetaFile) {
      mAggregator->setMetaFileOutputDir(metaFileDir);
    }
    // TODO mAggregator should get an option to set the binning externally (expose TrackResiduals::setBinning methods to user? as command line option?)
    mAggregator->setSlotLength(slotLength);
    mAggregator->setMaxSlotsDelay(delay);
    mAggregator->setCheckIntervalInfiniteSlot(updateInterval);
    mAggregator->initOutput();
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    auto data = pc.inputs().get<gsl::span<o2::tpc::TrackResiduals::UnbinnedResid>>("input");
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mAggregator->getCurrentTFInfo());
    if (!isLHCPeriodSet) {
      // read the LHC period information only once
      const std::string NAStr = "NA";
      auto LHCPeriodStr = pc.services().get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("LHCPeriod", NAStr);
      if (LHCPeriodStr == NAStr) {
        const char* months[12] = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};
        time_t now = time(nullptr);
        auto ltm = gmtime(&now);
        LHCPeriodStr = months[ltm->tm_mon];
        LOG(warning) << "LHCPeriod is not available, using current month " << LHCPeriodStr;
      }
      mAggregator->setLHCPeriod(LHCPeriodStr);
      isLHCPeriodSet = true;
    }
    LOG(debug) << "Processing TF " << mAggregator->getCurrentTFInfo().tfCounter << " with " << data.size() << " unbinned residuals";
    mAggregator->process(data);
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(info) << "Finalizing calibration for end of stream";
    mAggregator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    mAggregator.reset(); // must invoke destructor manually here, otherwise we get a segfault
  }

 private:
  std::unique_ptr<o2::tpc::ResidualAggregator> mAggregator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  bool isLHCPeriodSet{false};
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTPCResidualAggregatorSpec()
{
  std::vector<InputSpec> inputs{{"input", "GLO", "UNBINNEDRES"}};
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    "residual-aggregator",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<o2::calibration::ResidualAggregatorDevice>(ccdbRequest)},
    Options{
      {"tf-per-slot", VariantType::UInt32, 6'000u, {"number of TFs per calibration time slot (put 0 for infinite slot length)"}},
      {"updateInterval", VariantType::UInt32, 6'000u, {"update interval in number of TFs in case slot length is infinite"}},
      {"max-delay", VariantType::UInt32, 10u, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 0, {"minimum number of entries on average per voxel"}},
      {"output-dir", VariantType::String, "none", {"Output directory for residuals, must exist"}},
      {"meta-output-dir", VariantType::String, "/dev/null", {"Residuals metadata output directory, must exist (if not /dev/null)"}}}};
}

} // namespace framework
} // namespace o2

#endif // O2_TPC_RESIDUALAGGREGATORSPEC_H
