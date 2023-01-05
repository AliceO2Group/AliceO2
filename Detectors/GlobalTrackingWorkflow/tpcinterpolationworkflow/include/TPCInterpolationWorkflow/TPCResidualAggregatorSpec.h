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
#include <chrono>

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace calibration
{

class ResidualAggregatorDevice : public o2::framework::Task
{
 public:
  ResidualAggregatorDevice(std::shared_ptr<o2::base::GRPGeomRequest> req, bool trackInput, bool ctpInput, bool writeUnbinnedResiduals, bool writeBinnedResiduals, bool writeTrackData, std::shared_ptr<o2::globaltracking::DataRequest> dataRequest) : mCCDBRequest(req), mTrackInput(trackInput), mCTPInput(ctpInput), mWriteUnbinnedResiduals(writeUnbinnedResiduals), mWriteBinnedResiduals(writeBinnedResiduals), mWriteTrackData(writeTrackData), mDataRequest(dataRequest) {}

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    int minEnt = ic.options().get<int>("min-entries");
    auto slotLength = ic.options().get<uint32_t>("sec-per-slot");
    bool useInfiniteSlotLength = false;
    if (slotLength == 0) {
      useInfiniteSlotLength = true;
    }
    auto updateInterval = ic.options().get<uint32_t>("updateInterval");
    auto delay = ic.options().get<uint32_t>("max-delay");

    // should we write meta files for epn2eos?
    bool storeMetaFile = false;
    std::string metaFileDir = ic.options().get<std::string>("meta-output-dir");
    if (metaFileDir != "/dev/null") {
      metaFileDir = o2::utils::Str::rectifyDirectory(metaFileDir);
      storeMetaFile = true;
    }

    // where should the ROOT output be written? in case its set to /dev/null
    // we don't write anything, also no meta files of course
    bool writeOutput = true;
    std::string outputDir = ic.options().get<std::string>("output-dir");
    if (outputDir != "/dev/null") {
      outputDir = o2::utils::Str::rectifyDirectory(outputDir);
    } else {
      writeOutput = false;
      storeMetaFile = false;
    }

    LOGP(info, "Creating aggregator with {} entries per voxel minimum. Output file writing enabled: {}, meta file writing enabled: {}",
         minEnt, writeOutput, storeMetaFile);
    mAggregator = std::make_unique<o2::tpc::ResidualAggregator>(minEnt);
    if (writeOutput) {
      mAggregator->setOutputDir(outputDir);
    }
    if (storeMetaFile) {
      mAggregator->setMetaFileOutputDir(metaFileDir);
    }

    int autosave = ic.options().get<int>("autosave-interval");
    mAggregator->setAutosaveInterval(autosave);
    // TODO mAggregator should get an option to set the binning externally (expose TrackResiduals::setBinning methods to user? as command line option?)
    mAggregator->setMaxSlotsDelay(delay);
    if (useInfiniteSlotLength) {
      mAggregator->setSlotLength(o2::calibration::INFINITE_TF);
      mAggregator->setCheckIntervalInfiniteSlot(updateInterval);
    } else {
      mAggregator->setSlotLengthInSeconds(slotLength);
    }
    mAggregator->setWriteBinnedResiduals(mWriteBinnedResiduals);
    mAggregator->setWriteUnbinnedResiduals(mWriteUnbinnedResiduals);
    mAggregator->setWriteTrackData(mWriteTrackData);
    mAggregator->setCompression(ic.options().get<int>("compression"));
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto runStartTime = std::chrono::high_resolution_clock::now();
    o2::globaltracking::RecoContainer recoCont;
    recoCont.collectData(pc, *mDataRequest);
    updateTimeDependentParams(pc);
    std::chrono::duration<double, std::milli> ccdbUpdateTime = std::chrono::high_resolution_clock::now() - runStartTime;

    // assume that the orbit reset time (given here in ms) can change within a run
    auto orbitResetTime = o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS();

    // we always require the unbinned residuals and the associated track references
    auto residualsData = pc.inputs().get<gsl::span<o2::tpc::UnbinnedResid>>("unbinnedRes");
    auto trackRefs = pc.inputs().get<gsl::span<o2::tpc::TrackDataCompact>>("trackRefs");

    // track data input is optional
    const gsl::span<const o2::tpc::TrackData>* trkDataPtr = nullptr;
    using trkDataType = std::decay_t<decltype(pc.inputs().get<gsl::span<o2::tpc::TrackData>>(""))>;
    std::optional<trkDataType> trkData;
    if (mTrackInput) {
      trkData.emplace(pc.inputs().get<gsl::span<o2::tpc::TrackData>>("trkData"));
      trkDataPtr = &trkData.value();
    }
    // CTP lumi input (optional)
    const o2::ctp::LumiInfo* lumi = nullptr;
    using lumiDataType = std::decay_t<decltype(pc.inputs().get<o2::ctp::LumiInfo>(""))>;
    std::optional<lumiDataType> lumiInput;
    if (mCTPInput) {
      recoCont.getCTPLumi();
      lumiInput = recoCont.getCTPLumi();
      lumi = &lumiInput.value();
    }

    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mAggregator->getCurrentTFInfo());
    LOG(info) << "Processing TF " << mAggregator->getCurrentTFInfo().tfCounter << " with " << trkData->size() << " tracks and " << residualsData.size() << " unbinned residuals associated to them";
    mAggregator->process(residualsData, trackRefs, orbitResetTime, trkDataPtr, lumi);
    std::chrono::duration<double, std::milli> runDuration = std::chrono::high_resolution_clock::now() - runStartTime;
    LOGP(info, "Duration for run method: {} ms. From this taken for time dependent param update: {} ms",
         std::chrono::duration_cast<std::chrono::milliseconds>(runDuration).count(),
         std::chrono::duration_cast<std::chrono::milliseconds>(ccdbUpdateTime).count());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(info) << "Finalizing calibration for end of stream";
    mAggregator->checkSlotsToFinalize();
    mAggregator.reset(); // must invoke destructor manually here, otherwise we get a segfault
  }

 private:
  void updateTimeDependentParams(ProcessingContext& pc)
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    static bool initOnceDone = false;
    if (!initOnceDone) {
      initOnceDone = true;
      mAggregator->setDataTakingContext(pc.services().get<DataTakingContext>());
    }
  }
  std::unique_ptr<o2::tpc::ResidualAggregator> mAggregator; ///< the TimeSlotCalibration device
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  std::shared_ptr<o2::globaltracking::DataRequest> mDataRequest; ///< optional CTP input
  bool mTrackInput{false};             ///< flag whether to expect track data as input
  bool mCTPInput{false};               ///< flag whether to expect luminosity input from CTP
  bool mWriteBinnedResiduals{false};   ///< flag, whether to write binned residuals to output file
  bool mWriteUnbinnedResiduals{false}; ///< flag, whether to write unbinned residuals to output file
  bool mWriteTrackData{false};         ///< flag, whether to write track data to output file
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTPCResidualAggregatorSpec(bool trackInput, bool ctpInput, bool writeUnbinnedResiduals, bool writeBinnedResiduals, bool writeTrackData)
{
  std::shared_ptr<o2::globaltracking::DataRequest> dataRequest = std::make_shared<o2::globaltracking::DataRequest>();
  if (ctpInput) {
    dataRequest->requestClusters(GID::getSourcesMask("CTP"), false);
  }
  auto& inputs = dataRequest->inputs;
  inputs.emplace_back("unbinnedRes", "GLO", "UNBINNEDRES");
  inputs.emplace_back("trackRefs", "GLO", "TRKREFS");
  if (trackInput) {
    inputs.emplace_back("trkData", "GLO", "TRKDATA");
  }
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
    AlgorithmSpec{adaptFromTask<o2::calibration::ResidualAggregatorDevice>(ccdbRequest, trackInput, ctpInput, writeUnbinnedResiduals, writeBinnedResiduals, writeTrackData, dataRequest)},
    Options{
      {"sec-per-slot", VariantType::UInt32, 600u, {"number of seconds per calibration time slot (put 0 for infinite slot length)"}},
      {"updateInterval", VariantType::UInt32, 6'000u, {"update interval in number of TFs (only used in case slot length is infinite)"}},
      {"max-delay", VariantType::UInt32, 1u, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 0, {"minimum number of entries on average per voxel"}},
      {"compression", VariantType::Int, 505, {"ROOT compression setting for output file (see TFile documentation for meaning of this number)"}},
      {"output-dir", VariantType::String, "none", {"Output directory for residuals. Defaults to current working directory. Output is disabled in case set to /dev/null"}},
      {"meta-output-dir", VariantType::String, "/dev/null", {"Residuals metadata output directory, must exist (if not /dev/null)"}},
      {"autosave-interval", VariantType::Int, 0, {"Write output to file for every n-th TF. 0 means this feature is OFF"}}}};
}

} // namespace framework
} // namespace o2

#endif // O2_TPC_RESIDUALAGGREGATORSPEC_H
