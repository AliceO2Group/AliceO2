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

/// \file CalibratorPadGainTracksSpec.h
/// \brief Workflow for the track based dE/dx gain map extraction on an aggregator node
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#ifndef O2_TPC_TPCCALIBRATORPADGAINTRACKSSPEC_H_
#define O2_TPC_TPCCALIBRATORPADGAINTRACKSSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "TPCCalibration/CalibratorPadGainTracks.h"
#include "CCDB/CcdbApi.h"
#include "CommonUtils/NameConf.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCBase/CDBInterface.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsCalibration/Utils.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;

namespace o2::tpc
{

class CalibratorPadGainTracksDevice : public Task
{
 public:
  CalibratorPadGainTracksDevice(std::shared_ptr<o2::base::GRPGeomRequest> req, const bool useLastExtractedMapAsReference) : mUseLastExtractedMapAsReference(useLastExtractedMapAsReference), mCCDBRequest(req) {}
  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    const auto slotLength = ic.options().get<uint32_t>("tf-per-slot");
    const auto maxDelay = ic.options().get<uint32_t>("max-delay");
    const int minEntries = ic.options().get<int>("min-entries");
    const int gainNorm = ic.options().get<int>("gainNorm");
    const bool debug = ic.options().get<bool>("file-dump");
    const auto lowTrunc = ic.options().get<float>("lowTrunc");
    const auto upTrunc = ic.options().get<float>("upTrunc");
    const auto minAcceptedRelgain = ic.options().get<float>("minAcceptedRelgain");
    const auto maxAcceptedRelgain = ic.options().get<float>("maxAcceptedRelgain");
    const int minEntriesMean = ic.options().get<int>("minEntriesMean");

    mCalibrator = std::make_unique<CalibratorPadGainTracks>();
    mCalibrator->setMinEntries(minEntries);
    mCalibrator->setSlotLength(slotLength);
    mCalibrator->setMaxSlotsDelay(maxDelay);
    mCalibrator->setTruncationRange(lowTrunc, upTrunc);
    mCalibrator->setRelGainRange(minAcceptedRelgain, maxAcceptedRelgain);
    mCalibrator->setWriteDebug(debug);
    mCalibrator->setNormalizationType(static_cast<CalibPadGainTracksBase::NormType>(gainNorm));
    mCalibrator->setUseLastExtractedMapAsReference(mUseLastExtractedMapAsReference);
    mCalibrator->setMinEntriesMean(minEntriesMean);
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    const auto histomaps = pc.inputs().get<CalibPadGainTracksBase::DataTHistos*>("gainhistos");
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
    mCalibrator->process(*histomaps.get());
    const auto& infoVec = mCalibrator->getTFinterval();
    LOGP(info, "Created {} objects for TF {} and time stamp {}", infoVec.size(), mCalibrator->getCurrentTFInfo().tfCounter, mCalibrator->getCurrentTFInfo().creation);

    if (mCalibrator->hasCalibrationData()) {
      mRunNumber = mCalibrator->getCurrentTFInfo().runNumber;
      mCreationTime = mCalibrator->getCurrentTFInfo().creation;
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    LOGP(info, "Finalizing calibration");
    mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    sendOutput(eos.outputs());
  }

 private:
  void sendOutput(DataAllocator& output)
  {
    auto calibrations = std::move(*mCalibrator).getCalibs();

    for (uint32_t iCalib = 0; iCalib < calibrations.size(); ++iCalib) {
      const auto& calib = calibrations[iCalib];
      const auto& infoVec = mCalibrator->getTFinterval();
      const auto firstTF = infoVec[iCalib].first;
      const auto lastTF = infoVec[iCalib].second;
      LOGP(info, "Writing pad-by-pad gain map to CCDB for TF {} to {}", firstTF, lastTF);
      o2::ccdb::CcdbObjectInfo ccdbInfo(CDBTypeMap.at(CDBType::CalPadGainResidual), std::string{}, std::string{}, std::map<std::string, std::string>{{"runNumber", std::to_string(mRunNumber)}}, mCreationTime, o2::calibration::INFINITE_TF);
      auto imageIDCDelta = o2::ccdb::CcdbApi::createObjectImage(&calib, &ccdbInfo);
      LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfo.getPath(), ccdbInfo.getFileName(), imageIDCDelta->size(), ccdbInfo.getStartValidityTimestamp(), ccdbInfo.getEndValidityTimestamp());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibResGain", iCalib}, *imageIDCDelta.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibResGain", iCalib}, ccdbInfo);
    }
    mCalibrator->initOutput(); // empty the outputs after they are send
  }

  const bool mUseLastExtractedMapAsReference{false};    ///< whether to use the last extracted gain map as a reference gain map
  std::unique_ptr<CalibratorPadGainTracks> mCalibrator; ///< calibrator object for creating the pad-by-pad gain map
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  uint64_t mRunNumber{0};    ///< processed run number
  uint64_t mCreationTime{0}; ///< creation time of current TF
};

/// create a processor spec
o2::framework::DataProcessorSpec getTPCCalibPadGainTracksSpec(const bool useLastExtractedMapAsReference)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibResGain"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibResGain"}, Lifetime::Sporadic);

  std::vector<InputSpec> inputs{{"gainhistos", "TPC", "TRACKGAINHISTOS", 0, Lifetime::Sporadic}};
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    "tpc-calibrator-gainmap-tracks",
    inputs,
    outputs,
    adaptFromTask<CalibratorPadGainTracksDevice>(ccdbRequest, useLastExtractedMapAsReference),
    Options{
      {"tf-per-slot", VariantType::UInt32, 100u, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::UInt32, 3u, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 0, {"minimum entries per pad-by-pad histogram which are required"}},
      {"lowTrunc", VariantType::Float, 0.05f, {"lower truncation range for calculating the rel gain"}},
      {"upTrunc", VariantType::Float, 0.6f, {"upper truncation range for calculating the rel gain"}},
      {"minAcceptedRelgain", VariantType::Float, 0.1f, {"minimum accpeted relative gain (if the gain is below this value it will be set to 1)"}},
      {"maxAcceptedRelgain", VariantType::Float, 2.f, {"maximum accpeted relative gain (if the gain is above this value it will be set to 1)"}},
      {"gainNorm", VariantType::Int, 2, {"normalization method for the extracted gain map: 0=no normalization, 1=median per stack, 2=median per region"}},
      {"minEntriesMean", VariantType::Int, 10, {"minEntries minimum number of entries in pad-by-pad histogram to calculate the mean"}},
      {"file-dump", VariantType::Bool, false, {"directly write calibration to a file"}}}};
}

} // namespace o2::tpc

#endif
