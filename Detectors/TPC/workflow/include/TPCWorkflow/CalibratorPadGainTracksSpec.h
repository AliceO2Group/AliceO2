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
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/NameConf.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCBase/CDBInterface.h"
#include "TPCWorkflow/ProcessingHelpers.h"

using namespace o2::framework;

namespace o2::tpc
{

class CalibratorPadGainTracksDevice : public Task
{
 public:
  void init(framework::InitContext& ic) final
  {
    const int slotLength = ic.options().get<int>("tf-per-slot");
    const int maxDelay = ic.options().get<int>("max-delay");
    const int minEntries = ic.options().get<int>("min-entries");
    const bool debug = ic.options().get<bool>("file-dump");
    const auto lowTrunc = ic.options().get<float>("lowTrunc");
    const auto upTrunc = ic.options().get<float>("upTrunc");

    mCalibrator = std::make_unique<CalibratorPadGainTracks>();
    mCalibrator->setMinEntries(minEntries);
    mCalibrator->setSlotLength(slotLength);
    mCalibrator->setMaxSlotsDelay(maxDelay);
    mCalibrator->setTruncationRange(lowTrunc, upTrunc);
    mCalibrator->setWriteDebug(debug);

    // setting up the CCDB
    mDBapi.init(ic.options().get<std::string>("ccdb-uri")); // or http://localhost:8080 for a local installation
    mWriteToDB = mDBapi.isHostReachable() ? true : false;
  }

  void run(ProcessingContext& pc) final
  {
    const auto tfcounter = o2::header::get<DataProcessingHeader*>(pc.inputs().get("gainhistos").header)->startTime;
    const auto histomaps = pc.inputs().get<CalibPadGainTracksBase::DataTHistos*>("gainhistos");
    mCalibrator->process(tfcounter, *histomaps.get());
    const auto& infoVec = mCalibrator->getTFinterval();
    LOGP(info, "Created {} objects for TF {}", infoVec.size(), tfcounter);

    if (mCalibrator->hasCalibrationData()) {
      mRunNumber = processing_helpers::getRunNumber(pc);
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    LOGP(info, "Finalizing calibration");
    constexpr calibration::TFType INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput(eos.outputs());
  }

 private:
  void sendOutput(DataAllocator& output)
  {
    const auto& calibrations = mCalibrator->getCalibs();
    for (uint32_t iCalib = 0; iCalib < calibrations.size(); ++iCalib) {
      const auto& calib = calibrations[iCalib];
      const auto& infoVec = mCalibrator->getTFinterval();
      const auto firstTF = infoVec[iCalib].first;
      const auto lastTF = infoVec[iCalib].second;
      // store in CCDB
      if (mWriteToDB) {
        LOGP(info, "Writing pad-by-pad gain map to CCDB for TF {} to {}", firstTF, lastTF);
        mMetadata["runNumber"] = std::to_string(mRunNumber);
        mDBapi.storeAsTFileAny<o2::tpc::CalPad>(&calib, CDBTypeMap.at(CDBType::CalPadGainResidual), mMetadata, firstTF, lastTF + 1);
      }
    }
    mCalibrator->initOutput(); // empty the outputs after they are send
  }

  std::unique_ptr<CalibratorPadGainTracks> mCalibrator; ///< calibrator object for creating the pad-by-pad gain map
  bool mWriteToDB{};                                    ///< flag if writing to CCDB will be done
  o2::ccdb::CcdbApi mDBapi;                             ///< API for storing the gain map in the CCDB
  std::map<std::string, std::string> mMetadata;         ///< meta data of the stored object in CCDB
  uint64_t mRunNumber{0};                               ///< processed run number
};

/// create a processor spec
o2::framework::DataProcessorSpec getTPCCalibPadGainTracksSpec()
{
  std::vector<OutputSpec> outputs;
  return DataProcessorSpec{
    "tpc-calibrator-gainmap-tracks",
    Inputs{
      InputSpec{"gainhistos", "TPC", "TRACKGAINHISTOS"},
    },
    outputs,
    adaptFromTask<CalibratorPadGainTracksDevice>(),
    Options{
      {"ccdb-uri", VariantType::String, o2::base::NameConf::getCCDBServer(), {"URI for the CCDB access."}},
      {"tf-per-slot", VariantType::Int, 100, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int, 3, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 50, {"minimum entries per pad-by-pad histogram which are required"}},
      {"lowTrunc", VariantType::Float, 0.05f, {"lower truncation range for calculating the rel gain"}},
      {"upTrunc", VariantType::Float, 0.6f, {"upper truncation range for calculating the rel gain"}},
      {"file-dump", VariantType::Bool, false, {"directly write calibration to a file"}}}};
}

} // namespace o2::tpc

#endif
