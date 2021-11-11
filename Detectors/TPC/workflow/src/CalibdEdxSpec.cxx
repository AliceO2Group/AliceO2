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

/// \file CalibdEdxSpec.cxx
/// \brief Workflow for time based dE/dx calibration.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#include "TPCWorkflow/CalibdEdxSpec.h"

#include <vector>
#include <memory>

// o2 includes
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCCalibration/CalibdEdx.h"

using namespace o2::framework;

namespace o2::tpc
{

class CalibdEdxDevice : public Task
{
 public:
  void init(framework::InitContext& ic) final
  {
    const int slotLength = ic.options().get<int>("tf-per-slot");
    const int maxDelay = ic.options().get<int>("max-delay");
    const int minEntries = std::max(50, ic.options().get<int>("min-entries"));
    const int nbins = std::max(10, ic.options().get<int>("nbins"));
    const bool applyCuts = ic.options().get<bool>("apply-cuts");
    const float minP = ic.options().get<float>("min-momentum");
    const float maxP = ic.options().get<float>("max-momentum");
    const int minClusters = std::max(10, ic.options().get<int>("min-clusters"));
    const bool dumpData = ic.options().get<bool>("direct-file-dump");

    assert(minP < maxP);

    mCalibrator = std::make_unique<tpc::CalibdEdx>(nbins, minEntries, minP, maxP, minClusters);
    mCalibrator->setApplyCuts(applyCuts);

    mCalibrator->setSlotLength(slotLength);
    mCalibrator->setMaxSlotsDelay(maxDelay);

    if (dumpData) {
      mCalibrator->enableDebugOutput("calib_dEdx.root");
    }
  }

  void run(ProcessingContext& pc) final
  {
    const auto tfcounter = o2::header::get<DataProcessingHeader*>(pc.inputs().get("tracks").header)->startTime;
    const auto tracks = pc.inputs().get<gsl::span<tpc::TrackTPC>>("tracks");

    LOG(INFO) << "Processing TF " << tfcounter << " with " << tracks.size() << " tracks";

    mCalibrator->process(tfcounter, tracks);
    sendOutput(pc.outputs());

    const auto& infoVec = mCalibrator->getInfoVector();
    LOG(INFO) << "Created " << infoVec.size() << " objects for TF " << tfcounter;
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    LOG(INFO) << "Finalizing calibration";
    constexpr calibration::TFType INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput(eos.outputs());

    if (mCalibrator->hasDebugOutput()) {
      mCalibrator->finalizeDebugOutput();
    }
  }

 private:
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    using clbUtils = o2::calibration::Utils;
    const auto& payloadVec = mCalibrator->getMIPVector();
    auto& infoVec = mCalibrator->getInfoVector(); // use non-const version as we update it
    assert(payloadVec.size() == infoVec.size());

    // FIXME: not sure about this
    for (unsigned int i = 0; i < payloadVec.size(); i++) {
      auto& entry = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &entry);
      LOG(INFO) << "Sending object " << entry.getPath() << "/" << entry.getFileName() << " of size " << image->size()
                << " bytes, valid for " << entry.getStartValidityTimestamp() << " : " << entry.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_MIPS", i}, *image.get()); // vector<char>
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_MIPS", i}, entry);        // root-serialized
    }
    if (!payloadVec.empty()) {
      mCalibrator->initOutput(); // reset the outputs once they are already sent
    }
  }

  std::unique_ptr<CalibdEdx> mCalibrator;
};

DataProcessorSpec getCalibdEdxSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_MIPS"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_MIPS"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-calib-dEdx",
    Inputs{
      InputSpec{"tracks", "TPC", "MIPS"},
    },
    outputs,
    adaptFromTask<CalibdEdxDevice>(),
    Options{
      {"tf-per-slot", VariantType::Int, 100, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int, 3, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 100, {"minimum number of entries to fit single time slot"}},
      {"apply-cuts", VariantType::Bool, false, {"enable tracks filter using cut values passed as options"}},
      {"min-momentum", VariantType::Float, 0.4f, {"minimum momentum cut"}},
      {"max-momentum", VariantType::Float, 0.6f, {"maximum momentum cut"}},
      {"min-clusters", VariantType::Int, 60, {"minimum number of clusters in a track"}},
      {"nbins", VariantType::Int, 200, {"number of bins for stored"}},
      {"direct-file-dump", VariantType::Bool, false, {"directly dump calibration to file"}}}};
}

} // namespace o2::tpc
