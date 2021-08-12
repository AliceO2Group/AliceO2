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

// o2 includes
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
    const int nbins = std::max(10, ic.options().get<int>("nbins"));
    const int mindEdx = std::max(0.0f, ic.options().get<float>("min-dedx"));
    const int maxdEdx = std::max(10.0f, ic.options().get<float>("max-dedx"));
    const bool applyCuts = ic.options().get<bool>("apply-cuts");
    const float minP = ic.options().get<float>("min-momentum");
    const float maxP = ic.options().get<float>("max-momentum");
    const int minClusters = std::max(10, ic.options().get<int>("min-clusters"));
    const bool dumpData = ic.options().get<bool>("direct-file-dump");

    assert(minP < maxP);

    mCalib = std::make_unique<CalibdEdx>(nbins, mindEdx, maxdEdx, minP, maxP, minClusters);
    mCalib->setApplyCuts(applyCuts);

    mDumpToFile = dumpData;
  }

  void run(ProcessingContext& pc) final
  {
    const auto tfcounter = o2::header::get<DataProcessingHeader*>(pc.inputs().get("tracks").header)->startTime;
    const auto tracks = pc.inputs().get<gsl::span<TrackTPC>>("tracks");

    LOGP(info, "Processing TF {} with {} tracks", tfcounter, tracks.size());

    mCalib->fill(tracks);
    // sendOutput(pc.outputs());
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    LOGP(info, "Finalizing calibration");
    mCalib->finalize();
    mCalib->print();
    // sendOutput(eos.outputs());

    if (mDumpToFile) {
      mCalib->dumpToFile("calibdEdx.root");
    }
  }

 private:
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h
  }

  bool mDumpToFile{};
  std::unique_ptr<CalibdEdx> mCalib;
};

DataProcessorSpec getCalibdEdxSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{"TPC", "dEdxCalibData"});

  return DataProcessorSpec{
    "tpc-calib-dEdx",
    Inputs{
      InputSpec{"tracks", "TPC", "MIPS"},
    },
    outputs,
    adaptFromTask<CalibdEdxDevice>(),
    Options{
      {"apply-cuts", VariantType::Bool, false, {"enable tracks filter using cut values passed as options"}},
      {"min-momentum", VariantType::Float, 0.4f, {"minimum momentum cut"}},
      {"max-momentum", VariantType::Float, 0.6f, {"maximum momentum cut"}},
      {"min-clusters", VariantType::Int, 60, {"minimum number of clusters in a track"}},
      {"nbins", VariantType::Int, 100, {"number of bins for the dEdx histograms"}},
      {"min-dedx", VariantType::Float, 5.0f, {"minimum value for the dEdx histograms"}},
      {"max-dedx", VariantType::Float, 100.0f, {"maximum value for the dEdx histograms"}},
      {"direct-file-dump", VariantType::Bool, false, {"directly dump calibration to file"}}}};
}

} // namespace o2::tpc
