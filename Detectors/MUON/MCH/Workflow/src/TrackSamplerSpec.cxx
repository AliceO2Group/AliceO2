// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackSamplerSpec.cxx
/// \brief Implementation of a data processor to read and send tracks
///
/// \author Philippe Pillot, Subatech

#include "TrackSamplerSpec.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/OutputRef.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "DataFormatsMCH/TrackMCH.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHBase/TrackBlock.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TrackSamplerTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file from the context
    LOG(INFO) << "initializing track sampler";

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, ios::binary);
    if (!mInputFile.is_open()) {
      throw invalid_argument("Cannot open input file" + inputFileName);
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop track sampler";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// send the tracks with attached clusters of the current event

    // read the number of tracks at vertex, MCH tracks and attached clusters
    int nTracksAtVtx(-1);
    mInputFile.read(reinterpret_cast<char*>(&nTracksAtVtx), sizeof(int));
    if (mInputFile.fail()) {
      pc.services().get<ControlService>().endOfStream();
      return; // probably reached eof
    }
    int nMCHTracks(-1);
    mInputFile.read(reinterpret_cast<char*>(&nMCHTracks), sizeof(int));
    int nClusters(-1);
    mInputFile.read(reinterpret_cast<char*>(&nClusters), sizeof(int));

    if (nTracksAtVtx < 0 || nMCHTracks < 0 || nClusters < 0) {
      throw length_error("invalid data input");
    }
    if (nMCHTracks > 0 && nClusters == 0) {
      throw out_of_range("clusters are missing");
    }

    // create the output messages
    auto tracks = pc.outputs().make<TrackMCH>(OutputRef{"tracks"}, nMCHTracks);
    auto clusters = pc.outputs().make<ClusterStruct>(OutputRef{"clusters"}, nClusters);

    // skip the tracks at vertex if any
    if (nTracksAtVtx > 0) {
      mInputFile.seekg(nTracksAtVtx * sizeof(TrackAtVtxStruct), std::ios::cur);
    }

    // read the MCH tracks and the attached clusters
    if (nMCHTracks > 0) {
      mInputFile.read(reinterpret_cast<char*>(tracks.data()), tracks.size_bytes());
      mInputFile.read(reinterpret_cast<char*>(clusters.data()), clusters.size_bytes());
    }
  }

 private:
  struct TrackAtVtxStruct {
    TrackParamStruct paramAtVertex{};
    double dca = 0.;
    double rAbs = 0.;
    int mchTrackIdx = 0;
  };

  std::ifstream mInputFile{}; ///< input file
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackSamplerSpec(bool forTrackFitter)
{
  Outputs outputs{};
  if (forTrackFitter) {
    outputs.emplace_back(OutputLabel{"tracks"}, "MCH", "TRACKSIN", 0, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"clusters"}, "MCH", "TRACKCLUSTERSIN", 0, Lifetime::Timeframe);
  } else {
    outputs.emplace_back(OutputLabel{"tracks"}, "MCH", "TRACKS", 0, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"clusters"}, "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "TrackSampler",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TrackSamplerTask>()},
    Options{{"infile", VariantType::String, "", {"input filename"}}}};
}

} // end namespace mch
} // end namespace o2
