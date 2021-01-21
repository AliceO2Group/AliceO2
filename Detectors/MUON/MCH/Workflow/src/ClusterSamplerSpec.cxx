// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterSamplerSpec.cxx
/// \brief Implementation of a data processor to read and send clusters
///
/// \author Philippe Pillot, Subatech

#include "ClusterSamplerSpec.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHBase/Digit.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class ClusterSamplerTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file from the context
    LOG(INFO) << "initializing cluster sampler";

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, ios::binary);
    if (!mInputFile.is_open()) {
      throw invalid_argument("cannot open input file" + inputFileName);
    }
    if (mInputFile.peek() == EOF) {
      throw length_error("input file is empty");
    }

    mNEventsPerTF = ic.options().get<int>("nEventsPerTF");
    if (mNEventsPerTF < 1) {
      throw invalid_argument("number of events per time frame must be >= 1");
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop cluster sampler";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// send the clusters of the next events in the current TF

    static uint32_t event(0);

    // reached eof
    if (mInputFile.peek() == EOF) {
      pc.services().get<ControlService>().endOfStream();
      //pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      return;
    }

    // create the output messages
    auto& rofs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"rofs"});
    auto& clusters = pc.outputs().make<std::vector<ClusterStruct>>(OutputRef{"clusters"});

    // loop over the requested number of events (or until eof) and fill the messages
    for (int iEvt = 0; iEvt < mNEventsPerTF && mInputFile.peek() != EOF; ++iEvt) {
      int nClusters = readOneEvent(clusters);
      rofs.emplace_back(o2::InteractionRecord{0, event++}, clusters.size() - nClusters, nClusters);
    }
  }

 private:
  //_________________________________________________________________________________________________
  int readOneEvent(std::vector<ClusterStruct, o2::pmr::polymorphic_allocator<ClusterStruct>>& clusters)
  {
    /// fill the internal buffer with the clusters of the current event

    // get the number of clusters
    int nClusters(-1);
    mInputFile.read(reinterpret_cast<char*>(&nClusters), sizeof(int));
    if (mInputFile.fail()) {
      throw length_error("invalid input");
    }

    // get the number of associated digits
    int nDigits(-1);
    mInputFile.read(reinterpret_cast<char*>(&nDigits), sizeof(int));
    if (mInputFile.fail()) {
      throw length_error("invalid input");
    }

    if (nClusters < 0 || nDigits < 0) {
      throw length_error("invalid input");
    }

    // fill clusters in O2 format, if any
    if (nClusters > 0) {
      int clusterOffset = clusters.size();
      clusters.resize(clusterOffset + nClusters);
      mInputFile.read(reinterpret_cast<char*>(&clusters[clusterOffset]), nClusters * sizeof(ClusterStruct));
      if (mInputFile.fail()) {
        throw length_error("invalid input");
      }
    } else {
      LOG(INFO) << "event is empty";
    }

    // skip the digits if any
    if (nDigits > 0) {
      mInputFile.seekg(nDigits * sizeof(Digit), std::ios::cur);
      if (mInputFile.fail()) {
        throw length_error("invalid input");
      }
    }

    return nClusters;
  }

  std::ifstream mInputFile{}; ///< input file
  int mNEventsPerTF = 1;      ///< number of events per time frame
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getClusterSamplerSpec()
{
  return DataProcessorSpec{
    "ClusterSampler",
    Inputs{},
    Outputs{OutputSpec{{"rofs"}, "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe},
            OutputSpec{{"clusters"}, "MCH", "CLUSTERS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<ClusterSamplerTask>()},
    Options{{"infile", VariantType::String, "", {"input filename"}},
            {"nEventsPerTF", VariantType::Int, 1, {"number of events per time frame"}}}};
}

} // end namespace mch
} // end namespace o2
