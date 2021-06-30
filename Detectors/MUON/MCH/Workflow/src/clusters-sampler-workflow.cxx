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

/// \file clusters-sampler-workflow.cxx
/// \brief Implementation of a DPL device to send clusters read from a binary file
///
/// \author Philippe Pillot, Subatech

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "MCHBase/ClusterBlock.h"
#include "DataFormatsMCH/Digit.h"

using namespace o2::framework;

//_________________________________________________________________________________________________
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  /// add workflow options. Note that customization needs to be declared before including Framework/runDataProcessing
  workflowOptions.emplace_back("global", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"assume the read clusters are in global reference frame"});
}

#include "Framework/runDataProcessing.h"

using namespace o2::mch;

class ClusterSamplerTask
{
 public:
  //_________________________________________________________________________________________________
  void init(InitContext& ic)
  {
    /// Get the input file from the context
    LOG(INFO) << "initializing cluster sampler";

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, std::ios::binary);
    if (!mInputFile.is_open()) {
      throw std::invalid_argument("cannot open input file" + inputFileName);
    }
    if (mInputFile.peek() == EOF) {
      throw std::length_error("input file is empty");
    }

    mNEventsPerTF = ic.options().get<int>("nEventsPerTF");
    if (mNEventsPerTF < 1) {
      throw std::invalid_argument("number of events per time frame must be >= 1");
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop cluster sampler";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(ProcessingContext& pc)
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
      throw std::length_error("invalid input");
    }

    // get the number of associated digits
    int nDigits(-1);
    mInputFile.read(reinterpret_cast<char*>(&nDigits), sizeof(int));
    if (mInputFile.fail()) {
      throw std::length_error("invalid input");
    }

    if (nClusters < 0 || nDigits < 0) {
      throw std::length_error("invalid input");
    }

    // fill clusters in O2 format, if any
    if (nClusters > 0) {
      int clusterOffset = clusters.size();
      clusters.resize(clusterOffset + nClusters);
      mInputFile.read(reinterpret_cast<char*>(&clusters[clusterOffset]), nClusters * sizeof(ClusterStruct));
      if (mInputFile.fail()) {
        throw std::length_error("invalid input");
      }
    } else {
      LOG(INFO) << "event is empty";
    }

    // skip the digits if any
    if (nDigits > 0) {
      mInputFile.seekg(nDigits * sizeof(Digit), std::ios::cur);
      if (mInputFile.fail()) {
        throw std::length_error("invalid input");
      }
    }

    return nClusters;
  }

  std::ifstream mInputFile{}; ///< input file
  int mNEventsPerTF = 1;      ///< number of events per time frame
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getClusterSamplerSpec(bool globalReferenceSystem)
{

  std::string spec = fmt::format("clusters:MCH/{}CLUSTERS/0", globalReferenceSystem ? "GLOBAL" : "");
  InputSpec itmp = o2::framework::select(spec.c_str())[0];

  return DataProcessorSpec{
    "ClusterSampler",
    Inputs{},
    Outputs{OutputSpec{{"rofs"}, "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe},
            DataSpecUtils::asOutputSpec(itmp)},
    AlgorithmSpec{adaptFromTask<ClusterSamplerTask>()},
    Options{{"infile", VariantType::String, "", {"input filename"}},
            {"nEventsPerTF", VariantType::Int, 1, {"number of events per time frame"}}}};
}

//_________________________________________________________________________________________________
WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  return WorkflowSpec{getClusterSamplerSpec(cc.options().get<bool>("global"))};
}
