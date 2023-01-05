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
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/Digit.h"

using namespace o2::framework;

//_________________________________________________________________________________________________
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  /// add workflow options. Note that customization needs to be declared before including Framework/runDataProcessing
  workflowOptions.emplace_back("global", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"assume the read clusters are in global reference frame"});
  workflowOptions.emplace_back("no-digits", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"do not look for digits"});
}

#include "Framework/runDataProcessing.h"

using namespace o2::mch;

class ClusterSamplerTask
{
 public:
  //_________________________________________________________________________________________________
  ClusterSamplerTask(bool doDigits) : mDoDigits(doDigits) {}

  //_________________________________________________________________________________________________
  void init(InitContext& ic)
  {
    /// Get the input file from the context
    LOG(info) << "initializing cluster sampler";

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
      LOG(info) << "stop cluster sampler";
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
      // pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      return;
    }

    // create the output messages
    auto& rofs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"rofs"});
    auto& clusters = pc.outputs().make<std::vector<Cluster>>(OutputRef{"clusters"});
    std::vector<Digit, o2::pmr::polymorphic_allocator<Digit>>* digits(nullptr);
    if (mDoDigits) {
      digits = &pc.outputs().make<std::vector<Digit>>(OutputRef{"digits"});
    }

    // loop over the requested number of events (or until eof) and fill the messages
    for (int iEvt = 0; iEvt < mNEventsPerTF && mInputFile.peek() != EOF; ++iEvt) {
      int nClusters = readOneEvent(clusters, digits);
      rofs.emplace_back(o2::InteractionRecord{0, event++}, clusters.size() - nClusters, nClusters);
    }
  }

 private:
  //_________________________________________________________________________________________________
  int readOneEvent(std::vector<Cluster, o2::pmr::polymorphic_allocator<Cluster>>& clusters,
                   std::vector<Digit, o2::pmr::polymorphic_allocator<Digit>>* digits)
  {
    /// fill the internal buffers with the clusters and digits of the current event

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

    // stop here in case of empty event
    if (nClusters == 0) {
      if (nDigits > 0) {
        throw std::length_error("invalid input");
      }
      LOG(info) << "event is empty";
      return 0;
    }

    // fill clusters in O2 format
    int clusterOffset = clusters.size();
    clusters.resize(clusterOffset + nClusters);
    mInputFile.read(reinterpret_cast<char*>(&clusters[clusterOffset]), nClusters * sizeof(Cluster));
    if (mInputFile.fail()) {
      throw std::length_error("invalid input");
    }

    // read digits if requested or skip them if any
    if (mDoDigits) {
      if (nDigits == 0) {
        throw std::length_error("missing digits");
      }
      int digitOffset = digits->size();
      digits->resize(digitOffset + nDigits);
      mInputFile.read(reinterpret_cast<char*>(&(*digits)[digitOffset]), nDigits * sizeof(Digit));
      if (mInputFile.fail()) {
        throw std::length_error("invalid input");
      }
      for (auto itCluster = clusters.begin() + clusterOffset; itCluster < clusters.end(); ++itCluster) {
        itCluster->firstDigit += digitOffset;
      }
    } else if (nDigits > 0) {
      mInputFile.seekg(nDigits * sizeof(Digit), std::ios::cur);
      if (mInputFile.fail()) {
        throw std::length_error("invalid input");
      }
    }

    return nClusters;
  }

  std::ifstream mInputFile{}; ///< input file
  bool mDoDigits = false;     ///< read the associated digits
  int mNEventsPerTF = 1;      ///< number of events per time frame
};

//_________________________________________________________________________________________________
DataProcessorSpec getClusterSamplerSpec(const char* specName, bool global, bool doDigits)
{
  std::vector<OutputSpec> outputSpecs{};
  outputSpecs.emplace_back(OutputSpec{{"rofs"}, "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe});
  auto clusterDesc = global ? o2::header::DataDescription{"GLOBALCLUSTERS"} : o2::header::DataDescription{"CLUSTERS"};
  outputSpecs.emplace_back(OutputSpec{{"clusters"}, "MCH", clusterDesc, 0, Lifetime::Timeframe});
  if (doDigits) {
    outputSpecs.emplace_back(OutputSpec{{"digits"}, "MCH", "CLUSTERDIGITS", 0, Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    specName,
    Inputs{},
    outputSpecs,
    AlgorithmSpec{adaptFromTask<ClusterSamplerTask>(doDigits)},
    Options{{"infile", VariantType::String, "", {"input filename"}},
            {"nEventsPerTF", VariantType::Int, 1, {"number of events per time frame"}}}};
}

//_________________________________________________________________________________________________
WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  bool global = cc.options().get<bool>("global");
  bool doDigits = !cc.options().get<bool>("no-digits");
  return WorkflowSpec{getClusterSamplerSpec("mch-cluster-sampler", global, doDigits)};
}
