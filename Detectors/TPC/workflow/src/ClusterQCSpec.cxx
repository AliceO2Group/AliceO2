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

/// \file ClusterQCSpec.cxx
/// \brief Workflow to run clusterQC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <algorithm>
#include <iterator>
#include <vector>
#include <memory>
#include <random>

// o2 includes
#include "TPCQC/Clusters.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "DataFormatsTPC/Constants.h"

#include "TPCWorkflow/ClusterQCSpec.h"

using namespace o2::framework;
using namespace o2::tpc::constants;

namespace o2::tpc
{

class ClusterQCDevice : public Task
{
 public:
  void init(framework::InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& eos) final;

 private:
  void sendOutput(DataAllocator& output);
  void endInterval();

  unsigned int mProcessEveryNthTF{1}; ///< process every Nth TF only
  uint32_t mTFCounter{0};             ///< counter to keep track of the TFs
  uint32_t mFirstTF{0};               ///< first time frame number of analysis interval
  uint32_t mLastTF{0};                ///< last time frame number of analysis interval
  uint64_t mFirstCreation{0};         ///< first creation time of analysis interval
  uint64_t mLastCreation{0};          ///< last creation time of analysis interval
  uint32_t mRunNumber{0};             ///< run number
  int mMaxTFPerFile{-1};              ///< maximum number of TFs per file
  std::string mOutputFileName;        ///< output file name
  bool mNewInterval{true};            ///< start a new interval
  qc::Clusters mClusterQC;            ///< cluster QC
};

void ClusterQCDevice::init(framework::InitContext& ic)
{
  mOutputFileName = ic.options().get<std::string>("output-file-name");
  mMaxTFPerFile = ic.options().get<int>("max-tf-per-file");

  mProcessEveryNthTF = ic.options().get<int>("processEveryNthTF");
  if (mProcessEveryNthTF <= 0) {
    mProcessEveryNthTF = 1;
  }

  if (mProcessEveryNthTF > 1) {
    std::mt19937 rng(std::time(nullptr));
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, mProcessEveryNthTF);
    mTFCounter = dist(rng);
    LOGP(info, "Skipping first {} TFs", mProcessEveryNthTF - mTFCounter);
  }
}

void ClusterQCDevice::run(ProcessingContext& pc)
{
  if (mTFCounter++ % mProcessEveryNthTF) {
    const auto currentTF = processing_helpers::getCurrentTF(pc);
    LOGP(info, "Skipping TF {}", currentTF);
    return;
  }

  if (mNewInterval) {
    mRunNumber = processing_helpers::getRunNumber(pc);
    mFirstTF = processing_helpers::getCurrentTF(pc);
    mFirstCreation = processing_helpers::getCreationTime(pc);
    mNewInterval = false;
  }

  mLastTF = processing_helpers::getCurrentTF(pc);
  mLastCreation = processing_helpers::getCreationTime(pc);

  const auto& clustersInputs = getWorkflowTPCInput(pc);
  const auto& clusterIndex = clustersInputs->clusterIndex;

  for (int sector = 0; sector < MAXSECTOR; ++sector) {
    for (int padrow = 0; padrow < MAXGLOBALPADROW; ++padrow) {

      for (size_t icl = 0; icl < clusterIndex.nClusters[sector][padrow]; ++icl) {
        const auto& cl = clusterIndex.clusters[sector][padrow][icl];
        mClusterQC.processCluster(cl, sector, padrow);
      }
    }
  }

  mClusterQC.endTF();
  if ((mMaxTFPerFile > 0) && (mClusterQC.getProcessedTFs() % mMaxTFPerFile) == 0) {
    endInterval();
  }
}

void ClusterQCDevice::endInterval()
{
  if (mClusterQC.getProcessedTFs() == 0) {
    return;
  }

  mClusterQC.analyse();
  LOGP(info, "End interval for run: {}, TFs: {} - {}, creation: {} - {}, processed TFs: {}",
       mRunNumber, mFirstTF, mLastTF, mFirstCreation, mLastCreation, mClusterQC.getProcessedTFs());

  const auto outputFileName = fmt::format(mOutputFileName, fmt::arg("run", mRunNumber),
                                          fmt::arg("firstTF", mFirstTF), fmt::arg("lastTF", mLastTF),
                                          fmt::arg("firstCreation", mFirstCreation), fmt::arg("lastCreation", mLastCreation));
  std::unique_ptr<TFile> f(TFile::Open(outputFileName.data(), "recreate"));
  f->WriteObject(&mClusterQC, "ClusterQC");
  f->Close();

  mClusterQC.reset();
  mRunNumber = mFirstTF = mLastTF = mFirstCreation = mLastCreation = 0;
  mNewInterval = true;
}

void ClusterQCDevice::endOfStream(EndOfStreamContext& eos)
{
  LOG(info) << "Finalizig Cluster QC filter";
  endInterval();
}

DataProcessorSpec getClusterQCSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-cluster-qc",
    inputs,
    outputs,
    adaptFromTask<ClusterQCDevice>(),
    Options{
      {"output-file-name", VariantType::String, "clusterQC_{run}_{firstCreation}_{lastCreation}_{firstTF}_{lastTF}.root", {"name of the output file"}},
      {"processEveryNthTF", VariantType::Int, 1, {"Using only a fraction of the data: 1: Use every TF, 10: Process only every tenth TF."}},
      {"max-tf-per-file", VariantType::Int, -1, {"Number of TFs to process before a file is written. -1 = all"}},
    }};
}

} // namespace o2::tpc
