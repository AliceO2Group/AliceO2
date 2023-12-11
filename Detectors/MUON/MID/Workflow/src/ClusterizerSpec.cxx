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

/// \file   MID/Workflow/src/ClusterizerSpec.cxx
/// \brief  Data processor spec for MID clustering device
/// \author Gabriele G. Fronze <gfronze at cern.ch>
/// \date   9 July 2018

#include "MIDWorkflow/ClusterizerSpec.h"

#include <array>
#include <vector>
#include <chrono>
#include <gsl/gsl>
#include "fmt/format.h"
#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsMID/Cluster.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MIDClustering/PreCluster.h"
#include "MIDClustering/PreClusterizer.h"
#include "MIDClustering/Clusterizer.h"
#include "MIDSimulation/ClusterLabeler.h"
#include "DataFormatsMID/MCLabel.h"
#include "MIDSimulation/PreClusterLabeler.h"
#include "MIDWorkflow/ColumnDataSpecsUtils.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class ClusterizerDeviceDPL
{
 public:
  ClusterizerDeviceDPL(bool isMC) : mIsMC(isMC) {}
  ~ClusterizerDeviceDPL() = default;

  void init(o2::framework::InitContext& ic)
  {
    bool isClusterizerInit = false;
    if (mIsMC) {
      mCorrelation.clear();
      isClusterizerInit = mClusterizer.init([&](size_t baseIndex, size_t relatedIndex) { mCorrelation.push_back({baseIndex, relatedIndex}); });
    } else {
      isClusterizerInit = mClusterizer.init();
    }
    if (!isClusterizerInit) {
      LOG(error) << "Initialization of MID clusterizer device failed";
    }

    auto stop = [this]() {
      double scaleFactor = (mNROFs == 0) ? 0. : 1.e6 / mNROFs;
      LOG(info) << "Processing time / " << mNROFs << " ROFs: full: " << mTimer.count() * scaleFactor << " us  pre-clustering: " << mTimerPreCluster.count() * scaleFactor << " us  clustering: " << mTimerCluster.count() * scaleFactor << " us";
    };
    ic.services().get<of::CallbackService>().set<of::CallbackService::Id::Stop>(stop);
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();

    gsl::span<const ColumnData> patterns = specs::getData(pc, "mid_cluster_in", EventType::Standard);
    gsl::span<const ROFRecord> inROFRecords = specs::getRofs(pc, "mid_cluster_in", EventType::Standard);

    // Pre-clustering
    auto tAlgoStart = std::chrono::high_resolution_clock::now();
    mPreClusterizer.process(patterns, inROFRecords);
    mTimerPreCluster += std::chrono::high_resolution_clock::now() - tAlgoStart;
    LOG(debug) << "Generated " << mPreClusterizer.getPreClusters().size() << " PreClusters";

    // Clustering
    tAlgoStart = std::chrono::high_resolution_clock::now();
    mClusterizer.process(mPreClusterizer.getPreClusters(), mPreClusterizer.getROFRecords());
    mTimerCluster += std::chrono::high_resolution_clock::now() - tAlgoStart;

    if (mIsMC) {
      // Labelling
      auto labels = specs::getLabels(pc, "mid_cluster_in");
      mPreClusterLabeler.process(mPreClusterizer.getPreClusters(), *labels, mPreClusterizer.getROFRecords(), inROFRecords);
      mClusterLabeler.process(mPreClusterizer.getPreClusters(), mPreClusterLabeler.getContainer(), mClusterizer.getClusters(), mCorrelation);
      // Clear the index correlations that will be used in the next cluster processing
      mCorrelation.clear();

      pc.outputs().snapshot(of::Output{"MID", "CLUSTERSLABELS", 0}, mClusterLabeler.getContainer());
      LOG(debug) << "Sent " << mClusterLabeler.getContainer().getIndexedSize() << " indexed clusters";
    }

    pc.outputs().snapshot(of::Output{"MID", "CLUSTERS", 0}, mClusterizer.getClusters());
    LOG(debug) << "Sent " << mClusterizer.getClusters().size() << " clusters";
    pc.outputs().snapshot(of::Output{"MID", "CLUSTERSROF", 0}, mClusterizer.getROFRecords());
    LOG(debug) << "Sent " << mClusterizer.getROFRecords().size() << " ROF";

    mTimer += std::chrono::high_resolution_clock::now() - tStart;
    mNROFs += inROFRecords.size();
  }

 private:
  bool mIsMC = false;
  PreClusterizer mPreClusterizer{};
  Clusterizer mClusterizer{};
  PreClusterLabeler mPreClusterLabeler{};
  ClusterLabeler mClusterLabeler{};
  std::vector<std::array<size_t, 2>> mCorrelation{};
  std::chrono::duration<double> mTimer{0};           ///< full timer
  std::chrono::duration<double> mTimerPreCluster{0}; ///< pre-clustering timer
  std::chrono::duration<double> mTimerCluster{0};    ///< clustering timer
  unsigned long mNROFs{0};                           ///< Total number of processed ROFs
};

framework::DataProcessorSpec getClusterizerSpec(bool isMC, std::string_view inDataDesc, std::string_view inRofDesc, std::string_view inLabelsDesc)
{
  std::vector<of::OutputSpec> outputSpecs{of::OutputSpec{header::gDataOriginMID, "CLUSTERS"}, of::OutputSpec{header::gDataOriginMID, "CLUSTERSROF"}};

  if (isMC) {
    outputSpecs.emplace_back(of::OutputSpec{header::gDataOriginMID, "CLUSTERSLABELS"});
  }
  auto inputSpecs = specs::buildInputSpecs("mid_cluster_in", inDataDesc, inRofDesc, inLabelsDesc, isMC);

  return of::DataProcessorSpec{
    "MIDClusterizer",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::ClusterizerDeviceDPL>(isMC)}};
}
} // namespace mid
} // namespace o2
