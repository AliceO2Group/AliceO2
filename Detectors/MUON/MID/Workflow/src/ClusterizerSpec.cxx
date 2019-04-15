// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <gsl/gsl>
#include "Framework/DataRefUtils.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/Cluster2D.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MIDClustering/PreCluster.h"
#include "MIDClustering/PreClusterizer.h"
#include "MIDClustering/Clusterizer.h"

#include <fairlogger/Logger.h>

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class ClusterizerDeviceDPL
{
 public:
  ClusterizerDeviceDPL(const char* inputBinding, bool isMC) : mInputBinding(inputBinding), mIsMC(isMC), mPreClusterizer(), mClusterizer(){};
  ~ClusterizerDeviceDPL() = default;

  void init(o2::framework::InitContext& ic)
  {
    if (!mPreClusterizer.init()) {
      LOG(ERROR) << "Initialization of MID pre-clusterizer device failed";
    }

    bool isClusterizerInit = false;
    if (mIsMC) {
      isClusterizerInit = mClusterizer.init([&](size_t baseIndex, size_t relatedIndex) { mCorrelation.push_back({ baseIndex, relatedIndex }); });
      mCorrelation.clear();
    } else {
      isClusterizerInit = mClusterizer.init();
    }
    if (!isClusterizerInit) {
      LOG(ERROR) << "Initialization of MID clusterizer device failed";
    }
  }

  void
    run(o2::framework::ProcessingContext& pc)
  {
    auto msg = pc.inputs().get(mInputBinding.c_str());
    gsl::span<const ColumnData> patterns = of::DataRefUtils::as<const ColumnData>(msg);

    // Pre-clustering
    mPreClusterizer.process(patterns);
    LOG(INFO) << "Generated " << mPreClusterizer.getPreClusters().size() << " PreClusters";

    // Clustering
    gsl::span<const PreCluster> preClusters(mPreClusterizer.getPreClusters().data(), mPreClusterizer.getPreClusters().size());
    mClusterizer.process(preClusters);

    pc.outputs().snapshot(of::Output{ "MID", "CLUSTERS", 0, of::Lifetime::Timeframe }, mClusterizer.getClusters());
    LOG(INFO) << "Sent " << mClusterizer.getClusters().size() << " clusters";

    if (mIsMC) {
      pc.outputs().snapshot(of::Output{ "MID", "PRECLUSTERS", 0, of::Lifetime::Timeframe }, mPreClusterizer.getPreClusters());
      LOG(INFO) << "Sent " << mPreClusterizer.getPreClusters().size() << " pre-clusters";
      // Clear the index correlations that will be used in the next cluster processing
      pc.outputs().snapshot(of::Output{ "MID", "CLUSTERSCORR", 0, of::Lifetime::Timeframe }, mCorrelation);
      LOG(INFO) << "Sent " << mCorrelation.size() << " correlations";
    }
    mCorrelation.clear();
  }

 private:
  std::string mInputBinding;
  bool mIsMC = false;
  PreClusterizer mPreClusterizer;
  Clusterizer mClusterizer;
  std::vector<std::array<size_t, 2>> mCorrelation;
};

framework::DataProcessorSpec getClusterizerSpec(bool isMC)
{
  std::string inputBinding = "mid_data";
  std::vector<of::InputSpec> inputSpecs{ of::InputSpec{ inputBinding, "MID", "DATA" } };
  std::vector<of::OutputSpec> outputSpecs{ of::OutputSpec{ "MID", "CLUSTERS" } };
  if (isMC) {
    outputSpecs.emplace_back(of::OutputSpec{ "MID", "PRECLUSTERS" });
    outputSpecs.emplace_back(of::OutputSpec{ "MID", "CLUSTERSCORR" });
  }

  return of::DataProcessorSpec{
    "MIDClusterizer",
    { inputSpecs },
    { outputSpecs },
    of::AlgorithmSpec{ of::adaptFromTask<o2::mid::ClusterizerDeviceDPL>(inputBinding.c_str(), isMC) }
  };
}
} // namespace mid
} // namespace o2