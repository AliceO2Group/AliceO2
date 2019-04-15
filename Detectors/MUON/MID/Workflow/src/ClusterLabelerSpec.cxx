// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/ClusterLabelerSpec.cxx
/// \brief  Data processor spec for MID cluster labeler device
/// \author Diego Stocco <diego.stocco at cern.ch>
/// \date   18 June 2019

#include "MIDWorkflow/ClusterLabelerSpec.h"

#include <array>
#include <vector>
#include <gsl/gsl>
#include "Framework/DataRefUtils.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDClustering/PreCluster.h"
#include "MIDSimulation/ClusterLabeler.h"
#include "MIDSimulation/MCLabel.h"
#include "MIDSimulation/PreClusterLabeler.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class ClusterLabelerDeviceDPL
{
 public:
  ClusterLabelerDeviceDPL(const char* inputPreClustersBinding, const char* inputClustersBinding, const char* inputCorrelationBinding, const char* inputLabelsBinding) : mInputPreClustersBinding(inputPreClustersBinding), mInputClustersBinding(inputClustersBinding), mInputCorrelationBinding(inputCorrelationBinding), mInputLabelsBinding(inputLabelsBinding), mPreClusterLabeler(), mClusterLabeler(){};
  ~ClusterLabelerDeviceDPL() = default;

  void init(o2::framework::InitContext& ic)
  {
  }

  void
    run(o2::framework::ProcessingContext& pc)
  {
    auto msgPreClusters = pc.inputs().get(mInputPreClustersBinding.c_str());
    gsl::span<const PreCluster> preClusters = of::DataRefUtils::as<const PreCluster>(msgPreClusters);

    auto msgClusters = pc.inputs().get(mInputClustersBinding.c_str());
    gsl::span<const Cluster2D> clusters = of::DataRefUtils::as<const Cluster2D>(msgClusters);

    auto msgCorrelation = pc.inputs().get(mInputCorrelationBinding.c_str());
    gsl::span<const std::array<size_t, 2>> correlation = of::DataRefUtils::as<const std::array<size_t, 2>>(msgCorrelation);

    std::unique_ptr<const o2::dataformats::MCTruthContainer<MCLabel>> labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<MCLabel>*>(mInputLabelsBinding.c_str());

    mPreClusterLabeler.process(preClusters, *labels);

    mClusterLabeler.process(preClusters, mPreClusterLabeler.getContainer(), clusters, correlation);

    pc.outputs().snapshot(of::Output{ "MID", "CLUSTERSLABELS", 0, of::Lifetime::Timeframe }, mClusterLabeler.getContainer());
    LOG(INFO) << "Sent " << mClusterLabeler.getContainer().getIndexedSize() << " indexed clusters";
  }

 private:
  std::string mInputPreClustersBinding;
  std::string mInputClustersBinding;
  std::string mInputCorrelationBinding;
  std::string mInputLabelsBinding;
  PreClusterLabeler mPreClusterLabeler;
  ClusterLabeler mClusterLabeler;
};

framework::DataProcessorSpec getClusterLabelerSpec()
{
  std::string inputPreClustersBinding = "mid_preClusters";
  std::string inputClustersBinding = "mid_clusters_data";
  std::string inputCorrelationBinding = "mid_clusters_correlation";
  std::string inputLabelsBinding = "mid_data_labels";
  std::vector<of::InputSpec> inputSpecs{
    of::InputSpec{ inputPreClustersBinding, "MID", "PRECLUSTERS" },
    of::InputSpec{ inputClustersBinding, "MID", "CLUSTERS_MC" },
    of::InputSpec{ inputCorrelationBinding, "MID", "CLUSTERSCORR" },
    of::InputSpec{ inputLabelsBinding, "MID", "DATALABELS" },
  };
  std::vector<of::OutputSpec> outputSpecs{ of::OutputSpec{ "MID", "CLUSTERSLABELS" } };

  return of::DataProcessorSpec{
    "MIDClusterLabeler",
    { inputSpecs },
    { outputSpecs },
    of::AlgorithmSpec{ of::adaptFromTask<o2::mid::ClusterLabelerDeviceDPL>(inputPreClustersBinding.c_str(), inputClustersBinding.c_str(), inputCorrelationBinding.c_str(), inputLabelsBinding.c_str()) }
  };
}
} // namespace mid
} // namespace o2