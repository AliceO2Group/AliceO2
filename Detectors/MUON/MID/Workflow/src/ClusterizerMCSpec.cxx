// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/ClusterizerMCSpec.cxx
/// \brief  Data processor spec for MID MC clustering device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   27 September 2019

#include "MIDWorkflow/ClusterizerMCSpec.h"

#include <array>
#include <vector>
#include <gsl/gsl>
#include "Framework/DataRefUtils.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MIDClustering/PreClusterizer.h"
#include "MIDClustering/Clusterizer.h"
#include "MIDSimulation/ClusterLabeler.h"
#include "MIDSimulation/MCLabel.h"
#include "MIDSimulation/PreClusterLabeler.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class ClusterizerMCDeviceDPL
{
 public:
  ClusterizerMCDeviceDPL(const char* inputBinding, const char* inputROFBinding, const char* inputLabelsBinding) : mInputBinding(inputBinding), mInputROFBinding(inputROFBinding), mInputLabelsBinding(inputLabelsBinding), mPreClusterizer(), mClusterizer(), mCorrelation(){};
  ~ClusterizerMCDeviceDPL() = default;

  void init(o2::framework::InitContext& ic)
  {
    if (!mPreClusterizer.init()) {
      LOG(ERROR) << "Initialization of MID pre-clusterizer device failed";
    }

    mCorrelation.clear();

    bool isClusterizerInit = mClusterizer.init([&](size_t baseIndex, size_t relatedIndex) { mCorrelation.push_back({baseIndex, relatedIndex}); });

    if (!isClusterizerInit) {
      LOG(ERROR) << "Initialization of MID clusterizer device failed";
    }
  }
  void run(o2::framework::ProcessingContext& pc)
  {
    auto msg = pc.inputs().get(mInputBinding.c_str());
    gsl::span<const ColumnData> patterns = of::DataRefUtils::as<const ColumnData>(msg);

    auto msgROF = pc.inputs().get(mInputROFBinding.c_str());
    gsl::span<const ROFRecord> inROFRecords = of::DataRefUtils::as<const ROFRecord>(msgROF);

    std::unique_ptr<const o2::dataformats::MCTruthContainer<MCLabel>> labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<MCLabel>*>(mInputLabelsBinding.c_str());

    // Pre-clustering
    mPreClusterizer.process(patterns, inROFRecords);
    LOG(DEBUG) << "Generated " << mPreClusterizer.getPreClusters().size() << " PreClusters";

    // Clustering
    mClusterizer.process(mPreClusterizer.getPreClusters(), mPreClusterizer.getROFRecords());

    // Labelling
    mPreClusterLabeler.process(mPreClusterizer.getPreClusters(), *labels, mPreClusterizer.getROFRecords(), inROFRecords);
    mClusterLabeler.process(mPreClusterizer.getPreClusters(), mPreClusterLabeler.getContainer(), mClusterizer.getClusters(), mCorrelation);
    // Clear the index correlations that will be used in the next cluster processing
    mCorrelation.clear();

    pc.outputs().snapshot(of::Output{"MID", "CLUSTERS", 0, of::Lifetime::Timeframe}, mClusterizer.getClusters());
    LOG(DEBUG) << "Sent " << mClusterizer.getClusters().size() << " clusters";
    pc.outputs().snapshot(of::Output{"MID", "CLUSTERSROF", 0, of::Lifetime::Timeframe}, mClusterizer.getROFRecords());
    LOG(DEBUG) << "Sent " << mClusterizer.getROFRecords().size() << " ROF";

    pc.outputs().snapshot(of::Output{"MID", "CLUSTERSLABELS", 0, of::Lifetime::Timeframe}, mClusterLabeler.getContainer());
    LOG(DEBUG) << "Sent " << mClusterLabeler.getContainer().getIndexedSize() << " indexed clusters";
  }

 private:
  std::string mInputBinding;
  std::string mInputROFBinding;
  std::string mInputLabelsBinding;
  PreClusterizer mPreClusterizer;
  Clusterizer mClusterizer;
  PreClusterLabeler mPreClusterLabeler;
  ClusterLabeler mClusterLabeler;
  std::vector<std::array<size_t, 2>> mCorrelation;
};

framework::DataProcessorSpec getClusterizerMCSpec()
{
  std::string inputBinding = "mid_data";
  std::string inputROFBinding = "mid_data_rof";
  std::string inputLabelsBinding = "mid_data_labels";
  std::vector<of::InputSpec> inputSpecs{
    of::InputSpec{inputBinding, "MID", "DATA"},
    of::InputSpec{inputROFBinding, "MID", "DATAROF"},
    of::InputSpec{inputLabelsBinding, "MID", "DATALABELS"}};
  std::vector<of::OutputSpec> outputSpecs{of::OutputSpec{"MID", "CLUSTERS"}, of::OutputSpec{"MID", "CLUSTERSROF"}, of::OutputSpec{"MID", "CLUSTERSLABELS"}};

  return of::DataProcessorSpec{
    "MIDClusterizerMC",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::ClusterizerMCDeviceDPL>(inputBinding.c_str(), inputROFBinding.c_str(), inputLabelsBinding.c_str())}};
}
} // namespace mid
} // namespace o2