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
#include <chrono>
#include <gsl/gsl>
#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/Cluster2D.h"
#include "MIDClustering/PreCluster.h"
#include "MIDClustering/PreClusterizer.h"
#include "MIDClustering/Clusterizer.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class ClusterizerDeviceDPL
{
 public:
  ClusterizerDeviceDPL(const char* inputBinding, const char* inputROFBinding) : mInputBinding(inputBinding), mInputROFBinding(inputROFBinding), mPreClusterizer(), mClusterizer(), mTimer(0), mTimerPreCluster(0), mTimerCluster(0){};
  ~ClusterizerDeviceDPL() = default;

  void init(o2::framework::InitContext& ic)
  {
    if (!mPreClusterizer.init()) {
      LOG(ERROR) << "Initialization of MID pre-clusterizer device failed";
    }

    if (!mClusterizer.init()) {
      LOG(ERROR) << "Initialization of MID clusterizer device failed";
    }

    auto stop = [this]() {
      LOG(INFO) << "Capacities: ROFRecords: " << mClusterizer.getROFRecords().capacity() << "  preclusters: " << mPreClusterizer.getPreClusters().capacity() << "  clusters: " << mClusterizer.getClusters().capacity();
      LOG(INFO) << "Processing times: full: " << mTimer.count() << " s  pre-clustering: " << mTimerPreCluster.count() << " s  clustering: " << mTimerCluster.count() << " s";
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();

    auto msg = pc.inputs().get(mInputBinding.c_str());
    gsl::span<const ColumnData> patterns = of::DataRefUtils::as<const ColumnData>(msg);

    auto msgROF = pc.inputs().get(mInputROFBinding.c_str());
    gsl::span<const ROFRecord> inROFRecords = of::DataRefUtils::as<const ROFRecord>(msgROF);

    // Pre-clustering
    auto tAlgoStart = std::chrono::high_resolution_clock::now();
    mPreClusterizer.process(patterns, inROFRecords);
    mTimerPreCluster += std::chrono::high_resolution_clock::now() - tAlgoStart;
    LOG(DEBUG) << "Generated " << mPreClusterizer.getPreClusters().size() << " PreClusters";

    // Clustering
    tAlgoStart = std::chrono::high_resolution_clock::now();
    mClusterizer.process(mPreClusterizer.getPreClusters(), mPreClusterizer.getROFRecords());
    mTimerCluster += std::chrono::high_resolution_clock::now() - tAlgoStart;

    pc.outputs().snapshot(of::Output{"MID", "CLUSTERS", 0, of::Lifetime::Timeframe}, mClusterizer.getClusters());
    LOG(DEBUG) << "Sent " << mClusterizer.getClusters().size() << " clusters";
    pc.outputs().snapshot(of::Output{"MID", "CLUSTERSROF", 0, of::Lifetime::Timeframe}, mClusterizer.getROFRecords());
    LOG(DEBUG) << "Sent " << mClusterizer.getROFRecords().size() << " ROF";

    mTimer += std::chrono::high_resolution_clock::now() - tStart;

    pc.services().get<of::ControlService>().readyToQuit(of::QuitRequest::Me);
  }

 private:
  std::string mInputBinding;
  std::string mInputROFBinding;
  PreClusterizer mPreClusterizer;
  Clusterizer mClusterizer;
  std::chrono::duration<double> mTimer{0};           ///< full timer
  std::chrono::duration<double> mTimerPreCluster{0}; ///< pre-clustering timer
  std::chrono::duration<double> mTimerCluster{0};    ///< clustering timer
};

framework::DataProcessorSpec getClusterizerSpec()
{
  std::string inputBinding = "mid_data";
  std::string inputROFBinding = "mid_data_rof";
  std::vector<of::InputSpec> inputSpecs{of::InputSpec{inputBinding, "MID", "DATA"}, of::InputSpec{inputROFBinding, "MID", "DATAROF"}};
  std::vector<of::OutputSpec> outputSpecs{of::OutputSpec{"MID", "CLUSTERS"}, of::OutputSpec{"MID", "CLUSTERSROF"}};

  return of::DataProcessorSpec{
    "MIDClusterizer",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::ClusterizerDeviceDPL>(inputBinding.c_str(), inputROFBinding.c_str())}};
}
} // namespace mid
} // namespace o2