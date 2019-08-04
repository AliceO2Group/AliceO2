// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/TrackerSpec.cxx
/// \brief  Data processor spec for MID tracker device
/// \author Gabriele G. Fronze <gfronze at cern.ch>
/// \date   9 July 2018

#include "MIDWorkflow/TrackerSpec.h"

#include "Framework/DataRefUtils.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/Cluster3D.h"
#include "DataFormatsMID/Track.h"
#include "MIDSimulation/Geometry.h"
#include "MIDTracking/Tracker.h"
#include "DetectorsBase/GeometryManager.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{
class TrackerDeviceDPL
{
 public:
  explicit TrackerDeviceDPL(const char* inputBinding) : mInputBinding(inputBinding), mTracker(nullptr){};
  ~TrackerDeviceDPL() = default;

  void init(o2::framework::InitContext& ic)
  {

    if (!gGeoManager) {
      o2::base::GeometryManager::loadGeometry();
    }

    mTracker = std::make_unique<Tracker>(createTransformationFromManager(gGeoManager));

    if (!mTracker->init()) {
      LOG(ERROR) << "Initialization of MID tracker device failed";
    }
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    auto msg = pc.inputs().get(mInputBinding.c_str());
    gsl::span<const Cluster2D> clusters = of::DataRefUtils::as<const Cluster2D>(msg);

    mTracker->process(clusters);

    pc.outputs().snapshot(of::Output{"MID", "TRACKS", 0, of::Lifetime::Timeframe}, mTracker->getTracks());
    LOG(INFO) << "Sent " << mTracker->getTracks().size() << " tracks.";

    pc.outputs().snapshot(of::Output{"MID", "TRACKCLUSTERS", 0, of::Lifetime::Timeframe}, mTracker->getClusters());
    LOG(INFO) << "Sent " << mTracker->getClusters().size() << " track clusters.";
  }

 private:
  std::string mInputBinding;
  std::unique_ptr<Tracker> mTracker = nullptr;
};

framework::DataProcessorSpec getTrackerSpec(bool useMC)
{
  std::string inputBinding = "mid_clusters";

  std::vector<of::InputSpec> inputSpecs;
  if (useMC) {
    inputSpecs.emplace_back(of::InputSpec{inputBinding, "MID", "CLUSTERS_DATA"});
  } else {
    inputSpecs.emplace_back(of::InputSpec{inputBinding, "MID", "CLUSTERS"});
  }

  std::vector<of::OutputSpec> outputSpecs{of::OutputSpec{"MID", "TRACKS"}, of::OutputSpec{"MID", "TRACKCLUSTERS"}};

  return of::DataProcessorSpec{
    "Tracker",
    {inputSpecs},
    {outputSpecs},
    of::adaptFromTask<o2::mid::TrackerDeviceDPL>(inputBinding.c_str())};
}
} // namespace mid
} // namespace o2