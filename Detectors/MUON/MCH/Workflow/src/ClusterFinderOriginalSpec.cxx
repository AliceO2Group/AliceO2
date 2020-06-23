// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterFinderOriginalSpec.cxx
/// \brief Implementation of a data processor to run the original MLEM cluster finder
///
/// \author Philippe Pillot, Subatech

#include "MCHWorkflow/ClusterFinderOriginalSpec.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MCHBase/Digit.h"
#include "MCHBase/PreCluster.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHClustering/ClusterFinderOriginal.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class ClusterFinderOriginalTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the clusterizer
    LOG(INFO) << "initializing cluster finder";

    mClusterFinder.init();

    /// Print the timer and clear the clusterizer when the processing is over
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, [this]() {
      LOG(INFO) << "cluster finder duration = " << mTimeClusterFinder.count() << " s";
      this->mClusterFinder.deinit();
    });
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the preclusters and associated digits, clusterize and send the clusters

    // get the input preclusters and associated digits
    auto preClusters = pc.inputs().get<gsl::span<PreCluster>>("preclusters");
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    // clusterize every preclusters
    auto tStart = std::chrono::high_resolution_clock::now();
    mClusterFinder.reset();
    for (const auto& preCluster : preClusters) {
      mClusterFinder.findClusters(digits.subspan(preCluster.firstDigit, preCluster.nDigits));
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    mTimeClusterFinder += tEnd - tStart;

    // send the output messages
    pc.outputs().snapshot(Output{"MCH", "CLUSTERS", 0, Lifetime::Timeframe}, mClusterFinder.getClusters());
    pc.outputs().snapshot(Output{"MCH", "CLUSTERDIGITS", 0, Lifetime::Timeframe}, mClusterFinder.getUsedDigits());
  }

 private:
  ClusterFinderOriginal mClusterFinder{};             ///< clusterizer
  std::chrono::duration<double> mTimeClusterFinder{}; ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getClusterFinderOriginalSpec()
{
  return DataProcessorSpec{
    "ClusterFinderOriginal",
    Inputs{InputSpec{"preclusters", "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe},
           InputSpec{"digits", "MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "CLUSTERS", 0, Lifetime::Timeframe},
            OutputSpec{"MCH", "CLUSTERDIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<ClusterFinderOriginalTask>()},
    Options{}};
}

} // end namespace mch
} // end namespace o2
