// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterFinderGEMSpec.cxx
/// \brief Implementation of a data processor to run the GEM MLEM cluster finder
///
/// \author Philippe Pillot, Subatech

#include "MCHWorkflow/ClusterFinderGEMSpec.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <stdexcept>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "MCHBase/PreCluster.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHClustering/ClusterFinderGEM.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class ClusterFinderGEMTask
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
    /// read the preclusters and associated digits, clusterize and send the clusters for all events in the TF

    // get the input preclusters and associated digits
    auto preClusterROFs = pc.inputs().get<gsl::span<ROFRecord>>("preclusterrofs");
    auto preClusters = pc.inputs().get<gsl::span<PreCluster>>("preclusters");
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    //LOG(INFO) << "received time frame with " << preClusterROFs.size() << " interactions";

    // create the output messages for clusters and attached digits
    auto& clusterROFs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"clusterrofs"});
    auto& clusters = pc.outputs().make<std::vector<ClusterStruct>>(OutputRef{"clusters"});
    auto& usedDigits = pc.outputs().make<std::vector<Digit>>(OutputRef{"clusterdigits"});

    clusterROFs.reserve(preClusterROFs.size());
    for (const auto& preClusterROF : preClusterROFs) {

      LOG(INFO) << "processing interaction: ev ??? " << preClusterROF.getBCData().orbit << "...";
      // GG infos
      // uint16_t bc = DummyBC;       ///< bunch crossing ID of interaction
      // uint32_t orbit = DummyOrbit; ///< LHC orbit
      // clusterize every preclusters
      uint16_t bCrossing = preClusterROF.getBCData().bc;
      uint32_t orbit= preClusterROF.getBCData().orbit;
      uint32_t iROF=0;
      auto tStart = std::chrono::high_resolution_clock::now();
      //
      mClusterFinder.reset();
      for (const auto& preCluster : preClusters.subspan(preClusterROF.getFirstIdx(), preClusterROF.getNEntries())) {
        mClusterFinder.findClusters(digits.subspan(preCluster.firstDigit, preCluster.nDigits), bCrossing, orbit, iROF);
        iROF++;
      }
      auto tEnd = std::chrono::high_resolution_clock::now();
      mTimeClusterFinder += tEnd - tStart;

      // fill the ouput messages
      clusterROFs.emplace_back(preClusterROF.getBCData(), clusters.size(), mClusterFinder.getClusters().size());
      writeClusters(clusters, usedDigits);
    }
  }

 private:
  //_________________________________________________________________________________________________
  void writeClusters(std::vector<ClusterStruct, o2::pmr::polymorphic_allocator<ClusterStruct>>& clusters,
                     std::vector<Digit, o2::pmr::polymorphic_allocator<Digit>>& usedDigits) const
  {
    /// fill the output messages with clusters and attached digits of the current event
    /// modify the references to the attached digits according to their position in the global vector

    auto clusterOffset = clusters.size();
    clusters.insert(clusters.end(), mClusterFinder.getClusters().begin(), mClusterFinder.getClusters().end());

    auto digitOffset = usedDigits.size();
    usedDigits.insert(usedDigits.end(), mClusterFinder.getUsedDigits().begin(), mClusterFinder.getUsedDigits().end());

    for (auto itCluster = clusters.begin() + clusterOffset; itCluster < clusters.end(); ++itCluster) {
      itCluster->firstDigit += digitOffset;
    }
  }

  ClusterFinderGEM mClusterFinder{};             ///< clusterizer
  std::chrono::duration<double> mTimeClusterFinder{}; ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getClusterFinderGEMSpec()
{
  return DataProcessorSpec{
    "ClusterFinderGEM",
    Inputs{InputSpec{"preclusterrofs", "MCH", "PRECLUSTERROFS", 0, Lifetime::Timeframe},
           InputSpec{"preclusters", "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe},
           InputSpec{"digits", "MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{{"clusterrofs"}, "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe},
            OutputSpec{{"clusters"}, "MCH", "CLUSTERS", 0, Lifetime::Timeframe},
            OutputSpec{{"clusterdigits"}, "MCH", "CLUSTERDIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<ClusterFinderGEMTask>()},
    Options{}};
}

} // end namespace mch
} // end namespace o2
