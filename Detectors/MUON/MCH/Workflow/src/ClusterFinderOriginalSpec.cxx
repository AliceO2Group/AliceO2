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
#include <string>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "MCHBase/Error.h"
#include "MCHBase/ErrorMap.h"
#include "MCHBase/PreCluster.h"
#include "DataFormatsMCH/Cluster.h"
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
    LOG(info) << "initializing cluster finder";

    auto config = ic.options().get<std::string>("mch-config");
    if (!config.empty()) {
      o2::conf::ConfigurableParam::updateFromFile(config, "MCHClustering", true);
    }
    bool run2Config = ic.options().get<bool>("run2-config");
    mClusterFinder.init(run2Config);

    mAttachInitalPrecluster = ic.options().get<bool>("attach-initial-precluster");

    /// Print the timer and clear the clusterizer when the processing is over
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>([this]() {
      LOG(info) << "cluster finder duration = " << mTimeClusterFinder.count() << " s";
      mErrorMap.forEach([](Error error) {
        LOGP(warning, error.asString());
      });
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

    // create the output messages for clusters and attached digits
    auto& clusterROFs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"clusterrofs"});
    auto& clusters = pc.outputs().make<std::vector<Cluster>>(OutputRef{"clusters"});
    auto& usedDigits = pc.outputs().make<std::vector<Digit>>(OutputRef{"clusterdigits"});

    clusterROFs.reserve(preClusterROFs.size());
    auto& errorMap = mClusterFinder.getErrorMap();
    errorMap.clear();
    for (const auto& preClusterROF : preClusterROFs) {

      // prepare to clusterize the current ROF
      auto clusterOffset = clusters.size();
      mClusterFinder.reset();

      for (const auto& preCluster : preClusters.subspan(preClusterROF.getFirstIdx(), preClusterROF.getNEntries())) {

        auto preclusterDigits = digits.subspan(preCluster.firstDigit, preCluster.nDigits);
        auto firstClusterIdx = mClusterFinder.getClusters().size();

        // clusterize the current precluster
        auto tStart = std::chrono::high_resolution_clock::now();
        mClusterFinder.findClusters(preclusterDigits);
        auto tEnd = std::chrono::high_resolution_clock::now();
        mTimeClusterFinder += tEnd - tStart;

        if (mAttachInitalPrecluster) {
          // store the new clusters and associate them to all the digits of the precluster
          writeClusters(preclusterDigits, firstClusterIdx, clusters, usedDigits);
        }
      }

      if (!mAttachInitalPrecluster) {
        // store all the clusters of the current ROF and the associated digits actually used in the clustering
        writeClusters(clusters, usedDigits);
      }

      // create the cluster ROF
      clusterROFs.emplace_back(preClusterROF.getBCData(), clusterOffset, clusters.size() - clusterOffset,
                               preClusterROF.getBCWidth());
    }

    // create the output message for clustering errors
    auto& clusterErrors = pc.outputs().make<std::vector<Error>>(OutputRef{"clustererrors"});
    errorMap.forEach([&clusterErrors](Error error) {
      clusterErrors.emplace_back(error);
    });
    mErrorMap.add(errorMap);

    LOGP(info, "Found {:4d} clusters from {:4d} preclusters in {:2d} ROFs",
         clusters.size(), preClusters.size(), preClusterROFs.size());
  }

 private:
  //_________________________________________________________________________________________________
  void writeClusters(const gsl::span<const Digit>& preclusterDigits, size_t firstClusterIdx,
                     std::vector<Cluster, o2::pmr::polymorphic_allocator<Cluster>>& clusters,
                     std::vector<Digit, o2::pmr::polymorphic_allocator<Digit>>& usedDigits) const
  {
    /// fill the output messages with the new clusters and all the digits from the corresponding precluster
    /// modify the references to the attached digits according to their position in the global vector

    if (firstClusterIdx == mClusterFinder.getClusters().size()) {
      return;
    }

    auto clusterOffset = clusters.size();
    clusters.insert(clusters.end(), mClusterFinder.getClusters().begin() + firstClusterIdx, mClusterFinder.getClusters().end());

    auto digitOffset = usedDigits.size();
    usedDigits.insert(usedDigits.end(), preclusterDigits.begin(), preclusterDigits.end());

    for (auto itCluster = clusters.begin() + clusterOffset; itCluster < clusters.end(); ++itCluster) {
      itCluster->firstDigit = digitOffset;
      itCluster->nDigits = preclusterDigits.size();
    }
  }

  //_________________________________________________________________________________________________
  void writeClusters(std::vector<Cluster, o2::pmr::polymorphic_allocator<Cluster>>& clusters,
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

  bool mAttachInitalPrecluster = false;               ///< attach all digits of initial precluster to cluster
  ClusterFinderOriginal mClusterFinder{};             ///< clusterizer
  ErrorMap mErrorMap{};                               ///< counting of encountered errors
  std::chrono::duration<double> mTimeClusterFinder{}; ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getClusterFinderOriginalSpec(const char* specName)
{
  return DataProcessorSpec{
    specName,
    Inputs{InputSpec{"preclusterrofs", "MCH", "PRECLUSTERROFS", 0, Lifetime::Timeframe},
           InputSpec{"preclusters", "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe},
           InputSpec{"digits", "MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{{"clusterrofs"}, "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe},
            OutputSpec{{"clusters"}, "MCH", "CLUSTERS", 0, Lifetime::Timeframe},
            OutputSpec{{"clusterdigits"}, "MCH", "CLUSTERDIGITS", 0, Lifetime::Timeframe},
            OutputSpec{{"clustererrors"}, "MCH", "CLUSTERERRORS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<ClusterFinderOriginalTask>()},
    Options{{"mch-config", VariantType::String, "", {"JSON or INI file with clustering parameters"}},
            {"run2-config", VariantType::Bool, false, {"setup for run2 data"}},
            {"attach-initial-precluster", VariantType::Bool, false, {"attach all digits of initial precluster to cluster"}}}};
}

} // end namespace mch
} // end namespace o2
