// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TimePreClusterFinderSpec.cxx
/// \brief Implementation of a data processor to run the preclusterizer after the time clusterizer
///
/// \author Andrea Ferrero, CEA
/// \author Philippe Pillot, Subatech

#include "MCHWorkflow/TimePreClusterFinderSpec.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

#include <stdexcept>

#include <fmt/core.h>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MCHBase/Digit.h"
#include "MCHBase/PreCluster.h"
#include "MCHPreClustering/PreClusterFinder.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TimePreClusterFinderTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the preclusterizer
    LOG(INFO) << "initializing preclusterizer";

    mPreClusterFinder.init();

    auto stop = [this]() {
      LOG(INFO) << "reset precluster finder duration = " << mTimeResetPreClusterFinder.count() << " ms";
      LOG(INFO) << "load digits duration = " << mTimeLoadDigits.count() << " ms";
      LOG(INFO) << "precluster finder duration = " << mTimePreClusterFinder.count() << " ms";
      LOG(INFO) << "store precluster duration = " << mTimeStorePreClusters.count() << " ms";
      /// Clear the preclusterizer
      auto tStart = std::chrono::high_resolution_clock::now();
      this->mPreClusterFinder.deinit();
      auto tEnd = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "deinitializing preclusterizer in: "
                << std::chrono::duration<double, std::milli>(tEnd - tStart).count() << " ms";
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    // get the input buffers
    auto preClusters = pc.inputs().get<gsl::span<PreCluster>>("preclusters");
    auto digits = pc.inputs().get<gsl::span<Digit>>("preclusterdigits");

    //if (mPrint) {
    //  std::cout << "Number of pre-clusters: " << preClusters.size() << std::endl;
    //}
    mPreClusters.clear();
    mUsedDigits.clear();

    /// read the time clusters, preclusterize and send the preclusters
    for (auto& preCluster : preClusters) {
      // get the digits of this precluster
      auto preClusterDigits = digits.subspan(preCluster.firstDigit, preCluster.nDigits);

      // prepare to receive new data
      auto tStart = std::chrono::high_resolution_clock::now();
      mPreClusterFinder.reset();
      auto tEnd = std::chrono::high_resolution_clock::now();
      mTimeResetPreClusterFinder += tEnd - tStart;

      // load the digits to get the fired pads
      tStart = std::chrono::high_resolution_clock::now();
      mPreClusterFinder.loadDigits(preClusterDigits);
      tEnd = std::chrono::high_resolution_clock::now();
      mTimeLoadDigits += tEnd - tStart;

      // preclusterize
      tStart = std::chrono::high_resolution_clock::now();
      int nPreClusters = mPreClusterFinder.run();
      tEnd = std::chrono::high_resolution_clock::now();
      mTimePreClusterFinder += tEnd - tStart;

      // get the preclusters and associated digits
      tStart = std::chrono::high_resolution_clock::now();
      mPreClusters.reserve(mPreClusters.size() + nPreClusters); // to avoid reallocation if
      mUsedDigits.reserve(mUsedDigits.size() + digits.size());  // the capacity is exceeded
      mPreClusterFinder.getPreClusters(mPreClusters, mUsedDigits);
      //if (mUsedDigits.size() != digits.size()) {
      //std::cout<<"some digits have been lost during the preclustering"<<std::endl;
      //for (auto& d : digits) {
      //  std::cout << fmt::format("  DE {:4d}  PAD {:5d}  ADC {:6d}  TIME {}-{:4d}",
      //      d.getDetID(), d.getPadID(), d.getADC(), d.getTime().bunchCrossing, d.getTimeStamp());
      //  std::cout << std::endl;
      //}
      //throw runtime_error("some digits have been lost during the preclustering");
      //}
      tEnd = std::chrono::high_resolution_clock::now();
      mTimeStorePreClusters += tEnd - tStart;
    }

    // send the output messages
    pc.outputs().snapshot(Output{"MCH", "PRECLUSTERS", 0, Lifetime::Timeframe}, mPreClusters);
    pc.outputs().snapshot(Output{"MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}, mUsedDigits);
  }

 private:
  PreClusterFinder mPreClusterFinder{};   ///< preclusterizer
  std::vector<PreCluster> mPreClusters{}; ///< vector of preclusters
  std::vector<Digit> mUsedDigits{};       ///< vector of digits in the preclusters

  std::chrono::duration<double, std::milli> mTimeResetPreClusterFinder{}; ///< timer
  std::chrono::duration<double, std::milli> mTimeLoadDigits{};            ///< timer
  std::chrono::duration<double, std::milli> mTimePreClusterFinder{};      ///< timer
  std::chrono::duration<double, std::milli> mTimeStorePreClusters{};      ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTimePreClusterFinderSpec()
{
  return DataProcessorSpec{
    "TimePreClusterFinder",
    Inputs{InputSpec{"preclusters", "MCH", "TCLUSTERS", 0, Lifetime::Timeframe},
           InputSpec{"preclusterdigits", "MCH", "TCLUSTERDIGITS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "PRECLUSTERS", 0, Lifetime::Timeframe},
            OutputSpec{"MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TimePreClusterFinderTask>()},
    Options{}};
}

} // end namespace mch
} // end namespace o2
