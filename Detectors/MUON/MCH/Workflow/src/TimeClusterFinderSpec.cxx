// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TimeClusterFinderSpec.cxx
/// \brief Implementation of a data processor to run the time clusterizer
///
/// \author Andrea Ferrero, CEA

#include "MCHWorkflow/TimeClusterFinderSpec.h"

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
#include "MCHTimeClustering/TimeClusterFinder.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TimeClusterFinderTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the preclusterizer
    LOG(INFO) << "initializing preclusterizer";

    mTimeClusterFinder.init();

    auto stop = [this]() {
      LOG(INFO) << "reset precluster finder duration = " << mTimeResetTimeClusterFinder.count() << " ms";
      LOG(INFO) << "load digits duration = " << mTimeLoadDigits.count() << " ms";
      LOG(INFO) << "precluster finder duration = " << mTimeTimeClusterFinder.count() << " ms";
      LOG(INFO) << "store precluster duration = " << mTimeStorePreClusters.count() << " ms";
      /// Clear the preclusterizer
      auto tStart = std::chrono::high_resolution_clock::now();
      this->mTimeClusterFinder.deinit();
      auto tEnd = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "deinitializing preclusterizer in: "
                << std::chrono::duration<double, std::milli>(tEnd - tStart).count() << " ms";
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the digits, preclusterize and send the preclusters

    // prepare to receive new data
    auto tStart = std::chrono::high_resolution_clock::now();
    mTimeClusterFinder.reset();
    mPreClusters.clear();
    mUsedDigits.clear();
    auto tEnd = std::chrono::high_resolution_clock::now();
    mTimeResetTimeClusterFinder += tEnd - tStart;

    // get the input digits
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    //std::cout<<"digits size: "<<digits.size()<<std::endl;

    // load the digits to get the fired pads
    tStart = std::chrono::high_resolution_clock::now();
    mTimeClusterFinder.loadDigits(digits);
    tEnd = std::chrono::high_resolution_clock::now();
    mTimeLoadDigits += tEnd - tStart;

    gsl::span<Digit> outputDigits;
    gsl::span<PreCluster> outputPreclusters;
    mTimeClusterFinder.getTimeClusters(outputPreclusters, outputDigits);

    //std::cout<<"output preclusters size: "<<outputPreclusters.size()<<std::endl;
    //std::cout<<"output digits size: "<<outputDigits.size()<<std::endl;
    // send the output messages
    auto freefct = [](void* data, void* /*hint*/) { free(data); };
    pc.outputs().adoptChunk(Output{"MCH", "TCLUSTERS", 0, Lifetime::Timeframe},
                            reinterpret_cast<char*>(outputPreclusters.data()), outputPreclusters.size_bytes(), freefct, nullptr);
    pc.outputs().adoptChunk(Output{"MCH", "TCLUSTERDIGITS", 0, Lifetime::Timeframe},
                            reinterpret_cast<char*>(outputDigits.data()), outputDigits.size_bytes(), freefct, nullptr);
  }

 private:
  TimeClusterFinder mTimeClusterFinder{}; ///< preclusterizer
  std::vector<PreCluster> mPreClusters{}; ///< vector of preclusters
  std::vector<Digit> mUsedDigits{};       ///< vector of digits in the preclusters

  std::chrono::duration<double, std::milli> mTimeResetTimeClusterFinder{}; ///< timer
  std::chrono::duration<double, std::milli> mTimeLoadDigits{};             ///< timer
  std::chrono::duration<double, std::milli> mTimeTimeClusterFinder{};      ///< timer
  std::chrono::duration<double, std::milli> mTimeStorePreClusters{};       ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTimeClusterFinderSpec()
{
  return DataProcessorSpec{
    "TimeClusterFinder",
    Inputs{InputSpec{"digits", "MCH", "DIGITS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "TCLUSTERS", 0, Lifetime::Timeframe},
            OutputSpec{"MCH", "TCLUSTERDIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TimeClusterFinderTask>()},
    Options{}};
}

} // end namespace mch
} // end namespace o2
