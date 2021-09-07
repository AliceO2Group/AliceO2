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
#include "Framework/Logger.h"

#include "MCHRawDecoder/OrbitInfo.h"
#include "MCHTimeClustering/ROFTimeClusterFinder.h"

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
    mTimeClusterWidth = ic.options().get<int>("max-cluster-width");
    mNbinsInOneWindow = ic.options().get<int>("peak-search-nbins");
    // number of bins must be >= 3
    if (mNbinsInOneWindow < 3) {
      mNbinsInOneWindow = 3;
    }
    // number of bins must be odd
    if ((mNbinsInOneWindow % 2) == 0) {
      mNbinsInOneWindow += 1;
    }

    auto stop = [this]() {
      LOGP(info, "\nfinder duration = {} us / TF\n", mTimeProcess.count() * 1000 / mTFcount);
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getByPos(0).header);
    auto firstTForbit = dh->firstTForbit;
    // get the input rofs
    auto rofs = pc.inputs().get<gsl::span<o2::mch::ROFRecord>>("rofs");

    /*auto orbits = pc.inputs().get<gsl::span<uint64_t>>("orbits");
    std::set<uint32_t> ordered_orbits;
    for (auto orbitInfo : orbits) {
      ordered_orbits.insert(orbitInfo & 0xFFFFFFFF);
    }

    firstTForbit = *(ordered_orbits.begin());
    if (mDebug) {
      std::cout << "First TF orbit: " << firstTForbit << std::endl;
    }*/

    o2::mch::ROFTimeClusterFinder rofProcessor(rofs, mTimeClusterWidth, mNbinsInOneWindow, 0);

    if (mDebug) {
      std::cout << "\n\n=====================\nInput ROFs:\n";
      rofProcessor.dumpInputROFs();
    }

    auto tStart = std::chrono::high_resolution_clock::now();
    rofProcessor.process();
    auto tEnd = std::chrono::high_resolution_clock::now();
    mTimeProcess += tEnd - tStart;

    if (mDebug) {
      std::cout << "\n=====================\nOutput ROFs:\n";
      rofProcessor.dumpOutputROFs();
    }

    // send the output buffer via DPL
    size_t rofsSize;
    char* rofsBuffer = rofProcessor.saveROFRsToBuffer(rofsSize);

    // create the output message
    auto freefct = [](void* data, void*) { free(data); };
    pc.outputs().adoptChunk(Output{header::gDataOriginMCH, "TIMECLUSTERROFS", 0}, rofsBuffer, rofsSize, freefct, nullptr);

    mTFcount += 1;
  }

 private:
  std::chrono::duration<double, std::milli> mTimeProcess{}; ///< timer

  uint32_t mTimeClusterWidth; ///< maximum size of one time cluster, in bunch crossings
  uint32_t mNbinsInOneWindow; ///< number of time bins considered for the peak search
  int mTFcount{0};            ///< number of processed time frames
  int mDebug{0};              ///< verbosity flag
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTimeClusterFinderSpec(const char* specName)
{
  return DataProcessorSpec{
    specName,
    Inputs{InputSpec{"rofs", header::gDataOriginMCH, "DIGITROFS", 0, Lifetime::Timeframe},
           InputSpec{"orbits", header::gDataOriginMCH, "ORBITS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{header::gDataOriginMCH, "TIMECLUSTERROFS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TimeClusterFinderTask>()},
    Options{{"debug", VariantType::Bool, false, {"enable verbose output"}},
            {"max-cluster-width", VariantType::Int, 1000 / 25, {"maximum time width of time clusters, in BC units"}},
            {"peak-search-nbins", VariantType::Int, 5, {"number of time bins for the peak search algorithm (must be an odd number >= 3)"}}}};
}

} // end namespace mch
} // end namespace o2
