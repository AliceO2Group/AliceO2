// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PreClusterFinderSpec.cxx
/// \brief Implementation of a data processor to run the preclusterizer
///
/// \author Philippe Pillot, Subatech

#include "MCHWorkflow/PreClusterFinderSpec.h"

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
#include "MCHBase/Digit.h"
#include "MCHBase/PreCluster.h"
#include "MCHPreClustering/PreClusterFinder.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

enum tCheckNoLeftoverDigits {
  CHECK_NO_LEFTOVER_DIGITS_OFF,
  CHECK_NO_LEFTOVER_DIGITS_ERROR,
  CHECK_NO_LEFTOVER_DIGITS_FATAL
};

class PreClusterFinderTask
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

    auto checkNoLeftoverDigits = ic.options().get<std::string>("check-no-leftover-digits");
    if (checkNoLeftoverDigits == "off") {
      mCheckNoLeftoverDigits = CHECK_NO_LEFTOVER_DIGITS_OFF;
    } else if (checkNoLeftoverDigits == "error") {
      mCheckNoLeftoverDigits = CHECK_NO_LEFTOVER_DIGITS_ERROR;
    } else if (checkNoLeftoverDigits == "fatal") {
      mCheckNoLeftoverDigits = CHECK_NO_LEFTOVER_DIGITS_FATAL;
    }
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the digits, preclusterize and send the preclusters for each event in the current TF

    // get the input messages with digits
    auto digitROFs = pc.inputs().get<gsl::span<ROFRecord>>("digitrofs");
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    //LOG(INFO) << "received time frame with " << digitROFs.size() << " interactions";

    // create the output message for precluster ROFs
    auto& preClusterROFs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"preclusterrofs"});
    preClusterROFs.reserve(digitROFs.size());

    // prepare to receive new data
    mPreClusters.clear();
    mUsedDigits.clear();
    mUsedDigits.reserve(digits.size());

    for (const auto& digitROF : digitROFs) {

      //LOG(INFO) << "processing interaction: " << digitROF.getBCData() << "...";

      // prepare to receive new data
      auto tStart = std::chrono::high_resolution_clock::now();
      mPreClusterFinder.reset();
      auto tEnd = std::chrono::high_resolution_clock::now();
      mTimeResetPreClusterFinder += tEnd - tStart;

      // load the digits to get the fired pads
      tStart = std::chrono::high_resolution_clock::now();
      mPreClusterFinder.loadDigits(digits.subspan(digitROF.getFirstIdx(), digitROF.getNEntries()));
      tEnd = std::chrono::high_resolution_clock::now();
      mTimeLoadDigits += tEnd - tStart;

      // preclusterize
      tStart = std::chrono::high_resolution_clock::now();
      int nPreClusters = mPreClusterFinder.run();
      tEnd = std::chrono::high_resolution_clock::now();
      mTimePreClusterFinder += tEnd - tStart;

      // get the preclusters and associated digits
      tStart = std::chrono::high_resolution_clock::now();
      preClusterROFs.emplace_back(digitROF.getBCData(), mPreClusters.size(), nPreClusters);
      mPreClusterFinder.getPreClusters(mPreClusters, mUsedDigits);
      tEnd = std::chrono::high_resolution_clock::now();
      mTimeStorePreClusters += tEnd - tStart;
    }

    // check sizes of input and output digits vectors
    bool digitsSizesDiffer = (mUsedDigits.size() != digits.size());
    switch (mCheckNoLeftoverDigits) {
      case CHECK_NO_LEFTOVER_DIGITS_OFF:
        break;
      case CHECK_NO_LEFTOVER_DIGITS_ERROR:
        if (digitsSizesDiffer) {
          LOG(ERROR) << "some digits have been lost during the preclustering";
        }
        break;
      case CHECK_NO_LEFTOVER_DIGITS_FATAL:
        if (digitsSizesDiffer) {
          throw runtime_error("some digits have been lost during the preclustering");
        }
        break;
    };

    // create the output messages for preclusters and associated digits
    pc.outputs().snapshot(OutputRef{"preclusters"}, mPreClusters);
    pc.outputs().snapshot(OutputRef{"preclusterdigits"}, mUsedDigits);
  }

 private:
  PreClusterFinder mPreClusterFinder{};   ///< preclusterizer
  std::vector<PreCluster> mPreClusters{}; ///< vector of preclusters
  std::vector<Digit> mUsedDigits{};       ///< vector of digits in the preclusters

  int mCheckNoLeftoverDigits{CHECK_NO_LEFTOVER_DIGITS_ERROR}; ///< digits vector size check option

  std::chrono::duration<double, std::milli> mTimeResetPreClusterFinder{}; ///< timer
  std::chrono::duration<double, std::milli> mTimeLoadDigits{};            ///< timer
  std::chrono::duration<double, std::milli> mTimePreClusterFinder{};      ///< timer
  std::chrono::duration<double, std::milli> mTimeStorePreClusters{};      ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getPreClusterFinderSpec()
{
  std::string helpstr = "[off/error/fatal] check that all digits are included in pre-clusters";
  return DataProcessorSpec{
    "PreClusterFinder",
    Inputs{InputSpec{"digitrofs", "MCH", "DIGITROFS", 0, Lifetime::Timeframe},
           InputSpec{"digits", "MCH", "DIGITS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{{"preclusterrofs"}, "MCH", "PRECLUSTERROFS", 0, Lifetime::Timeframe},
            OutputSpec{{"preclusters"}, "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe},
            OutputSpec{{"preclusterdigits"}, "MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<PreClusterFinderTask>()},
    Options{{"check-no-leftover-digits", VariantType::String, "error", {helpstr}}}};
}

} // end namespace mch
} // end namespace o2
