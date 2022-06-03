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
#include "Framework/WorkflowSpec.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "MCHBase/PreCluster.h"
#include "MCHPreClustering/PreClusterFinder.h"
#include "MCHBase/SanityCheck.h"

#include <iostream>
#include <chrono>
#include <vector>

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
    LOG(info) << "initializing preclusterizer";

    mPreClusterFinder.init();

    auto stop = [this]() {
      LOG(info) << "reset precluster finder duration = " << mTimeResetPreClusterFinder.count() << " ms";
      LOG(info) << "load digits duration = " << mTimeLoadDigits.count() << " ms";
      LOG(info) << "discard high occupancy duration = " << mTimeDiscardHighOccupancy.count() << " ms";
      LOG(info) << "precluster finder duration = " << mTimePreClusterFinder.count() << " ms";
      LOG(info) << "store precluster duration = " << mTimeStorePreClusters.count() << " ms";
      /// Clear the preclusterizer
      auto tStart = std::chrono::high_resolution_clock::now();
      this->mPreClusterFinder.deinit();
      auto tEnd = std::chrono::high_resolution_clock::now();
      LOG(info) << "deinitializing preclusterizer in: "
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
    mDiscardHighOccDEs = ic.options().get<bool>("discard-high-occupancy-des");
    mDiscardHighOccEvents = ic.options().get<bool>("discard-high-occupancy-events");
    mSanityCheck = ic.options().get<bool>("sanity-check");
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the digits, preclusterize and send the preclusters for each event in the current TF

    // get the input messages with digits
    auto digitROFs = pc.inputs().get<gsl::span<ROFRecord>>("digitrofs");
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    bool abort{false};
    if (mSanityCheck) {
      auto error = sanityCheck(digitROFs, digits);

      if (!isOK(error)) {
        if (error.nofOutOfBounds > 0) {
          // FIXME: replace this error log with a counters' message ?
          LOGP(error, asString(error));
          LOGP(error, "in a TF with {} rofs and {} digits", digitROFs.size(), digits.size());
          abort = true;
        }
      }
    }

    // create the output message for precluster ROFs
    auto& preClusterROFs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"preclusterrofs"});

    // count the number of digits associated with the input ROFs. This can be smaller than the
    // total number of digits if we are processing filtered ROFs.
    int nDigitsInRofs = 0;
    for (const auto& digitROF : digitROFs) {
      nDigitsInRofs += digitROF.getNEntries();
    }

    // prepare to receive new data
    mPreClusters.clear();
    mUsedDigits.clear();

    if (!abort) {

      preClusterROFs.reserve(digitROFs.size());
      mUsedDigits.reserve(digits.size());
      int nRemovedDigits(0);

      for (const auto& digitROF : digitROFs) {

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

        // discard high-occupancy (noisy) DEs and/or events
        tStart = std::chrono::high_resolution_clock::now();
        nRemovedDigits += mPreClusterFinder.discardHighOccupancy(mDiscardHighOccDEs, mDiscardHighOccEvents);
        tEnd = std::chrono::high_resolution_clock::now();
        mTimeDiscardHighOccupancy += tEnd - tStart;

        // preclusterize
        tStart = std::chrono::high_resolution_clock::now();
        int nPreClusters = mPreClusterFinder.run();
        tEnd = std::chrono::high_resolution_clock::now();
        mTimePreClusterFinder += tEnd - tStart;

        // get the preclusters and associated digits
        tStart = std::chrono::high_resolution_clock::now();
        preClusterROFs.emplace_back(digitROF.getBCData(), mPreClusters.size(), nPreClusters, digitROF.getBCWidth());
        mPreClusterFinder.getPreClusters(mPreClusters, mUsedDigits);
        tEnd = std::chrono::high_resolution_clock::now();
        mTimeStorePreClusters += tEnd - tStart;
      }

      // check sizes of input and output digits vectors
      bool digitsSizesDiffer = (nRemovedDigits + mUsedDigits.size() != nDigitsInRofs);
      switch (mCheckNoLeftoverDigits) {
        case CHECK_NO_LEFTOVER_DIGITS_OFF:
          break;
        case CHECK_NO_LEFTOVER_DIGITS_ERROR:
          if (digitsSizesDiffer) {
            static int nAlarms = 0;
            if (nAlarms++ < 5) {
              LOG(alarm) << "some digits have been lost during the preclustering";
            }
          }
          break;
        case CHECK_NO_LEFTOVER_DIGITS_FATAL:
          if (digitsSizesDiffer) {
            throw runtime_error("some digits have been lost during the preclustering");
          }
          break;
      };
    }

    // create the output messages for preclusters and associated digits
    pc.outputs().snapshot(OutputRef{"preclusters"}, mPreClusters);
    pc.outputs().snapshot(OutputRef{"preclusterdigits"}, mUsedDigits);

    LOGP(info, "Processed {} digit rofs with {} digits and output {} precluster rofs with {} preclusters and {} digits",
         digitROFs.size(),
         digits.size(),
         preClusterROFs.size(),
         mPreClusters.size(), mUsedDigits.size());
  }

 private:
  PreClusterFinder mPreClusterFinder{};   ///< preclusterizer
  std::vector<PreCluster> mPreClusters{}; ///< vector of preclusters
  std::vector<Digit> mUsedDigits{};       ///< vector of digits in the preclusters

  int mCheckNoLeftoverDigits{CHECK_NO_LEFTOVER_DIGITS_ERROR};             ///< digits vector size check option
  bool mDiscardHighOccDEs = false;                                        ///< discard DEs with occupancy > 20%
  bool mDiscardHighOccEvents = false;                                     ///< discard events with >= 5 DEs above 20% occupancy
  bool mSanityCheck = false;                                              ///< perform some input digit sanity checks
  std::chrono::duration<double, std::milli> mTimeResetPreClusterFinder{}; ///< timer
  std::chrono::duration<double, std::milli> mTimeLoadDigits{};            ///< timer
  std::chrono::duration<double, std::milli> mTimeDiscardHighOccupancy{};  ///< timer
  std::chrono::duration<double, std::milli> mTimePreClusterFinder{};      ///< timer
  std::chrono::duration<double, std::milli> mTimeStorePreClusters{};      ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getPreClusterFinderSpec(const char* specName,
                                                         std::string_view inputDigitDataDescription,
                                                         std::string_view inputDigitRofDataDescription)
{
  std::string input =
    fmt::format("digits:MCH/{}/0;digitrofs:MCH/{}/0",
                inputDigitDataDescription,
                inputDigitRofDataDescription);
  std::string helpstr = "[off/error/fatal] check that all digits are included in pre-clusters";

  return DataProcessorSpec{
    specName,
    o2::framework::select(input.c_str()),
    Outputs{OutputSpec{{"preclusterrofs"}, "MCH", "PRECLUSTERROFS", 0, Lifetime::Timeframe},
            OutputSpec{{"preclusters"}, "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe},
            OutputSpec{{"preclusterdigits"}, "MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<PreClusterFinderTask>()},
    Options{{"check-no-leftover-digits", VariantType::String, "error", {helpstr}},
            {{"sanity-check"}, VariantType::Bool, false, {"perform some input digit sanity checks"}},
            {"discard-high-occupancy-des", VariantType::Bool, false, {"discard DEs with occupancy > 20%"}},
            {"discard-high-occupancy-events", VariantType::Bool, false, {"discard events with >= 5 DEs above 20% occupancy"}}}};
}

} // end namespace mch
} // end namespace o2
