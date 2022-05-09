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

#include "TimeClusterFinderSpec.h"

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
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

#include "MCHTimeClustering/ROFTimeClusterFinder.h"
#include "MCHBase/Trackable.h"
#include "MCHTracking/TrackerParam.h"
#include "MCHTimeClustering/TimeClusterizerParam.h"

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
    const auto& param = TimeClusterizerParam::Instance();
    mTimeClusterWidth = param.maxClusterWidth;
    mNbinsInOneWindow = param.peakSearchNbins;
    mMinDigitPerROF = param.minDigitsPerROF;
    mOnlyTrackable = param.onlyTrackable;
    mDebug = ic.options().get<bool>("mch-debug");

    if (mDebug) {
      fair::Logger::SetConsoleColor(true);
    }
    // number of bins must be >= 3
    if (mNbinsInOneWindow < 3) {
      mNbinsInOneWindow = 3;
    }
    // number of bins must be odd
    if ((mNbinsInOneWindow % 2) == 0) {
      mNbinsInOneWindow += 1;
    }

    LOGP(info, "max-cluster-width : {}", mTimeClusterWidth);
    LOGP(info, "peak-search-nbins: {} ", mNbinsInOneWindow);
    LOGP(info, "min-digits-per-rof: {}", mMinDigitPerROF);
    LOGP(info, "only-trackable : {}", mOnlyTrackable);

    auto stop = [this]() {
      if (mTFcount) {
        LOGP(info, "\nduration = {} us / TF\n", mTimeProcess.count() * 1000 / mTFcount);
      }
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    auto rofs = pc.inputs().get<gsl::span<o2::mch::ROFRecord>>("rofs");
    auto digits = pc.inputs().get<gsl::span<o2::mch::Digit>>("digits");

    o2::mch::ROFTimeClusterFinder rofProcessor(rofs, mTimeClusterWidth, mNbinsInOneWindow, 0);

    if (mDebug) {
      LOGP(warning, "{:=>60} ", fmt::format("{:6d} Input ROFS", rofs.size()));
      // rofProcessor.dumpInputROFs();
    }

    auto tStart = std::chrono::high_resolution_clock::now();
    rofProcessor.process();
    auto tEnd = std::chrono::high_resolution_clock::now();
    mTimeProcess += tEnd - tStart;

    if (mDebug) {
      LOGP(warning, "{:=>60} ", fmt::format("{:6d} Output ROFS", rofProcessor.getROFRecords().size()));
      // rofProcessor.dumpOutputROFs();
    }

    auto& outRofs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"rofs"});
    const auto& pRofs = rofProcessor.getROFRecords();

    const auto& trackerParam = TrackerParam::Instance();
    std::array<bool, 5> requestStation{
      trackerParam.requestStation[0],
      trackerParam.requestStation[1],
      trackerParam.requestStation[2],
      trackerParam.requestStation[3],
      trackerParam.requestStation[4]};

    auto tfilter = createTrackableFilter(digits, mOnlyTrackable,
                                         requestStation,
                                         trackerParam.moreCandidates);

    std::copy_if(begin(pRofs),
                 end(pRofs),
                 std::back_inserter(outRofs),
                 [this, tfilter](const o2::mch::ROFRecord& rof) {
                   return rof.getNEntries() > mMinDigitPerROF &&
                          tfilter(rof);
                 });

    const float p1 = rofs.size() > 0 ? 100. * pRofs.size() / rofs.size() : 0;
    const float p2 = rofs.size() > 0 ? 100. * outRofs.size() / rofs.size() : 0;

    LOGP(info,
         "TF {} Processed {} input ROFs, "
         "time-clusterized them into {} ROFs ({:3.0f}%) "
         "and output {} ({:3.0f}%) "
         "of them {}",
         mTFcount, rofs.size(),
         pRofs.size(), p1,
         outRofs.size(), p2,
         mOnlyTrackable ? "(only trackable ones)" : "");
    mTFcount += 1;
  }

 private:
  std::chrono::duration<double, std::milli>
    mTimeProcess{}; ///< timer

  uint32_t mTimeClusterWidth; ///< maximum size of one time cluster, in bunch crossings
  uint32_t mNbinsInOneWindow; ///< number of time bins considered for the peak search
  int mTFcount{0};            ///< number of processed time frames
  int mDebug{0};              ///< verbosity flag
  int mMinDigitPerROF;        ///< minimum digit per ROF threshold
  bool mOnlyTrackable;        ///< only keep ROFs that are trackable
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec
  getTimeClusterFinderSpec(const char* specName,
                           std::string_view inputDigitDataDescription,
                           std::string_view inputDigitRofDataDescription,
                           std::string_view outputDigitRofDataDescription)
{
  std::string input = fmt::format("rofs:MCH/{}/0;digits:MCH/{}/0",
                                  inputDigitRofDataDescription.data(),
                                  inputDigitDataDescription.data());
  std::string output = fmt::format("rofs:MCH/{}/0", outputDigitRofDataDescription.data());

  std::vector<OutputSpec> outputs;
  auto matchers = select(output.c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
  }

  return DataProcessorSpec{
    specName,
    Inputs{select(input.c_str())},
    outputs,
    AlgorithmSpec{adaptFromTask<TimeClusterFinderTask>()},
    Options{{"mch-debug", VariantType::Bool, false, {"enable verbose output"}}}};
}
} // end namespace mch
} // end namespace o2
