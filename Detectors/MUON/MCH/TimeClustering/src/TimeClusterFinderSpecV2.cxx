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

/// \file TimeClusterFinderSpecV2.cxx
/// \brief Implementation of a data processor to run the time clusterizer
///
/// \author Andrea Ferrero, CEA

#include "MCHTimeClustering/TimeClusterFinderSpecV2.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <fmt/core.h>

#include "CommonDataFormat/IRFrame.h"

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

#include "MCHBase/TrackerParam.h"
#include "MCHDigitFiltering/DigitFilter.h"
#include "MCHROFFiltering/IRFrameFilter.h"
#include "MCHROFFiltering/MultiplicityFilter.h"
#include "MCHROFFiltering/TrackableFilter.h"
#include "MCHTimeClustering/ROFTimeClusterFinderV2.h"
#include "MCHTimeClustering/TimeClusterizerParamV2.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TimeClusterFinderTaskV2
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    const auto& param = TimeClusterizerParamV2::Instance();
    mTimeClusterWidth = param.maxClusterWidth;
    mPeakSearchWindow = param.peakSearchWindow;
    mPeakSearchNbins = param.peakSearchNbins;
    mPeakSearchNDigitsMin = param.peakSearchNDigitsMin;
    mMinDigitPerROF = param.minDigitsPerROF;
    mOnlyTrackable = param.onlyTrackable;
    mPeakSearchSignalOnly = param.peakSearchSignalOnly;
    mMergeROFs = param.mergeROFs;
    mIRFramesOnly = param.irFramesOnly;
    mDebug = ic.options().get<bool>("mch-debug");
    mROFRejectionFraction = param.rofRejectionFraction;

    if (mDebug) {
      fair::Logger::SetConsoleColor(true);
    }
    // number of bins must be >= 3
    if (mPeakSearchNbins < 3) {
      mPeakSearchNbins = 3;
    }
    // number of bins must be odd
    if ((mPeakSearchNbins % 2) == 0) {
      mPeakSearchNbins += 1;
    }
    if (mROFRejectionFraction > 0) {
      std::random_device rd;
      mGenerator = std::mt19937(rd());
    }

    LOGP(info, "TimeClusterWidth      : {}", mTimeClusterWidth);
    LOGP(info, "PeakSearchNbins       : {}", mPeakSearchNbins);
    LOGP(info, "MinDigitPerROF        : {}", mMinDigitPerROF);
    LOGP(info, "OnlyTrackable         : {}", mOnlyTrackable);
    LOGP(info, "PeakSearchSignalOnly  : {}", mPeakSearchSignalOnly);
    LOGP(info, "IRFramesOnly          : {}", mIRFramesOnly);
    LOGP(info, "ROFRejectionFraction  : {}", mROFRejectionFraction);

    auto stop = [this]() {
      if (mTFcount) {
        LOGP(info, "duration = {} us / TF", mTimeProcess.count() * 1000 / mTFcount);
      }
    };
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(stop);
  }

  ROFFilter createRandomRejectionFilter(float rejectionFraction)
  {
    return [this, rejectionFraction](const ROFRecord& /*rof*/) {
      double rnd = mDistribution(mGenerator);
      return rnd > rejectionFraction;
    };
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    auto rofs = pc.inputs().get<gsl::span<o2::mch::ROFRecord>>("rofs");
    auto digits = pc.inputs().get<gsl::span<o2::mch::Digit>>("digits");

    o2::mch::ROFTimeClusterFinderV2 rofProcessor(rofs,
                                                 digits,
                                                 mTimeClusterWidth,
                                                 mPeakSearchWindow,
                                                 mPeakSearchNbins,
                                                 mPeakSearchNDigitsMin,
                                                 mPeakSearchSignalOnly,
                                                 mMergeROFs,
                                                 mDebug);

    if (mDebug) {
      LOGP(warning, "{:=>60} ", fmt::format("{:6d} Input ROFS", rofs.size()));
    }

    auto tStart = std::chrono::high_resolution_clock::now();
    rofProcessor.process();
    auto tEnd = std::chrono::high_resolution_clock::now();
    mTimeProcess += tEnd - tStart;

    if (mDebug) {
      LOGP(warning, "{:=>60} ", fmt::format("{:6d} Output ROFS", rofProcessor.getROFRecords().size()));
    }

    auto& outRofs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"rofs"});
    const auto& pRofs = rofProcessor.getROFRecords();

    // prepare the list of filters we want to apply to ROFs
    std::vector<ROFFilter> filters;

    if (mOnlyTrackable) {
      // selects only ROFs that are trackable
      const auto& trackerParam = TrackerParam::Instance();
      std::array<bool, 5> requestStation{
        trackerParam.requestStation[0],
        trackerParam.requestStation[1],
        trackerParam.requestStation[2],
        trackerParam.requestStation[3],
        trackerParam.requestStation[4]};
      filters.emplace_back(createTrackableFilter(digits,
                                                 requestStation,
                                                 trackerParam.moreCandidates));
    }
    if (mMinDigitPerROF > 0) {
      // selects only those ROFs have that minimum number of digits
      filters.emplace_back(createMultiplicityFilter(mMinDigitPerROF));
    }
    if (mIRFramesOnly) {
      // selects only those ROFs that overlop some IRFrame
      auto irFrames = pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("irframes");
      filters.emplace_back(createIRFrameFilter(irFrames));
    }

    std::string extraMsg = "";

    if (mROFRejectionFraction > 0) {
      filters.emplace_back(createRandomRejectionFilter(mROFRejectionFraction));
      extraMsg = fmt::format(" (CAUTION : hard-rejected {:3.0f}% of the output ROFs)", mROFRejectionFraction * 100);
    }

    // a single filter which is the AND combination of the elements of the filters vector
    auto filter = createROFFilter(filters);

    std::copy_if(begin(pRofs),
                 end(pRofs),
                 std::back_inserter(outRofs),
                 filter);

    const float p1 = rofs.size() > 0 ? 100. * pRofs.size() / rofs.size() : 0;
    const float p2 = rofs.size() > 0 ? 100. * outRofs.size() / rofs.size() : 0;

    LOGP(info,
         "NEW TF {} Processed {} input ROFs, "
         "time-clusterized them into {} ROFs ({:3.0f}%) "
         "and output {} ({:3.0f}%) of them{}",
         mTFcount, rofs.size(),
         pRofs.size(), p1,
         outRofs.size(), p2, extraMsg);
    mTFcount += 1;

    std::vector<std::pair<int,int>> orbits = {
        //{376285608, 1455},
        //{376285618, 1075},
        //{376293180, 2152},
        //{376353595, 1807},
        //{376376230, 1441},
        //{376285604, 828},
        //{376429114, 2550},
        //{376670762, 2548},
        //{376678313, 453},
        //{376867115, 2335},
        //{378166083, 1029}
        //{376270496, 2752},
        //{376300693, 3249},
        //{376376238, 352},
        //{376376238, 1843},
        //{376421564, 1702}
        //{376361141, 2781},
        //{376784078, 3156}
    };
    //int orbit = 376293180; //376285618; //376285608;
    for (auto o : orbits) {
      int orbit = o.first;
      int bc = o.second;
      std::ofstream rofsOutput;
      int id = 0;
      for (auto& rof : pRofs) {
        if (rof.getBCData().orbit != orbit) {
          continue;
        }
        if (rof.getBCData().bc < (bc - 500)) {
          continue;
        }
        if (rof.getBCData().bc > (bc + 500)) {
          continue;
        }
        if (!rofsOutput.is_open()) {
          rofsOutput.open(fmt::format("rofs-{}.txt", orbit));
        }
        for (auto& irof : rofs) {
          if (irof.getBCData().orbit != orbit) {
            continue;
          }
          if (irof.getBCData().bc < rof.getBCData().bc) {
            continue;
          }
          if (irof.getBCData().bc >= (rof.getBCData().bc + rof.getBCWidth())) {
            break;
          }
          rofsOutput << fmt::format("{:>3}", id) << " " << fmt::format("{:>3}", rof.getBCWidth()) << (filter(rof) ? " + " : " - ")
                << fmt::format("{:>10}", irof.getBCData().orbit) << " " << fmt::format("{:>4}", irof.getBCData().bc) << " " << fmt::format("{:>4} ", irof.getNEntries());
          for (int i = 0; i < irof.getNEntries(); i++) {
            rofsOutput << "*";
            if (i > 100) break;
          }
          rofsOutput << std::endl;
        }
        rofsOutput << "------" << std::endl;
        id += 1;
      }
      rofsOutput.close();
    }
    for (auto& rof : pRofs) {
      std::ofstream rofsOutput;
      if (rof.getBCWidth() < 200) {
        continue;
      }
      int orbit = rof.getBCData().orbit;
      if (!rofsOutput.is_open()) {
        rofsOutput.open(fmt::format("rofs-{}-{}.txt", orbit, rof.getBCData().bc));
      }
      for (auto& irof : rofs) {
        if (irof.getBCData().orbit != orbit) {
          continue;
        }
        if (irof.getBCData().bc < rof.getBCData().bc) {
          continue;
        }
        if (irof.getBCData().bc >= (rof.getBCData().bc + rof.getBCWidth())) {
          break;
        }
        rofsOutput << fmt::format("{:>3}", rof.getBCWidth()) << (filter(rof) ? " + " : " - ")
                  << fmt::format("{:>10}", irof.getBCData().orbit) << " " << fmt::format("{:>4}", irof.getBCData().bc) << " " << fmt::format("{:>4} ", irof.getNEntries());
        for (int i = 0; i < irof.getNEntries(); i++) {
          rofsOutput << "*";
          if (i > 100) break;
        }
        rofsOutput << std::endl;
      }
      rofsOutput.close();
    }
  }

 private:
  std::chrono::duration<double, std::milli>
    mTimeProcess{}; ///< timer

  uint32_t mTimeClusterWidth;     ///< maximum size of one time cluster, in bunch crossings
  uint32_t mPeakSearchWindow;     ///< width of the peak search window, in BC units
  uint32_t mPeakSearchNbins;      ///< number of time bins for the peak search algorithm (must be an odd number >= 3)
  uint32_t mPeakSearchNDigitsMin; ///< minimum number of digits for peak candidates

  int mTFcount{0};             ///< number of processed time frames
  int mDebug{0};               ///< verbosity flag
  int mMinDigitPerROF;         ///< minimum digit per ROF threshold
  bool mPeakSearchSignalOnly;  ///< only use signal-like hits in peak search
  bool mMergeROFs;             ///< whether to merge consecutive ROFs
  bool mOnlyTrackable;         ///< only keep ROFs that are trackable
  bool mIRFramesOnly;          ///< only keep ROFs that overlap some IRFrame
  float mROFRejectionFraction; ///< fraction of output ROFs to reject (to save time in sync reco). MUST BE 0 for anything but Pt2 reco !
  std::uniform_real_distribution<double> mDistribution{0.0, 1.0};
  std::mt19937 mGenerator;
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec
  getTimeClusterFinderSpecV2(const char* specName,
                             std::string_view inputDigitDataDescription,
                             std::string_view inputDigitRofDataDescription,
                             std::string_view outputDigitRofDataDescription,
                             std::string_view inputIRFrameDataDescription)
{
  std::string input = fmt::format("rofs:MCH/{}/0;digits:MCH/{}/0",
                                  inputDigitRofDataDescription.data(),
                                  inputDigitDataDescription.data());
  if (TimeClusterizerParamV2::Instance().irFramesOnly && inputIRFrameDataDescription.size()) {
    LOGP(info, "will select IRFrames from {}", inputIRFrameDataDescription);
    input += ";irframes:";
    input += inputIRFrameDataDescription;
  }
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
    AlgorithmSpec{adaptFromTask<TimeClusterFinderTaskV2>()},
    Options{{"mch-debug", VariantType::Bool, false, {"enable verbose output"}}}};
}
} // end namespace mch
} // end namespace o2
