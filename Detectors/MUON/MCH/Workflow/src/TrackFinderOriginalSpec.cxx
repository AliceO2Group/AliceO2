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

/// \file TrackFinderOriginalSpec.cxx
/// \brief Implementation of a data processor to read clusters, reconstruct tracks and send them
///
/// \author Philippe Pillot, Subatech

#include "TrackFinderOriginalSpec.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/Propagator.h"
#include "MCHTracking/TrackParam.h"
#include "MCHTracking/Track.h"
#include "MCHTracking/TrackFinderOriginal.h"
#include "MCHTracking/TrackExtrap.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TrackFinderTask
{
 public:
  //_________________________________________________________________________________________________
  TrackFinderTask(bool computeTime, bool digits, std::shared_ptr<base::GRPGeomRequest> req)
    : mComputeTime(computeTime), mDigits(digits), mCCDBRequest(req) {}

  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the track extrapolation tools

    LOG(info) << "initializing track finder";

    if (mCCDBRequest) {
      base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    } else {
      auto grpFile = ic.options().get<std::string>("grp-file");
      if (std::filesystem::exists(grpFile)) {
        const auto grp = parameters::GRPObject::loadFrom(grpFile);
        base::Propagator::initFieldFromGRP(grp);
        TrackExtrap::setField();
      } else {
        float l3Current = ic.options().get<float>("l3Current");
        float dipoleCurrent = ic.options().get<float>("dipoleCurrent");
        mTrackFinder.initField(l3Current, dipoleCurrent);
      }
    }

    auto config = ic.options().get<std::string>("mch-config");
    if (!config.empty()) {
      o2::conf::ConfigurableParam::updateFromFile(config, "MCHTracking", true);
    }
    mTrackFinder.init();

    auto debugLevel = ic.options().get<int>("mch-debug");
    mTrackFinder.debug(debugLevel);

    auto stop = [this]() {
      mTrackFinder.printStats();
      mTrackFinder.printTimers();
      LOG(info) << "tracking duration = " << mElapsedTime.count() << " s";
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj)
  {
    /// finalize the track extrapolation setting
    if (mCCDBRequest && base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      if (matcher == framework::ConcreteDataMatcher("GLO", "GRPMAGFIELD", 0)) {
        TrackExtrap::setField();
      }
    }
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// for each event in the current TF, read the clusters and find tracks, then send them all

    if (mCCDBRequest) {
      base::GRPGeomHelper::instance().checkUpdates(pc);
    }

    uint32_t firstTForbit = pc.services().get<o2::framework::TimingInfo>().firstTForbit;

    // get the input messages with clusters and associated digits if needed
    auto clusterROFs = pc.inputs().get<gsl::span<ROFRecord>>("clusterrofs");
    auto clustersIn = pc.inputs().get<gsl::span<Cluster>>("clusters");
    gsl::span<const Digit> digitsIn{};
    if (mComputeTime || mDigits) {
      digitsIn = pc.inputs().get<gsl::span<Digit>>("clusterdigits");
    }

    // create the output messages for tracks, attached clusters and associated digits if requested
    auto& trackROFs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"trackrofs"});
    auto& mchTracks = pc.outputs().make<std::vector<TrackMCH>>(OutputRef{"tracks"});
    auto& usedClusters = pc.outputs().make<std::vector<Cluster>>(OutputRef{"trackclusters"});
    std::vector<Digit, o2::pmr::polymorphic_allocator<Digit>>* usedDigits(nullptr);
    if (mDigits) {
      usedDigits = &pc.outputs().make<std::vector<Digit>>(OutputRef{"trackdigits"});
    }

    trackROFs.reserve(clusterROFs.size());
    auto timeStart = std::chrono::high_resolution_clock::now();

    for (const auto& clusterROF : clusterROFs) {

      // sort the input clusters of the current event per chamber
      std::array<std::list<const Cluster*>, 10> clusters{};
      for (const auto& cluster : clustersIn.subspan(clusterROF.getFirstIdx(), clusterROF.getNEntries())) {
        clusters[cluster.getChamberId()].emplace_back(&cluster);
      }

      // run the track finder
      auto tStart = std::chrono::high_resolution_clock::now();
      const auto& tracks = mTrackFinder.findTracks(clusters);
      auto tEnd = std::chrono::high_resolution_clock::now();
      mElapsedTime += tEnd - tStart;

      // fill the ouput messages
      int trackOffset(mchTracks.size());
      writeTracks(tracks, digitsIn, clusterROF, firstTForbit, mchTracks, usedClusters, usedDigits);
      trackROFs.emplace_back(clusterROF.getBCData(), trackOffset, mchTracks.size() - trackOffset,
                             clusterROF.getBCWidth());
    }

    auto timeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = timeEnd - timeStart;
    LOGP(info, "Found {:3d} MCH tracks from {:4d} clusters in {:2d} ROFs in {:8.0f} ms",
         mchTracks.size(), clustersIn.size(), clusterROFs.size(), elapsed.count());
  }

 private:
  //_________________________________________________________________________________________________
  TrackMCH::Time computeTrackTime(const Track& track, const gsl::span<const Digit>& digitsIn,
                                  const ROFRecord& clusterROF, uint32_t firstTForbit) const
  {
    /// compute the track time

    double trackBCinTF = 0.;
    int nDigits = 0;

    // loop over associated digits and compute the average digits time
    for (const auto& param : track) {
      for (const auto& digit : digitsIn.subspan(param.getClusterPtr()->firstDigit, param.getClusterPtr()->nDigits)) {
        nDigits += 1;
        trackBCinTF += (double(digit.getTime()) - trackBCinTF) / nDigits;
      }
    }

    // set the track time from the computed average digits time
    if (nDigits > 0) {
      // convert the average digit time from bunch-crossing units to microseconds
      // add 1.5 BC to account for the fact that the actual digit time in BC units
      // can be between t and t+3, hence t+1.5 in average
      float tMean = o2::constants::lhc::LHCBunchSpacingMUS * (trackBCinTF + 1.5);
      float tErr = o2::constants::lhc::LHCBunchSpacingMUS * mTrackTime3Sigma;
      return TrackMCH::Time(tMean, tErr);
    }

    // if no digits are found, compute the time directly from the cluster's ROF
    LOG(fatal) << "MCH: no digits found when computing the track mean time";
    return clusterROF.getTimeMUS({0, firstTForbit}).first;
  }

  //_________________________________________________________________________________________________
  void writeTracks(const std::list<Track>& tracks, const gsl::span<const Digit>& digitsIn,
                   const ROFRecord& clusterROF, uint32_t firstTForbit,
                   std::vector<TrackMCH, o2::pmr::polymorphic_allocator<TrackMCH>>& mchTracks,
                   std::vector<Cluster, o2::pmr::polymorphic_allocator<Cluster>>& usedClusters,
                   std::vector<Digit, o2::pmr::polymorphic_allocator<Digit>>* usedDigits) const
  {
    /// fill the output messages with tracks and attached clusters and digits if requested

    // map the location of the attached digits between the digitsIn and the usedDigits lists
    std::unordered_map<uint32_t, uint32_t> digitLocMap{};

    for (const auto& track : tracks) {

      TrackParam paramAtMID(track.last());
      if (!TrackExtrap::extrapToMID(paramAtMID)) {
        LOG(warning) << "propagation to MID failed --> track discarded";
        continue;
      }

      const auto time = mComputeTime ? computeTrackTime(track, digitsIn, clusterROF, firstTForbit)
                                     : clusterROF.getTimeMUS({0, firstTForbit}).first;

      const auto& param = track.first();
      mchTracks.emplace_back(param.getZ(), param.getParameters(), param.getCovariances(),
                             param.getTrackChi2(), usedClusters.size(), track.getNClusters(),
                             paramAtMID.getZ(), paramAtMID.getParameters(), paramAtMID.getCovariances(),
                             time);

      for (const auto& param : track) {

        usedClusters.emplace_back(*param.getClusterPtr());

        if (mDigits) {

          // map the location of the digits associated to this cluster in the usedDigits list, if not already done
          auto& cluster = usedClusters.back();
          auto digitLoc = digitLocMap.emplace(cluster.firstDigit, usedDigits->size());

          // add the digits associated to this cluster if not already there
          if (digitLoc.second) {
            auto itFirstDigit = digitsIn.begin() + cluster.firstDigit;
            usedDigits->insert(usedDigits->end(), itFirstDigit, itFirstDigit + cluster.nDigits);
          }

          // make the cluster point to the associated digits in the usedDigits list
          cluster.firstDigit = digitLoc.first->second;
        }
      }
    }
  }

  bool mComputeTime = false;                            ///< compute the track time from the associated digits
  bool mDigits = false;                                 ///< send to associated digits
  std::shared_ptr<base::GRPGeomRequest> mCCDBRequest{}; ///< pointer to the CCDB requests
  float mTrackTime3Sigma{6.0};                          ///< three times the digit time resolution, in BC units
  TrackFinderOriginal mTrackFinder{};                   ///< track finder
  std::chrono::duration<double> mElapsedTime{};         ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackFinderOriginalSpec(const char* specName, bool computeTime, bool digits,
                                                            bool disableCCDBMagField)
{
  std::vector<InputSpec> inputSpecs{};
  inputSpecs.emplace_back(InputSpec{"clusterrofs", "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe});
  inputSpecs.emplace_back(InputSpec{"clusters", "MCH", "GLOBALCLUSTERS", 0, Lifetime::Timeframe});
  if (computeTime || digits) {
    inputSpecs.emplace_back(InputSpec{"clusterdigits", "MCH", "CLUSTERDIGITS", 0, Lifetime::Timeframe});
  }

  std::vector<OutputSpec> outputSpecs{};
  outputSpecs.emplace_back(OutputSpec{{"trackrofs"}, "MCH", "TRACKROFS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"tracks"}, "MCH", "TRACKS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"trackclusters"}, "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe});
  if (digits) {
    outputSpecs.emplace_back(OutputSpec{{"trackdigits"}, "MCH", "TRACKDIGITS", 0, Lifetime::Timeframe});
  }

  auto ccdbRequest = disableCCDBMagField ? nullptr
                                         : std::make_shared<base::GRPGeomRequest>(false,                      // orbitResetTime
                                                                                  false,                      // GRPECS=true
                                                                                  false,                      // GRPLHCIF
                                                                                  true,                       // GRPMagField
                                                                                  false,                      // askMatLUT
                                                                                  base::GRPGeomRequest::None, // geometry
                                                                                  inputSpecs);

  return DataProcessorSpec{
    specName,
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TrackFinderTask>(computeTime, digits, ccdbRequest)},
    Options{{"l3Current", VariantType::Float, -30000.0f, {"L3 current"}},
            {"dipoleCurrent", VariantType::Float, -6000.0f, {"Dipole current"}},
            {"grp-file", VariantType::String, o2::base::NameConf::getGRPFileName(), {"Name of the grp file"}},
            {"mch-config", VariantType::String, "", {"JSON or INI file with tracking parameters"}},
            {"mch-debug", VariantType::Int, 0, {"debug level"}}}};
}

} // namespace mch
} // namespace o2
