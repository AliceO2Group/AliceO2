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

/// \file TrackMCLabelFinderSpec.cxx
/// \brief Implementation of a data processor to match the reconstructed tracks with the simulated ones
///
/// \author Philippe Pillot, Subatech

#include "TrackMCLabelFinderSpec.h"

#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <gsl/span>

#include "DetectorsCommonDataFormats/DetID.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/WorkflowSpec.h"

#include "Steer/MCKinematicsReader.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/TrackReference.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/Cluster.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TrackMCLabelFinderTask
{
 public:
  //_________________________________________________________________________________________________
  /// prepare the task from the context
  void init(framework::InitContext& ic)
  {
    LOG(info) << "initializing track MC label finder";
    if (!mMCReader.initFromDigitContext(ic.options().get<std::string>("incontext").c_str())) {
      throw invalid_argument("initialization of MCKinematicsReader failed");
    }
    double sigmaCut = ic.options().get<double>("sigma-cut");
    mMaxMatchingChi2 = 2. * sigmaCut * sigmaCut;
  }

  //_________________________________________________________________________________________________
  /// assign every reconstructed track a MC label, pointing to the corresponding particle or fake
  void run(framework::ProcessingContext& pc)
  {
    // get the input messages
    auto digitROFs = pc.inputs().get<gsl::span<ROFRecord>>("digitrofs");
    auto digitlabels = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("digitlabels");
    auto trackROFs = pc.inputs().get<gsl::span<ROFRecord>>("trackrofs");
    auto tracks = pc.inputs().get<gsl::span<TrackMCH>>("tracks");
    auto clusters = pc.inputs().get<gsl::span<Cluster>>("clusters");

    // digit and track ROFs must be synchronized
    int nROFs = digitROFs.size();
    if (trackROFs.size() != nROFs) {
      throw length_error(fmt::format("inconsistent ROFs: {} tracksROFs vs {} digitROFs", trackROFs.size(), nROFs));
    }

    std::vector<o2::MCCompLabel> tracklabels;

    for (int iROF = 0; iROF < nROFs; ++iROF) {

      // skip interation records with no reconstructed tracks
      auto trackROF = trackROFs[iROF];
      if (trackROF.getNEntries() == 0) {
        continue;
      }

      // digits and tracks must belong to the same IR
      auto digitROF = digitROFs[iROF];
      if (digitROF.getBCData() != trackROF.getBCData()) {
        throw logic_error("inconsistent ROF");
      }

      // get the MC labels of every particles contributing to that IR and produce the associated MC clusters
      std::unordered_map<MCCompLabel, std::vector<Cluster>> mcClusterMap{};
      for (int iDigit = digitROF.getFirstIdx(); iDigit <= digitROF.getLastIdx(); ++iDigit) {
        for (const auto& label : digitlabels->getLabels(iDigit)) {
          if (label.isCorrect() && mcClusterMap.count(label) == 0) {
            makeMCClusters(mMCReader.getTrackRefs(label.getSourceID(), label.getEventID(), label.getTrackID()),
                           mcClusterMap[label]);
          }
        }
      }

      // try to pair every reconstructed tracks with every MC tracks to set MC labels
      for (int iTrack = trackROF.getFirstIdx(); iTrack <= trackROF.getLastIdx(); ++iTrack) {
        for (const auto& [label, mcClusters] : mcClusterMap) {
          if (match(clusters.subspan(tracks[iTrack].getFirstClusterIdx(), tracks[iTrack].getNClusters()), mcClusters)) {
            tracklabels.push_back(label);
            break;
          }
        }
        if (tracklabels.size() != iTrack + 1) {
          tracklabels.push_back(MCCompLabel());
        }
      }
    }

    // send the track MC labels
    pc.outputs().snapshot(OutputRef{"tracklabels"}, tracklabels);
  }

 private:
  //_________________________________________________________________________________________________
  /// produce the MC clusters by taking the average position of the trackRefs at the entry and exit of each DE
  void makeMCClusters(gsl::span<const TrackReference> mcTrackRefs, std::vector<Cluster>& mcClusters) const
  {
    int deId(-1);
    int clusterIdx(0);
    for (const auto& trackRef : mcTrackRefs) {
      if (trackRef.getDetectorId() != o2::detectors::DetID::MCH) {
        deId = -1;
        continue;
      }
      if (trackRef.getUserId() == deId) {
        auto& cluster = mcClusters.back();
        cluster.x = (cluster.x + trackRef.X()) / 2.;
        cluster.y = (cluster.y + trackRef.Y()) / 2.;
        cluster.z = (cluster.z + trackRef.Z()) / 2.;
        deId = -1; // to create a new cluster in case the track re-enter the DE (loop)
      } else {
        deId = trackRef.getUserId();
        mcClusters.push_back({trackRef.X(), trackRef.Y(), trackRef.Z(), 0.f, 0.f,
                              Cluster::buildUniqueId(deId / 100 - 1, deId, clusterIdx++), 0u, 0u});
      }
    }
  }

  //_________________________________________________________________________________________________
  /// try to match a reconstructed track with a MC track. Matching conditions:
  /// - more than 50% of reconstructed clusters matched with MC clusters
  /// - at least 1 matched cluster before and 1 after the dipole
  bool match(gsl::span<const Cluster> clusters, const std::vector<Cluster>& mcClusters) const
  {
    int nMatched(0);
    bool hasMatched[5] = {false, false, false, false, false};
    for (const auto& cluster : clusters) {
      for (const auto& mcCluster : mcClusters) {
        if (cluster.getDEId() == mcCluster.getDEId()) {
          double dx = cluster.x - mcCluster.x;
          double dy = cluster.y - mcCluster.y;
          double chi2 = dx * dx / (cluster.ex * cluster.ex) + dy * dy / (cluster.ey * cluster.ey);
          if (chi2 <= mMaxMatchingChi2) {
            hasMatched[cluster.getChamberId() / 2] = true;
            ++nMatched;
            break;
          }
        }
      }
    }
    return ((hasMatched[0] || hasMatched[1]) && (hasMatched[3] || hasMatched[4]) && 2 * nMatched > clusters.size());
  }

  steer::MCKinematicsReader mMCReader{}; ///< MC reader to retrieve the trackRefs
  double mMaxMatchingChi2 = 32.;         ///< maximum chi2 to match simulated and reconstructed clusters
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackMCLabelFinderSpec(const char* specName,
                                                           const char* digitRofDataDescription,
                                                           const char* digitLabelDataDescription)
{
  std::string input =
    fmt::format("digitrofs:MCH/{}/0;digitlabels:MCH/{}/0", digitRofDataDescription,
                digitLabelDataDescription);
  input +=
    ";trackrofs:MCH/TRACKROFS/0;"
    "tracks:MCH/TRACKS/0;"
    "clusters:MCH/TRACKCLUSTERS/0";
  return DataProcessorSpec{
    specName,
    o2::framework::select(input.c_str()),
    Outputs{OutputSpec{{"tracklabels"}, "MCH", "TRACKLABELS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackMCLabelFinderTask>()},
    Options{{"incontext", VariantType::String, "collisioncontext.root", {"Take collision context from this file"}},
            {"sigma-cut", VariantType::Double, 4., {"Sigma cut to match simulated and reconstructed clusters"}}}};
}

} // end namespace mch
} // end namespace o2
