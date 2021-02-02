// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFTrackQualitySelector.cxx
/// \brief Tagging of track quality for HF studies
///
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
//#include "AnalysisDataModel/Centrality.h"
#include "Framework/HistogramRegistry.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

/// Track selection
struct HFTrackQualitySelector {
  Produces<aod::HFTrackQuality> trackQualityTable;

  Configurable<bool> b_doPlots{"b_doPlots", true, "fill histograms"};
  Configurable<int> d_tpcnclsfound{"d_tpcnclsfound", 70, ">= min. number of TPC clusters needed"};

  HistogramRegistry registry{
    "registry",
    {{"hpt_nocuts", "all tracks;#it{p}_{T}^{track} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hpt_cuts", "selected tracks;#it{p}_{T}^{track} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}}}};

  void process(aod::Collision const& collision,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::TrackSelection> const& tracks)
  {

    for (auto& track : tracks) {

      int trackQualityStatus = 0;
      if (b_doPlots) {
        registry.get<TH1>(HIST("hpt_nocuts"))->Fill(track.pt());
      }
      UChar_t clustermap = track.itsClusterMap();
      if (track.isGlobalTrack() && track.tpcNClsFound() >= d_tpcnclsfound &&
          track.flags() & o2::aod::track::ITSrefit &&
          (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1))) {
        trackQualityStatus = 1;
        if (b_doPlots) {
          registry.get<TH1>(HIST("hpt_cuts"))->Fill(track.pt());
        }
      }

      // fill table row
      trackQualityTable(trackQualityStatus);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFTrackQualitySelector>("hf-track-quality-selector")};
}
