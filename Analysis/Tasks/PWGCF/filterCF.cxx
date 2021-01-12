// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"

#include "AnalysisDataModel/CFDerived.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/Centrality.h"

#include <TH3F.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

#define O2_DEFINE_CONFIGURABLE(NAME, TYPE, DEFAULT, HELP) Configurable<TYPE> NAME{#NAME, DEFAULT, HELP};

struct FilterCF {

  // Configuration
  O2_DEFINE_CONFIGURABLE(cfgCutVertex, float, 7.0f, "Accepted z-vertex range")
  O2_DEFINE_CONFIGURABLE(cfgCutPt, float, 0.5f, "Minimal pT for tracks")
  O2_DEFINE_CONFIGURABLE(cfgCutEta, float, 0.8f, "Eta range for tracks")

  // Filters and input definitions
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex && aod::cent::centV0M <= 80.0f;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::pt > cfgCutPt) && ((aod::track::isGlobalTrack == (uint8_t) true) || (aod::track::isGlobalTrackSDD == (uint8_t) true));

  OutputObj<TH3F> yields{TH3F("yields", "centrality vs pT vs eta", 100, 0, 100, 40, 0, 20, 100, -2, 2)};
  OutputObj<TH3F> etaphi{TH3F("etaphi", "centrality vs eta vs phi", 100, 0, 100, 100, -2, 2, 200, 0, 2 * M_PI)};

  Produces<aod::CFCollisions> outputCollisions;
  Produces<aod::CFTracks> outputTracks;

  void init(o2::framework::InitContext&)
  {
  }

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents>>::iterator const& collision, aod::BCs const&, soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>> const& tracks)
  {
    LOGF(info, "Tracks for collision: %d | Vertex: %.1f | INT7: %d | V0M: %.1f", tracks.size(), collision.posZ(), collision.sel7(), collision.centV0M());

    if (!collision.alias()[kINT7]) {
      return;
    }
    if (!collision.sel7()) {
      return;
    }

    outputCollisions(collision.bc().runNumber(), collision.posZ(), collision.centV0M());

    for (auto& track : tracks) {
      uint8_t trackType = 0;
      if (track.isGlobalTrack()) {
        trackType = 1;
      } else if (track.isGlobalTrackSDD()) {
        trackType = 2;
      }

      outputTracks(outputCollisions.lastIndex(), track.pt(), track.eta(), track.phi(), track.charge(), trackType);

      yields->Fill(collision.centV0M(), track.pt(), track.eta());
      etaphi->Fill(collision.centV0M(), track.eta(), track.phi());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<FilterCF>("filter-cf")};
}
