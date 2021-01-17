// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include <TH1F.h>

#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// Example task generating a multiplicity distribution of
// collision which pass the INT7 selection and
// tracks which pass the "isGlobalTrack" selection
//
// Needs to run with event and track selection:
// o2-analysis-timestamp | o2-analysis-event-selection | o2-analysis-trackextension | o2-analysis-trackselection | o2-analysistutorial-multiplicity-event-track-selection

struct MultiplicityEventTrackSelection {

  OutputObj<TH1F> multiplicity{TH1F("multiplicity", "multiplicity", 5000, -0.5, 4999.5)};

  Filter collisionZFilter = nabs(aod::collision::posZ) < 10.0f;
  Filter trackFilter = (nabs(aod::track::eta) < 0.8f) && (aod::track::pt > 0.15f) && (aod::track::isGlobalTrack == (uint8_t) true);

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision,
               soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>> const& tracks)
  {
    if (!collision.alias()[kINT7]) {
      return;
    }
    if (!collision.sel7()) {
      return;
    }

    LOGP(INFO, "Collision with {} tracks", tracks.size());
    multiplicity->Fill(tracks.size());
  }
};

// Workflow definition
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<MultiplicityEventTrackSelection>("multiplicity-event-track-selection")};
}
