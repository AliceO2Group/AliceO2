// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
/// \brief Shows how to loop over collisions and tracks of a data frame.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

using namespace o2;
using namespace o2::framework;

struct ATask {
  void process(aod::Collision const& collision, aod::Tracks const& tracks)
  {
    // `tracks` contains tracks belonging to `collision`
    LOGF(info, "Collision index : %d", collision.index());
    LOGF(info, "Number of tracks: %d", tracks.size());

    // process the tracks of a given collision
    for (auto& track : tracks) {
      LOGF(info, "  track pT = %f GeV/c", track.pt());
    }
  }
};

struct BTask {

  void process(aod::Collisions const& collisions, aod::Tracks const& tracks)
  {
    // `tracks` contains all tracks of a data frame
    LOGF(info, "Number of collisions: %d", collisions.size());
    LOGF(info, "Number of tracks    : %d", tracks.size());
  }
};

struct CTask {

  void process(aod::Collision const& collision, aod::Tracks const& tracks, aod::V0s const& v0s)
  {
    // `tracks` contains tracks belonging to `collision`
    // `v0s`    contains V0s    belonging to `collision`
    LOGF(info, "Collision index : %d", collision.index());
    LOGF(info, "Number of tracks: %d", tracks.size());
    LOGF(info, "Number of v0s   : %d", v0s.size());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>(cfgc, TaskName{"collision-tracks-iteration-tutorial_A"}),
    adaptAnalysisTask<BTask>(cfgc, TaskName{"collision-tracks-iteration-tutorial_B"}),
    adaptAnalysisTask<CTask>(cfgc, TaskName{"collision-tracks-iteration-tutorial_C"}),
  };
}
