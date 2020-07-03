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

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// This is a very simple example showing how to iterate over tracks
// and create a new collection for them.
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct TTask {
  // FIXME: For some reason filtering with Charge does not work??
  //Partition<aod::Tracks> negativeTracksP = aod::track::Charge < 0;
  //Partition<aod::Tracks> positiveTracksP = aod::track::Charge > 0;
  Partition<aod::Tracks> negativeTracksP = aod::track::pt2 < 1.0f;
  Partition<aod::Tracks> positiveTracksP = aod::track::pt2 > 1.0f;

  void process(aod::Tracks const& tracks)
  {
    auto& negativeTracks = *(negativeTracksP.mFiltered);
    auto& positiveTracks = *(positiveTracksP.mFiltered);
    LOGF(INFO, "[negative tracks: %d] [positive tracks: %d]", negativeTracks.size(), positiveTracks.size());
    for (auto& track : negativeTracks) {
      LOGF(INFO, "negative track id: %d pt: %.3f < 1.0", track.collisionId(), track.pt2());
    }
    for (auto& track : positiveTracks) {
      LOGF(INFO, "positive track id: %d pt: %.3f > 1.0", track.collisionId(), track.pt2());
    }
  }
};

struct CTask {
  // FIXME: For some reason filtering with Charge does not work??
  //Partition<aod::Tracks> negativeTracksP = aod::track::Charge < 0;
  //Partition<aod::Tracks> positiveTracksP = aod::track::Charge > 0;
  Partition<aod::Tracks> negativeTracksP = aod::track::pt2 < 1.0f;
  Partition<aod::Tracks> positiveTracksP = aod::track::pt2 > 1.0f;

  void process(aod::Collision const& collision, aod::Tracks const& tracks)
  {
    auto& negativeTracks = *(negativeTracksP.mFiltered);
    auto& positiveTracks = *(positiveTracksP.mFiltered);
    LOGF(INFO, "[negative tracks: %d] [positive tracks: %d]", negativeTracks.size(), positiveTracks.size());
    for (auto& track : negativeTracks) {
      LOGF(INFO, "negative track id: %d pt: %.3f < 1.0", track.collisionId(), track.pt2());
    }
    for (auto& track : positiveTracks) {
      LOGF(INFO, "positive track id: %d pt: %.3f > 1.0", track.collisionId(), track.pt2());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TTask>("consume-tracks")};
    //adaptAnalysisTask<CTask>("consume-tracks-with-col")};
}
