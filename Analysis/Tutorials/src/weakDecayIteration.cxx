// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief Example how to enumerate V0s and cascades. Note ...
///        V0s = Join<TransientV0s, StoredV0s>
///        Cascades = Join<TransientCascades, StoredCascades>
///        TransientV0 and TransientCascades are filled by the helper task weak-decay-indices. Hence use ...
///        o2-analysis-weak-decay-indices --aod-file AO2D.root | o2-analysistutorial-weak-decay-iteration
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

using namespace o2;
using namespace o2::framework;

struct LoopV0s {
  void process(aod::V0s const& v0s, aod::Tracks const& tracks)
  {
    for (auto& v0 : v0s) {
      LOGF(DEBUG, "V0 (%d, %d, %d)", v0.posTrack().collisionId(), v0.negTrack().collisionId(), v0.collisionId());
    }
  }
};

struct LoopCascades {
  void process(aod::Cascades const& cascades, aod::V0s const& v0s, aod::Tracks const& tracks)
  {
    for (auto& cascade : cascades) {
      LOGF(DEBUG, "Cascade %d (%d, %d, %d, %d)", cascade.globalIndex(), cascade.bachelor().collisionId(), cascade.v0().posTrack().collisionId(), cascade.v0().negTrack().collisionId(), cascade.collisionId());
    }
  }
};

// Grouping V0s
struct GroupV0s {
  void process(aod::Collision const& collision, aod::V0s const& v0s, aod::Tracks const& tracks)
  {
    LOGF(INFO, "Collision %d has %d V0s", collision.globalIndex(), v0s.size());

    for (auto& v0 : v0s) {
      LOGF(DEBUG, "Collision %d V0 %d (%d, %d)", collision.globalIndex(), v0.globalIndex(), v0.posTrackId(), v0.negTrackId());
    }
  }
};

// Grouping V0s and cascades
// NOTE that you need to subscribe to V0s even if you only process cascades
struct GroupV0sCascades {
  void process(aod::Collision const& collision, aod::V0s const& v0s, aod::Cascades const& cascades, aod::Tracks const& tracks)
  {
    LOGF(INFO, "Collision %d has %d cascades (%d tracks)", collision.globalIndex(), cascades.size(), tracks.size());

    for (auto& cascade : cascades) {
      LOGF(INFO, "Collision %d Cascade %d (%d, %d, %d)", collision.globalIndex(), cascade.globalIndex(), cascade.v0().posTrackId(), cascade.v0().negTrackId(), cascade.bachelorId());
      LOGF(INFO, "             IDs: %d %d %d", cascade.v0().posTrack().collisionId(), cascade.v0().negTrack().collisionId(), cascade.bachelor().collisionId());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<LoopV0s>(cfgc),
    adaptAnalysisTask<LoopCascades>(cfgc),
    adaptAnalysisTask<GroupV0s>(cfgc),
    adaptAnalysisTask<GroupV0sCascades>(cfgc),
  };
}
