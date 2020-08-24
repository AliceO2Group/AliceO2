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

struct ATask {
  Produces<aod::TransientV0s> transientV0s;
  Produces<aod::TransientCascades> transientCascades;

  void process(aod::StoredV0s const& v0s, aod::StoredCascades const& cascades, aod::FullTracks const& tracks)
  {
    for (auto& v0 : v0s) {
      transientV0s(v0.posTrack().collisionId());
    }
    for (auto& cascade : cascades) {
      transientCascades(cascade.bachelor().collisionId());
    }
  }
};

struct BTask {
  void process(aod::V0s const& v0s, aod::FullTracks const& tracks)
  {
    for (auto& v0 : v0s) {
      LOGF(DEBUG, "V0 (%d, %d, %d)", v0.posTrack().collisionId(), v0.negTrack().collisionId(), v0.collisionId());
    }
  }
};

struct CTask {
  void process(aod::Cascades const& cascades, aod::FullTracks const& tracks)
  {
    for (auto& cascade : cascades) {
      LOGF(DEBUG, "Cascade (%d, %d)", cascade.bachelor().collisionId(), cascade.collisionId());
    }
  }
};

// Grouping V0s
struct DTask {
  void process(aod::Collision const& collision, aod::V0s const& v0s, aod::FullTracks const& tracks)
  {
    LOGF(INFO, "Collision %d has %d V0s", collision.globalIndex(), v0s.size());

    for (auto& v0 : v0s) {
      LOGF(DEBUG, "Collision %d V0 %d (%d, %d)", collision.globalIndex(), v0.globalIndex(), v0.posTrackId(), v0.negTrackId());
    }
  }
};

// Grouping V0s and cascades
// NOTE that you need to subscribe to V0s even if you only process cascades
struct ETask {
  void process(aod::Collision const& collision, aod::V0s const& v0s, aod::Cascades const& cascades, aod::FullTracks const& tracks)
  {
    LOGF(INFO, "Collision %d has %d cascades", collision.globalIndex(), cascades.size());

    for (auto& cascade : cascades) {
      LOGF(INFO, "Collision %d Cascade %d (%d, %d, %d)", collision.globalIndex(), cascade.globalIndex(), cascade.v0().posTrackId(), cascade.v0().negTrackId(), cascade.bachelorId());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-v0index"),
    adaptAnalysisTask<BTask>("consume-v0s"),
    adaptAnalysisTask<CTask>("consume-cascades"),
    adaptAnalysisTask<DTask>("consume-grouped-v0s"),
    adaptAnalysisTask<ETask>("consume-grouped-cascades"),
  };
}
