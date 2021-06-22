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
/// \brief This task shows how to access the Muons belong to a collision.
///        The association is made through the BC column (and in Run 3 may not be unique!).
///        Note that one has to subscribe to aod::Collisions const& to load
///        the relevant data even if you access the data itself through m.collision().
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// This uses the exclusive matcher, so you only get BCs which have a collision
// If you want also BCs without collision, see the example IterateMuonsSparse below
struct IterateMuonsExclusive {
  void process(aod::MatchedBCCollisionsExclusive::iterator const& m, aod::Collisions const&, aod::FwdTracks const& muons)
  {
    LOGF(INFO, "Vertex = %f has %d muons", m.collision().posZ(), muons.size());
    for (auto& muon : muons) {
      LOGF(info, "  pT = %.2f", muon.pt());
    }
  }
};

// Iterate on muon using the collision iterator in the dq-analysis style
struct IterateMuons {
  void process(aod::Collisions::iterator const& collision, aod::FwdTracks const& muons)
  {
    LOGF(INFO, "Vertex = %f has %d muons", collision.posZ(), muons.size());
    for (auto& muon : muons) {
      LOGF(info, "  pT = %.2f", muon.pt());
    }
  }
};

// This uses the sparse matcher, so you also get BCs without a collision.
// You need to check with m.has_collision()
struct IterateMuonsSparse {
  void process(aod::MatchedBCCollisionsSparse::iterator const& m, aod::Collisions const&, aod::FwdTracks const& muons)
  {
    if (m.has_collision()) {
      LOGF(INFO, "Vertex = %f has %d muons", m.collision().posZ(), muons.size());
    } else {
      LOGF(INFO, "BC without collision has %d muons", muons.size());
    }
    for (auto& muon : muons) {
      LOGF(info, "  pT = %.2f", muon.pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    //adaptAnalysisTask<IterateMuonsExclusives>(cfgc), // currently does not work
    adaptAnalysisTask<IterateMuons>(cfgc),
    adaptAnalysisTask<IterateMuonsSparse>(cfgc),
  };
}
