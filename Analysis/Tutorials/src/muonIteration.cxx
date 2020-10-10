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

// This task shows how to access the Muons belong to a collision
// The association is made through the BC column (and in Run 3 may not be unique!)
// To run this workflow, the o2-analysis-run(2,3)-matcher has to be run as well.
// Example: o2-analysis-run2-matcher --aod-file AO2D.root | o2-analysistutorial-muon-iteration
//
// Note that one has to subscribe to aod::Collisions const& to load
// the relevant data even if you access the data itself through m.collision()
// This uses the exclusive matcher, so you only get BCs which have a collision
// If you want also BCs without collision, you should use BCCollisionsSparse and check for m.has_collision()
struct IterateMuons {
  void process(aod::BCCollisionsExclusive::iterator const& m, aod::Collisions const&, aod::Muons const& muons)
  {
    LOGF(INFO, "Vertex = %f has %d muons", m.collision().posZ(), muons.size());
    for (auto& muon : muons) {
      LOGF(info, "  pT = %.2f", muon.pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<IterateMuons>("iterate-muons"),
  };
}
