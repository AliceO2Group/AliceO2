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

// This task shows how to access the ZDC and the Run2 V0 information which belongs to a collision
// The association is made through the BC column (and in Run 3 may not be unique!)
// To run this workflow, the o2-analysis-run2-matcher has to be run as well.
// Example: o2-analysis-run2-matcher --aod-file AO2D.root | o2-analysistutorial-zdc-vzero-iteration

// This example access the collisions and the related Run2V0 information.
// Note that one has to subscribe to aod::Collisions const& and aod::Run2V0s const& to load
// the relevant data even if you access the data itself through m.collision() and m.run2v0()
// Here the "sparse" matcher is used which means, there can be collisions without Run2V0 information
// To find out, m.has_run2v0() has to be called. Otherwise m.run2v0() will fail.
struct IterateV0 {
  void process(aod::Run2MatchedSparse::iterator const& m, aod::Collisions const&, aod::Run2V0s const&)
  {
    LOGF(INFO, "Vertex = %f", m.collision().posZ());
    if (m.has_run2v0()) {
      auto v0 = m.run2v0();
      LOGF(info, "V0: %f %f", v0.adc()[0], v0.adc()[1]);
    } else {
      LOGF(INFO, "No V0 info");
    }
  }
};

// This example is identical to IterateV0, but uses the exclusive match. This means that lines where any
// of the tables asked for in Run2MatchedExclusive (see AnalysisDataModel.h) are missing are not listed.
// Only to be used if one is sure that all your events have the desired information
struct IterateV0Exclusive {
  void process(aod::Run2MatchedExclusive::iterator const& m, aod::Collisions const&, aod::Run2V0s const&)
  {
    LOGF(INFO, "Vertex = %f", m.collision().posZ());
    auto v0 = m.run2v0();
    LOGF(info, "V0: %f %f", v0.adc()[0], v0.adc()[1]);
  }
};

// This example builds on IterateV0 and in addition accesses also the tracks grouped to the specific collision.
// The tracks are directly access through its pointer.
struct IterateV0Tracks {
  void process(aod::Run2MatchedSparse::iterator const& m, aod::Collisions const&, aod::Run2V0s const&, aod::Tracks const& tracks)
  {
    LOGF(INFO, "Vertex = %f. %d tracks", m.collision().posZ(), tracks.size());
    if (m.has_run2v0()) {
      auto v0 = m.run2v0();
      LOGF(info, "V0: %f %f", v0.adc()[0], v0.adc()[1]);
    } else {
      LOGF(INFO, "No V0 info");
    }
  }
};

// IterateV0Tracks with join. Desired version for good readability
// using CollisionMatchedRun2Sparse = soa::Join<aod::Run2MatchedSparse, aod::Collisions>::iterator;
// struct IterateV0Tracks2 {
//   void process(CollisionMatchedRun2Sparse const& m, aod::Run2V0s const&, aod::Tracks const& tracks)
//   {
//     LOGF(INFO, "Vertex = %f. %d tracks", m.posZ(), tracks.size());
//     LOGF(INFO, "Vertex = %f. %d tracks", m.collision().posZ(), tracks.size());
//     if (m.has_run2v0()) {
//       auto v0 = m.run2v0();
//       LOGF(info, "V0: %f %f", v0.adc()[0], v0.adc()[1]);
//     } else {
//       LOGF(INFO, "No V0 info");
//     }
//   }
// };

// This example accesses V0 and ZDC information
struct IterateV0ZDC {
  void process(aod::Run2MatchedSparse::iterator const& m, aod::Collisions const&, aod::Run2V0s const&, aod::Zdcs const&)
  {
    LOGF(INFO, "Vertex = %f", m.collision().posZ());
    if (m.has_run2v0()) {
      auto v0 = m.run2v0();
      LOGF(info, "V0: %f %f", v0.adc()[0], v0.adc()[1]);
    } else {
      LOGF(INFO, "No V0 info");
    }
    if (m.has_zdc()) {
      LOGF(INFO, "ZDC: E1 = %.3f; E2 = %.3f", m.zdc().energyZEM1(), m.zdc().energyZEM2());
    } else {
      LOGF(INFO, "No ZDC info");
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<IterateV0>("iterate-v0"),
    adaptAnalysisTask<IterateV0Exclusive>("iterate-v0-exclusive"),
    adaptAnalysisTask<IterateV0Tracks>("iterate-v0-tracks"),
    //     adaptAnalysisTask<IterateV0Tracks2>("iterate-v0-tracks2"),
    adaptAnalysisTask<IterateV0ZDC>("iterate-v0-zdc"),
  };
}
