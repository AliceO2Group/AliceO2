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

// This task shows how to access the ZDC and FV0A information which belongs to a collision
// The association is made through the BC column (and in Run 3 may not be unique!)

// This example access the collisions and the related FV0A information.
// Note that one has to subscribe to aod::Collisions const& and aod::FV0As const& to load
// the relevant data even if you access the data itself through m.collision() and m.fv0a()
// Here the "sparse" matcher is used which means, there can be collisions without FV0A information
// To find out, m.has_fv0a() has to be called. Otherwise m.fv0a() will fail.
struct IterateV0 {
  void process(aod::Run2MatchedSparse::iterator const& m, aod::Collisions const&, aod::FV0As const&)
  {
    LOGF(INFO, "Vertex = %f", m.collision().posZ());
    if (m.has_fv0a()) {
      auto v0a = m.fv0a();
      LOGF(info, "V0A: %f %f", v0a.amplitude()[0], v0a.amplitude()[1]);
    } else {
      LOGF(INFO, "No V0A info");
    }
  }
};

// This example is identical to IterateV0, but uses the exclusive match. This means that lines where any
// of the tables asked for in Run2MatchedExclusive (see AnalysisDataModel.h) are missing are not listed.
// Only to be used if one is sure that all your events have the desired information
struct IterateV0Exclusive {
  void process(aod::Run2MatchedExclusive::iterator const& m, aod::Collisions const&, aod::FV0As const&)
  {
    LOGF(INFO, "Vertex = %f", m.collision().posZ());
    auto v0a = m.fv0a();
    LOGF(info, "V0: %f %f", v0a.amplitude()[0], v0a.amplitude()[1]);
  }
};

// This example builds on IterateV0 and in addition accesses also the tracks grouped to the specific collision.
// The tracks are directly access through its pointer.
struct IterateV0Tracks {
  void process(aod::Run2MatchedSparse::iterator const& m, aod::Collisions const&, aod::FV0As const&, aod::Tracks const& tracks)
  {
    LOGF(INFO, "Vertex = %f. %d tracks", m.collision().posZ(), tracks.size());
    if (m.has_fv0a()) {
      auto v0a = m.fv0a();
      LOGF(info, "V0A: %f %f", v0a.amplitude()[0], v0a.amplitude()[1]);
    } else {
      LOGF(INFO, "No V0A info");
    }
  }
};

// IterateV0Tracks with join. Desired version for good readability
// using CollisionMatchedRun2Sparse = soa::Join<aod::Run2MatchedSparse, aod::Collisions>::iterator;
// struct IterateV0Tracks2 {
//   void process(CollisionMatchedRun2Sparse const& m, aod::FV0As const&, aod::Tracks const& tracks)
//   {
//     LOGF(INFO, "Vertex = %f. %d tracks", m.posZ(), tracks.size());
//     LOGF(INFO, "Vertex = %f. %d tracks", m.collision().posZ(), tracks.size());
//     if (m.has_fv0a()) {
//       auto v0a = m.fv0a();
//       LOGF(info, "V0A: %f %f", v0a.amplitude()[0], v0a.amplitude()[1]);
//     } else {
//       LOGF(INFO, "No V0A info");
//     }
//   }
// };

// This example accesses V0 and ZDC information
struct IterateV0ZDC {
  void process(aod::Run2MatchedSparse::iterator const& m, aod::Collisions const&, aod::FV0As const&, aod::Zdcs const&)
  {
    LOGF(INFO, "Vertex = %f", m.collision().posZ());
    if (m.has_fv0a()) {
      auto v0a = m.fv0a();
      LOGF(info, "V0A: %f %f", v0a.amplitude()[0], v0a.amplitude()[1]);
    } else {
      LOGF(INFO, "No V0A info");
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
