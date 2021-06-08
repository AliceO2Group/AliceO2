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
/// \brief These example tasks show how to access the ZDC and FV0A information which belongs to a collision.
///        The association is made through the BC column (and in Run 3 may not be unique!)
///        This example accesses the collisions and the related FV0A information.
///        Note that one has to subscribe to aod::FV0As const& to load
///        the relevant data even if you access the data itself through collisionMatched.fv0a().
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// Here the "sparse" matcher is used which means, there can be collisions without FV0A information
// To find out, collisionMatched.has_fv0a() has to be called. Otherwise collisionMatched.fv0a() will fail.
// NOTE: subscribing to Collisions separately will lead to a circular dependency due to forwarding
struct IterateV0 {
  void process(aod::CollisionMatchedRun2Sparse const& collisionMatched, aod::FV0As const&)
  {
    LOGF(INFO, "Vertex = %f", collisionMatched.posZ());
    if (collisionMatched.has_fv0a()) {
      auto v0a = collisionMatched.fv0a();
      LOGF(info, "V0A: %f %f", v0a.amplitude()[0], v0a.amplitude()[1]);
    } else {
      LOGF(INFO, "No V0A info");
    }
  }
};

// This example is identical to IterateV0, but uses the exclusive match. This means that collisions where any
// of the tables asked for in Run2MatchedExclusive (see AnalysisDataModel.h) are missing are not there.
// Therefore, the syntax is more complicated because we cannot join against Collision
// (the tables have different number of entries)
// Only to be used if one is sure that all your events have the desired information
struct IterateV0Exclusive {
  void process(aod::Run2MatchedExclusive::iterator const& matcher, aod::Collisions const&, aod::FV0As const&)
  {
    LOGF(INFO, "Vertex = %f", matcher.collision().posZ());
    auto fv0a = matcher.fv0a();
    LOGF(info, "V0: %f %f", fv0a.amplitude()[0], fv0a.amplitude()[1]);
  }
};

// This example builds on IterateV0 and in addition accesses also the tracks grouped to the specific collision.
// The tracks are directly accessed through its pointer as usual
struct IterateV0Tracks {
  void process(aod::CollisionMatchedRun2Sparse const& collisionMatched, aod::FV0As const&, aod::Tracks const& tracks)
  {
    LOGF(INFO, "Vertex = %f. %d tracks", collisionMatched.posZ(), tracks.size());
    if (collisionMatched.has_fv0a()) {
      auto v0a = collisionMatched.fv0a();
      LOGF(info, "V0A: %f %f", v0a.amplitude()[0], v0a.amplitude()[1]);
    } else {
      LOGF(INFO, "No V0A info");
    }
  }
};

// This example accesses V0 and ZDC information
struct IterateV0ZDC {
  void process(aod::CollisionMatchedRun2Sparse const& collisionMatched, aod::FV0As const&, aod::Zdcs const&)
  {
    LOGF(INFO, "Vertex = %f", collisionMatched.posZ());
    if (collisionMatched.has_fv0a()) {
      auto v0a = collisionMatched.fv0a();
      LOGF(info, "V0A: %f %f", v0a.amplitude()[0], v0a.amplitude()[1]);
    } else {
      LOGF(INFO, "No V0A info");
    }
    if (collisionMatched.has_zdc()) {
      LOGF(INFO, "ZDC: E1 = %.3f; E2 = %.3f", collisionMatched.zdc().energyZEM1(), collisionMatched.zdc().energyZEM2());
    } else {
      LOGF(INFO, "No ZDC info");
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<IterateV0>(cfgc),
    adaptAnalysisTask<IterateV0Exclusive>(cfgc),
    adaptAnalysisTask<IterateV0Tracks>(cfgc),
    adaptAnalysisTask<IterateV0ZDC>(cfgc),
  };
}
