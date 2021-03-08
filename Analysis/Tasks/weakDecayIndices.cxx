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

// Tasks to build transient indices to group V0s and cascades to collisions

struct IndexV0s {
  Produces<aod::TransientV0s> transientV0s;

  void process(aod::StoredV0s const& v0s, aod::Tracks const& tracks)
  {
    for (auto& v0 : v0s) {
      if (v0.posTrack().collisionId() != v0.negTrack().collisionId()) {
        LOGF(WARNING, "V0 %d has inconsistent collision information (%d, %d)", v0.globalIndex(), v0.posTrack().collisionId(), v0.negTrack().collisionId());
      }
      transientV0s(v0.posTrack().collisionId());
    }
  }
};

// NOTE These tasks have to be split because for the cascades, V0s and not StoredV0s are needed
struct IndexCascades {
  Produces<aod::TransientCascades> transientCascades;

  void process(aod::V0s const& v0s, aod::StoredCascades const& cascades, aod::Tracks const& tracks)
  {
    for (auto& cascade : cascades) {
      if (cascade.bachelor().collisionId() != cascade.v0().posTrack().collisionId() || cascade.v0().posTrack().collisionId() != cascade.v0().negTrack().collisionId()) {
        LOGF(WARNING, "Cascade %d has inconsistent collision information (%d, %d, %d)", cascade.globalIndex(), cascade.bachelor().collisionId(), cascade.v0().posTrack().collisionId(), cascade.v0().negTrack().collisionId());
      }
      transientCascades(cascade.bachelor().collisionId());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<IndexV0s>(cfgc, "weak-decay-indices-v0"),
    adaptAnalysisTask<IndexCascades>(cfgc, "weak-decay-indices-cascades"),
  };
}
