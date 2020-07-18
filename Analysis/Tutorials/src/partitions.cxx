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
struct ATask {
  float fPI = static_cast<float>(M_PI);
  float ptlow = 0.5f;
  float ptup = 2.0f;
  Filter ptFilter_a = aod::track::pt > ptlow;
  Filter ptFilter_b = aod::track::pt < ptup;

  float etalow = -1.0f;
  float etaup = 1.0f;
  Filter etafilter = (aod::track::eta < etaup) && (aod::track::eta > etalow);

  float philow = 1.0f;
  float phiup = 2.0f;
  Partition<soa::Filtered<aod::Tracks>> leftPhi = aod::track::phiraw < philow;
  Partition<soa::Filtered<aod::Tracks>> midPhi = aod::track::phiraw >= philow && aod::track::phiraw < phiup;
  Partition<soa::Filtered<aod::Tracks>> rightPhi = aod::track::phiraw >= phiup;

  void process(aod::Collision const& collision, soa::Filtered<aod::Tracks> const& tracks)
  {
    LOGF(INFO, "Collision: %d [N = %d] [left phis = %d] [mid phis = %d] [right phis = %d]",
         collision.globalIndex(), tracks.size(), leftPhi.size(), midPhi.size(), rightPhi.size());

    for (auto& track : leftPhi) {
      LOGF(INFO, "id = %d, from collision: %d, collision: %d; eta:  %.3f < %.3f < %.3f; phi: %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.collision().globalIndex(), collision.globalIndex(), etalow, track.eta(), etaup, track.phiraw(), philow, ptlow, track.pt(), ptup);
    }
    for (auto& track : midPhi) {
      LOGF(INFO, "id = %d, from collision: %d, collision: %d; eta: %.3f < %.3f < %.3f; phi: %.3f <= %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.collision().globalIndex(), collision.globalIndex(), etalow, track.eta(), etaup, philow, track.phiraw(), phiup, ptlow, track.pt(), ptup);
    }
    for (auto& track : rightPhi) {
      LOGF(INFO, "id = %d, from collision: %d, collision: %d; eta: %.3f < %.3f < %.3f; phi: %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.collision().globalIndex(), collision.globalIndex(), etalow, track.eta(), etaup, phiup, track.phiraw(), ptlow, track.pt(), ptup);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("consume-tracks")};
}
