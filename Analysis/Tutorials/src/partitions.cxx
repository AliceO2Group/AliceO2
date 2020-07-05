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
struct ETask {
  float fPI = static_cast<float>(M_PI);
  float ptlow = 0.5f;
  float ptup = 2.0f;
  float etalim = 0.0f;
  float philow = 1.0f;
  float phiup = 2.0f;
  Partition<aod::Tracks> negEtaLeftPhiP =
    aod::track::eta < etalim && aod::track::phiraw < philow &&
    aod::track::pt > ptlow && aod::track::pt < ptup;
  Partition<aod::Tracks> negEtaMidPhiP =
    aod::track::eta < etalim && aod::track::phiraw >= philow && aod::track::phiraw < phiup &&
    aod::track::pt > ptlow && aod::track::pt < ptup;
  Partition<aod::Tracks> negEtaRightPhiP =
    aod::track::eta < etalim && aod::track::phiraw >= phiup &&
    aod::track::pt > ptlow && aod::track::pt < ptup;

  void process(aod::Collision const& collision, aod::Tracks const& tracks)
  {
    auto& leftPhi = negEtaLeftPhiP.getPartition();
    auto& midPhi = negEtaMidPhiP.getPartition();
    auto& rightPhi = negEtaRightPhiP.getPartition();
    LOGF(INFO, "Collision: %d [N = %d] [left phis = %d] [mid phis = %d] [right phis = %d]",
         collision.globalIndex(), tracks.size(), leftPhi.size(), midPhi.size(), rightPhi.size());

    for (auto& track : tracks) {
      LOGF(INFO, "id = %d; pt: %.3f < %.3f < %.3f", track.collisionId(), ptlow, track.pt(), ptup);
    }

    for (auto& track : leftPhi) {
      LOGF(INFO, "id = %d; eta: %.3f < %.3f; phi: %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.eta(), etalim, track.phiraw(), philow, ptlow, track.pt(), ptup);
    }
    for (auto& track : midPhi) {
      LOGF(INFO, "id = %d; eta: %.3f < %.3f; phi: %.3f <= %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.eta(), etalim, philow, track.phiraw(), phiup, ptlow, track.pt(), ptup);
    }
    for (auto& track : rightPhi) {
      LOGF(INFO, "id = %d; eta: %.3f < %.3f; phi: %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.eta(), etalim, phiup, track.phiraw(), ptlow, track.pt(), ptup);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ETask>("consume-etaphi")};
}
