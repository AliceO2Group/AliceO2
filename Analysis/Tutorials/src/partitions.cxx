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

namespace o2::aod
{
namespace etaphi
{
DECLARE_SOA_COLUMN(Eta, eta2, float);
DECLARE_SOA_COLUMN(Phi, phi2, float);
} // namespace etaphi
DECLARE_SOA_TABLE(EtaPhi, "AOD", "ETAPHI",
                  etaphi::Eta, etaphi::Phi);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// This is a very simple example showing how to iterate over tracks
// and create a new collection for them.
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct TTask {
  Produces<aod::EtaPhi> etaphi;

  // FIXME: For some reason filtering with Charge does not work??
  //Partition<aod::Tracks> negativeTracksP = aod::track::Charge < 0;
  //Partition<aod::Tracks> positiveTracksP = aod::track::Charge > 0;
  Partition<aod::Tracks> negativeTracksP = aod::track::pt2 < 1.0f;
  Partition<aod::Tracks> positiveTracksP = aod::track::pt2 > 1.0f;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      etaphi(track.eta(), track.phi());
    }

    auto& negativeTracks = *(negativeTracksP.mFiltered);
    auto& positiveTracks = *(positiveTracksP.mFiltered);
    LOGF(INFO, "[negative tracks: %d] [positive tracks: %d]", negativeTracks.size(), positiveTracks.size());
    for (auto& track : negativeTracks) {
      LOGF(INFO, "negative track id: %d pt: %.3f < 1.0", track.collisionId(), track.pt2());
    }
    for (auto& track : positiveTracks) {
      LOGF(INFO, "positive track id: %d pt: %.3f > 1.0", track.collisionId(), track.pt2());
    }
  }
};

struct ETask {
  float fPI = static_cast<float>(M_PI);
  float ptlow = 0.5f;
  float ptup = 2.0f;
  float etalim = 0.0f;
  float philow = 1.0f;
  float phiup = 2.0f;
  Partition<soa::Join<aod::Tracks, aod::EtaPhi>> negEtaLeftPhiP =
    aod::etaphi::eta2 < etalim && aod::etaphi::phi2 < philow &&
    aod::track::pt2 > (ptlow * ptlow) && aod::track::pt2 < (ptup * ptup);
  Partition<soa::Join<aod::Tracks, aod::EtaPhi>> negEtaMidPhiP =
    aod::etaphi::eta2 < etalim && aod::etaphi::phi2 >= philow && aod::etaphi::phi2 < phiup &&
    aod::track::pt2 > (ptlow * ptlow) && aod::track::pt2 < (ptup * ptup);
  Partition<soa::Join<aod::Tracks, aod::EtaPhi>> negEtaRightPhiP =
    aod::etaphi::eta2 < etalim && aod::etaphi::phi2 >= phiup &&
    aod::track::pt2 > (ptlow * ptlow) && aod::track::pt2 < (ptup * ptup);

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::EtaPhi> const& tracks)
  {
    auto& leftPhi = *(negEtaLeftPhiP.mFiltered);
    auto& midPhi = *(negEtaMidPhiP.mFiltered);
    auto& rightPhi = *(negEtaRightPhiP.mFiltered);
    LOGF(INFO, "Collision: %d [N = %d] [left phis = %d] [mid phis = %d] [right phis = %d]",
         collision.globalIndex(), tracks.size(), leftPhi.size(), midPhi.size(), rightPhi.size());

    for (auto& track : tracks) {
      LOGF(INFO, "id = %d; pt: %.3f < %.3f < %.3f", track.collisionId(), ptlow, track.pt(), ptup);
    }

    for (auto& track : leftPhi) {
      LOGF(INFO, "id = %d; eta: %.3f < %.3f; phi: %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.eta2(), etalim, track.phi2(), philow, ptlow, track.pt(), ptup);
    }
    for (auto& track : midPhi) {
      LOGF(INFO, "id = %d; eta: %.3f < %.3f; phi: %.3f <= %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.eta2(), etalim, philow, track.phi2(), phiup, ptlow, track.pt(), ptup);
    }
    for (auto& track : rightPhi) {
      LOGF(INFO, "id = %d; eta: %.3f < %.3f; phi: %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), track.eta2(), etalim, phiup, track.phi2(), ptlow, track.pt(), ptup);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TTask>("produce-etaphi"),
    adaptAnalysisTask<ETask>("consume-etaphi")};
}
