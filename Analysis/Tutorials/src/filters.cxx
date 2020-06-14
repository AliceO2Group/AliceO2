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
struct ATask {
  Produces<aod::EtaPhi> etaphi;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      etaphi(track.eta(), track.phi());
    }
  }
};

struct BTask {
  float fPI = static_cast<float>(M_PI);
  float ptlow = 0.5f;
  float ptup = 2.0f;
  Filter ptFilter_a = aod::track::pt2 > (ptlow * ptlow);
  Filter ptFilter_b = aod::track::pt2 < (ptup * ptup);

  //  Filter ptFilter = ((aod::track::signed1Pt < 1.0f / ptlow) && (aod::track::signed1Pt > 1.0f / ptup)) || ((aod::track::signed1Pt < -1.0f / ptup) && (aod::track::signed1Pt < -1.0f / ptlow));
  float etalow = -1.0f;
  float etaup = 1.0f;
  Filter etafilter = (aod::etaphi::eta2 < etaup) && (aod::etaphi::eta2 > etalow);

  //  Filter etaFilter = (aod::track::tgl < tan(0.5f * fPI - 2.0f * atan(exp(etalow)))) && (aod::track::tgl > tan(0.5f * fPI - 2.0f * atan(exp(etaup))));
  float philow = 1.0f;
  float phiup = 2.0f;
  Filter phifilter = (aod::etaphi::phi2 < phiup) && (aod::etaphi::phi2 > philow);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::Tracks, aod::EtaPhi>> const& tracks)
  {
    LOGF(INFO, "Collision: %d [N = %d]", collision.globalIndex(), tracks.size());
    for (auto& track : tracks) {
      LOGF(INFO, "id = %d; eta:  %.3f < %.3f < %.3f; phi: %.3f < %.3f < %.3f; pt: %.3f < %.3f < %.3f", track.collisionId(), etalow, track.eta2(), etaup, philow, track.phi2(), phiup, ptlow, track.pt(), ptup);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-etaphi"),
    adaptAnalysisTask<BTask>("consume-etaphi")};
}
