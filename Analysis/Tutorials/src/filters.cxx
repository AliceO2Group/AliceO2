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
DECLARE_SOA_COLUMN(NPhi, nphi, float);
DECLARE_SOA_EXPRESSION_COLUMN(CosPhi, cosphi, float, ncos(aod::etaphi::nphi));
} // namespace etaphi
namespace track
{
DECLARE_SOA_EXPRESSION_COLUMN(SPt, spt, float, nabs(aod::track::sigma1Pt / aod::track::signed1Pt));
}
DECLARE_SOA_TABLE(TPhi, "AOD", "TPHI",
                  etaphi::NPhi);
DECLARE_SOA_EXTENDED_TABLE_USER(EPhi, TPhi, "EPHI", aod::etaphi::CosPhi);
using etracks = soa::Join<aod::Tracks, aod::TracksCov>;
DECLARE_SOA_EXTENDED_TABLE_USER(MTracks, etracks, "MTRACK", aod::track::SPt);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// This is a very simple example showing how to iterate over tracks
// and create a new collection for them.
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  Produces<aod::TPhi> tphi;
  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      tphi(track.phi());
    }
  }
};

struct BTask {
  Spawns<aod::EPhi> ephi;
  Spawns<aod::MTracks> mtrk;

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
  Filter phifilter = (aod::etaphi::nphi < phiup) && (aod::etaphi::nphi > philow);

  Filter posZfilter = nabs(aod::collision::posZ) < 10.0f;

  void process(soa::Filtered<aod::Collisions>::iterator const& collision, soa::Filtered<soa::Join<aod::Tracks, aod::TPhi>> const& tracks)
  {
    LOGF(INFO, "Collision: %d [N = %d out of %d], -10 < %.3f < 10",
         collision.globalIndex(), tracks.size(), tracks.tableSize(), collision.posZ());
    for (auto& track : tracks) {
      LOGF(INFO, "id = %d; eta:  %.3f < %.3f < %.3f; phi: %.3f < %.3f < %.3f; pt: %.3f < %.3f < %.3f",
           track.collisionId(), etalow, track.eta(), etaup, philow, track.nphi(), phiup, ptlow, track.pt(), ptup);
    }
  }
};

struct CTask {
  void process(aod::Collision const&, soa::Join<aod::Tracks, aod::EPhi> const& tracks)
  {
    for (auto& track : tracks) {
      LOGF(INFO, "%.3f == %.3f", track.cosphi(), std::cos(track.phi()));
    }
  }
};

struct DTask {
  Filter notTracklet = aod::track::trackType != static_cast<uint8_t>(aod::track::TrackTypeEnum::Run2Tracklet);
  void process(aod::Collision const&, soa::Filtered<aod::MTracks> const& tracks)
  {
    for (auto& track : tracks) {
      LOGF(INFO, "%.3f == %.3f", track.spt(), std::abs(track.sigma1Pt() / track.signed1Pt()));
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>(cfgc, "produce-normalizedphi"),
    adaptAnalysisTask<BTask>(cfgc, "consume-normalizedphi"),
    adaptAnalysisTask<CTask>(cfgc, "consume-spawned-ephi"),
    adaptAnalysisTask<DTask>(cfgc, "consume-spawned-mtracks")};
}
