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
DECLARE_SOA_COLUMN(AEta, etas, float);
DECLARE_SOA_COLUMN(APhi, phis, float);
DECLARE_SOA_COLUMN(APt, pts, float);
} // namespace etaphi

DECLARE_SOA_TABLE(EtaPhi, "AOD", "ETAPHI",
                  etaphi::AEta, etaphi::APhi);

namespace collision
{
DECLARE_SOA_COLUMN(Mult, mult, int32_t);
} // namespace collision

DECLARE_SOA_TABLE(CollisionsExtra, "AOD", "COLEXTRA",
                  collision::Mult);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

// This is a very simple example showing how to iterate over tracks
// and create a new collection for them.
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  Produces<aod::EtaPhi> etaphi;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      float phi = asin(track.snp()) + track.alpha() + static_cast<float>(M_PI);
      float eta = log(tan(0.25f * static_cast<float>(M_PI) - 0.5f * atan(track.tgl())));
      etaphi(eta, phi);
    }
  }
};

struct MTask {
  Produces<aod::CollisionsExtra> colextra;

  void process(aod::Collision const&, aod::Tracks const& tracks)
  {
    colextra(tracks.size());
  }
};

struct BTask {
  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::EtaPhi> const& extTracks)
  {
    LOGF(INFO, "ID: %d", collision.globalIndex());
    LOGF(INFO, "Tracks: %d", extTracks.size());
    for (auto& track : extTracks) {
      LOGF(INFO, "(%f, %f) - (%f, %f)", track.eta(), track.phi(), track.etas(), track.phis());
    }
  }
};

struct TTask {
  using myCol = soa::Join<aod::Collisions, aod::CollisionsExtra>;
  void process(soa::Join<aod::Collisions, aod::CollisionsExtra>::iterator const& col, aod::Tracks const& tracks)
  {
    LOGF(INFO, "[direct] ID: %d; %d == %d", col.globalIndex(), col.mult(), tracks.size());
    if (tracks.size() > 0) {
      auto track0 = tracks.begin();
      LOGF(INFO, "[index ] ID: %d; %d == %d", track0.collision_as<myCol>().globalIndex(), track0.collision_as<myCol>().mult(), tracks.size());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-etaphi"),
    adaptAnalysisTask<BTask>("consume-etaphi"),
    adaptAnalysisTask<MTask>("produce-mult"),
    adaptAnalysisTask<TTask>("consume-mult")};
}
