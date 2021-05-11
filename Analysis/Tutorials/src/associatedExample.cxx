// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
///
/// \brief index columns are used to relate information in non-joinable tables.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

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

namespace indices
{
DECLARE_SOA_INDEX_COLUMN(Track, track);
DECLARE_SOA_INDEX_COLUMN(HMPID, hmpid);
} // namespace indices

DECLARE_SOA_INDEX_TABLE_USER(HMPIDTracksIndex, Tracks, "HMPIDTRKIDX", indices::HMPIDId);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

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
      LOGF(INFO, "(%f, %f) - (%f, %f)", track.eta(), track.phiraw(), track.etas(), track.phis());
    }
  }
};

struct TTask {
  // use collisions with > 10 tracks only
  using myCol = soa::Join<aod::Collisions, aod::CollisionsExtra>;
  expressions::Filter multfilter = aod::collision::mult > 10;

  // group tracks according to collision
  void process(soa::Filtered<myCol>::iterator const& col, aod::Tracks const& tracks)
  {
    // get collision index and track multiplicity from the collisions ...
    LOGF(INFO, "[direct] ID: %d; %d == %d", col.globalIndex(), col.mult(), tracks.size());
    if (tracks.size() > 0) {
      // ... or through an associated track.
      auto track0 = tracks.begin();
      LOGF(INFO, "[index ] ID: %d; %d == %d", track0.collision_as<myCol>().globalIndex(), track0.collision_as<myCol>().mult(), tracks.size());
    }
  }
};

struct ZTask {
  using myCol = soa::Join<aod::Collisions, aod::CollisionsExtra>;

  void process(myCol const& collisions, aod::Tracks const& tracks)
  {
    // create multiplicity partitions of the Collisions table
    auto multbin0_10 = collisions.select(aod::collision::mult >= 0 && aod::collision::mult < 10);
    auto multbin10_30 = collisions.select(aod::collision::mult >= 10 && aod::collision::mult < 30);
    auto multbin30_100 = collisions.select(aod::collision::mult >= 30 && aod::collision::mult < 100);

    // loop over the collisions in each partition and ...
    LOGF(INFO, "Bin 0-10");
    for (auto& col : multbin0_10) {
      // ... find all the associated tracks
      auto groupedTracks = tracks.sliceBy(aod::track::collisionId, col.globalIndex());
      LOGF(INFO, "Collision %d; Ntrk = %d vs %d", col.globalIndex(), col.mult(), groupedTracks.size());
      if (groupedTracks.size() > 0) {
        auto track = groupedTracks.begin();
        LOGF(INFO, "Track 0 belongs to collision %d at Z = %f", track.collisionId(), track.collision_as<myCol>().posZ());
      }
    }

    LOGF(INFO, "Bin 10-30");
    for (auto& col : multbin10_30) {
      auto groupedTracks = tracks.sliceBy(aod::track::collisionId, col.globalIndex());
      LOGF(INFO, "Collision %d; Ntrk = %d vs %d", col.globalIndex(), col.mult(), groupedTracks.size());
    }

    LOGF(INFO, "Bin 30-100");
    for (auto& col : multbin30_100) {
      auto groupedTracks = tracks.sliceBy(aod::track::collisionId, col.globalIndex());
      LOGF(INFO, "Collision %d; Ntrk = %d vs %d", col.globalIndex(), col.mult(), groupedTracks.size());
    }
  }
};

struct BuildHmpidIndex {
  // build the index table HMPIDTracksIndex
  Builds<aod::HMPIDTracksIndex> idx;
  void init(InitContext const&){};
};

struct ConsumeHmpidIndex {
  using exTracks = soa::Join<aod::Tracks, aod::HMPIDTracksIndex>;

  // bind to Tracks and HMPIDs
  void process(aod::Collision const& collision, exTracks const& tracks, aod::HMPIDs const&)
  {
    LOGF(INFO, "Collision [%d]", collision.globalIndex());
    // loop over tracks of given collision
    for (auto& track : tracks) {
      // check weather given track has associated HMPID information
      if (track.has_hmpid()) {
        // access the HMIPD information associated with the actual track
        LOGF(INFO, "Track %d has HMPID info: %.2f", track.globalIndex(), track.hmpid().hmpidSignal());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>(cfgc, TaskName{"produce-etaphi"}),
    adaptAnalysisTask<BTask>(cfgc, TaskName{"consume-etaphi"}),
    adaptAnalysisTask<MTask>(cfgc, TaskName{"produce-mult"}),
    adaptAnalysisTask<TTask>(cfgc, TaskName{"consume-mult"}),
    adaptAnalysisTask<ZTask>(cfgc, TaskName{"partition-mult"}),
    adaptAnalysisTask<BuildHmpidIndex>(cfgc, TaskName{"build-index"}),
    adaptAnalysisTask<ConsumeHmpidIndex>(cfgc, TaskName{"consume-index"}),
  };
}
