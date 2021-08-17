// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief Accessing MC data and the related MC truth.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisCore/MC.h"

using namespace o2;
using namespace o2::framework;

// Simple access to collision
struct VertexDistribution {
  OutputObj<TH1F> vertex{TH1F("vertex", "vertex", 100, -10, 10)};

  // loop over MC truth McCollisions
  void process(aod::McCollision const& mcCollision)
  {
    LOGF(info, "MC. vtx-z = %f", mcCollision.posZ());
    vertex->Fill(mcCollision.posZ());
  }
};

// Grouping between MC particles and collisions
struct AccessMcData {
  OutputObj<TH1F> phiH{TH1F("phi", "phi", 100, 0., 2. * M_PI)};
  OutputObj<TH1F> etaH{TH1F("eta", "eta", 102, -2.01, 2.01)};

  // group according to McCollisions
  void process(aod::McCollision const& mcCollision, aod::McParticles const& mcParticles)
  {
    // access MC truth information with mcCollision() and mcParticle() methods
    LOGF(info, "MC. vtx-z = %f", mcCollision.posZ());
    LOGF(info, "First: %d | Length: %d", mcParticles.begin().index(), mcParticles.size());
    int count = 0;
    for (auto& mcParticle : mcParticles) {
      if (MC::isPhysicalPrimary(mcParticle)) {
        phiH->Fill(mcParticle.phi());
        etaH->Fill(mcParticle.eta());
        count++;
      }
    }
    LOGF(info, "Primaries for this collision: %d", count);
  }
};

// Access from tracks to MC particle
struct AccessMcTruth {
  OutputObj<TH1F> etaDiff{TH1F("etaDiff", ";eta_{MC} - eta_{Rec}", 100, -2, 2)};
  OutputObj<TH1F> phiDiff{TH1F("phiDiff", ";phi_{MC} - phi_{Rec}", 100, -M_PI, M_PI)};

  // group according to reconstructed Collisions
  void process(soa::Join<aod::Collisions, aod::McCollisionLabels>::iterator const& collision, soa::Join<aod::Tracks, aod::McTrackLabels> const& tracks,
               aod::McParticles const& mcParticles, aod::McCollisions const& mcCollisions)
  {
    // access MC truth information with mcCollision() and mcParticle() methods
    LOGF(info, "vtx-z (data) = %f | vtx-z (MC) = %f", collision.posZ(), collision.mcCollision().posZ());
    for (auto& track : tracks) {
      //if (track.trackType() != 0)
      //  continue;
      //if (track.labelMask() != 0)
      //  continue;
      auto particle = track.mcParticle();
      if (MC::isPhysicalPrimary(particle)) {
        etaDiff->Fill(particle.eta() - track.eta());
        auto delta = particle.phi() - track.phi();
        if (delta > M_PI) {
          delta -= 2 * M_PI;
        }
        if (delta < -M_PI) {
          delta += 2 * M_PI;
        }
        phiDiff->Fill(delta);
      }
      //LOGF(info, "eta: %.2f %.2f \t phi: %.2f %.2f | %d", track.mcParticle().eta(), track.eta(), track.mcParticle().phi(), track.phi(), track.mcParticle().index());
    }
  }
};

// Loop over MCColisions and get corresponding collisions (there can be more than one)
// For each of them get the corresponding tracks
struct LoopOverMcMatched {
  OutputObj<TH1F> etaDiff{TH1F("etaDiff", ";eta_{MC} - eta_{Rec}", 100, -2, 2)};
  void process(aod::McCollision const& mcCollision, soa::Join<aod::McCollisionLabels, aod::Collisions> const& collisions,
               soa::Join<aod::Tracks, aod::McTrackLabels> const& tracks, aod::McParticles const& mcParticles)
  {
    // access MC truth information with mcCollision() and mcParticle() methods
    LOGF(info, "MC collision at vtx-z = %f with %d mc particles and %d reconstructed collisions", mcCollision.posZ(), mcParticles.size(), collisions.size());
    for (auto& collision : collisions) {
      LOGF(info, "  Reconstructed collision at vtx-z = %f", collision.posZ());

      // NOTE this will be replaced by a improved grouping in the future
      auto groupedTracks = tracks.sliceBy(aod::track::collisionId, collision.globalIndex());
      LOGF(info, "  which has %d tracks", groupedTracks.size());
      for (auto& track : groupedTracks) {
        etaDiff->Fill(track.mcParticle().eta() - track.eta());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<VertexDistribution>(cfgc),
    adaptAnalysisTask<AccessMcData>(cfgc),
    adaptAnalysisTask<AccessMcTruth>(cfgc),
    adaptAnalysisTask<LoopOverMcMatched>(cfgc),
  };
}
