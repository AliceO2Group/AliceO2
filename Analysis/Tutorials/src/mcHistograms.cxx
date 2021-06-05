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
struct AccessMCData {
  OutputObj<TH1F> phiH{TH1F("phi", "phi", 100, 0., 2. * M_PI)};
  OutputObj<TH1F> etaH{TH1F("eta", "eta", 102, -2.01, 2.01)};

  // group according to McCollisions
  void process(aod::McCollision const& mcCollision, aod::McParticles& mcParticles)
  {
    // access MC truth information with mcCollision() and mcParticle() methods
    LOGF(info, "MC. vtx-z = %f", mcCollision.posZ());
    LOGF(info, "First: %d | Length: %d", mcParticles.begin().index(), mcParticles.size());
    if (mcParticles.size() > 0) {
      LOGF(info, "Particles mother: %d", mcParticles.begin().mother0());
    }
    for (auto& mcParticle : mcParticles) {
      if (MC::isPhysicalPrimary(mcParticles, mcParticle)) {
        phiH->Fill(mcParticle.phi());
        etaH->Fill(mcParticle.eta());
      }
    }
  }
};

// Access from tracks to MC particle
struct AccessMCTruth {
  OutputObj<TH1F> etaDiff{TH1F("etaDiff", ";eta_{MC} - eta_{Rec}", 100, -2, 2)};
  OutputObj<TH1F> phiDiff{TH1F("phiDiff", ";phi_{MC} - phi_{Rec}", 100, -M_PI, M_PI)};

  // group according to reconstructed Collisions
  void process(soa::Join<aod::Collisions, aod::McCollisionLabels>::iterator const& collision, soa::Join<aod::Tracks, aod::McTrackLabels> const& tracks, aod::McParticles const& mcParticles, aod::McCollisions const& mcCollisions)
  {
    // access MC truth information with mcCollision() and mcParticle() methods
    LOGF(info, "vtx-z (data) = %f | vtx-z (MC) = %f", collision.posZ(), collision.mcCollision().posZ());
    for (auto& track : tracks) {
      //if (track.trackType() != 0)
      //  continue;
      //if (track.labelMask() != 0)
      //  continue;
      etaDiff->Fill(track.mcParticle().eta() - track.eta());
      auto delta = track.mcParticle().phi() - track.phi();
      if (delta > M_PI) {
        delta -= 2 * M_PI;
      }
      if (delta < -M_PI) {
        delta += 2 * M_PI;
      }
      phiDiff->Fill(delta);
      //LOGF(info, "eta: %.2f %.2f \t phi: %.2f %.2f | %d", track.mcParticle().eta(), track.eta(), track.mcParticle().phi(), track.phi(), track.mcParticle().index());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<VertexDistribution>(cfgc),
    adaptAnalysisTask<AccessMCData>(cfgc),
    adaptAnalysisTask<AccessMCTruth>(cfgc),
  };
}
