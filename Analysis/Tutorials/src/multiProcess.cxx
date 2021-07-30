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
/// \brief This is an example of a task with more than one process function.
///        Here a configurable is used to decide which process functions need to
///        be executed.
/// \author
/// \since

#include "Framework/AnalysisTask.h"
#include "AnalysisCore/MC.h"

#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
#include "Framework/runDataProcessing.h"

namespace
{
inline auto normalize(float a)
{
  if (a > M_PI) {
    a -= 2 * M_PI;
  }
  if (a < -M_PI) {
    a += 2 * M_PI;
  }
  return a;
}
} // namespace

// Analysis task with several process functions with different signatures
struct MultipleProcessExample {
  HistogramRegistry registry{
    "registry",
    {
      {"etaDiff", ";#eta_{MC} - #eta_{Rec}", {HistType::kTH1F, {{100, -2, 2}}}},
      {"phiDiff", ";#phi_{MC} - #phi_{Rec}", {HistType::kTH1F, {{100, -M_PI, M_PI}}}},
      {"etaRec", ";#eta_{Rec}", {HistType::kTH1F, {{100, -2, 2}}}},
      {"phiRec", ";#phi_{Rec}", {HistType::kTH1F, {{100, 0, 2 * M_PI}}}},
      {"etaMC", ";#eta_{MC}", {HistType::kTH1F, {{100, -2, 2}}}},
      {"phiMC", ";#phi_{MC}", {HistType::kTH1F, {{100, 0, 2 * M_PI}}}} ///
    }                                                                  ///
  };

  Filter RecColVtxZ = nabs(aod::collision::posZ) < 10.f;
  Filter GenColVtxZ = nabs(aod::mccollision::posZ) < 10.f;

  void processRec(soa::Filtered<aod::Collisions>::iterator const&, aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      registry.fill(HIST("etaRec"), track.eta());
      registry.fill(HIST("phiRec"), track.phi());
    }
  }

  /// name, description, function pointer, default value
  /// note that it has to be declared after the function, so that the pointer is known
  PROCESS_SWITCH(MultipleProcessExample, processRec, "Process reco level", true);

  void processGen(soa::Filtered<aod::McCollisions>::iterator const&, aod::McParticles const& mcParticles)
  {
    for (auto& particle : mcParticles) {
      registry.fill(HIST("etaMC"), particle.eta());
      registry.fill(HIST("phiMC"), particle.phi());
    }
  }

  PROCESS_SWITCH(MultipleProcessExample, processGen, "Process gen level", false);

  void processResolution(soa::Filtered<soa::Join<aod::Collisions, aod::McCollisionLabels>>::iterator const& collision, soa::Join<aod::Tracks, aod::McTrackLabels> const& tracks, aod::McParticles const&, aod::McCollisions const&)
  {
    LOGF(info, "vtx-z (data) = %f | vtx-z (MC) = %f", collision.posZ(), collision.mcCollision().posZ());
    for (auto& track : tracks) {
      registry.fill(HIST("etaDiff"), track.mcParticle().eta() - track.eta());
      registry.fill(HIST("phiDiff"), normalize(track.mcParticle().phi() - track.phi()));
    }
  }

  PROCESS_SWITCH(MultipleProcessExample, processResolution, "Process reco/gen matching", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<MultipleProcessExample>(cfgc) //
  };
}
