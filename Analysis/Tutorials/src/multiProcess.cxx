// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisCore/MC.h"

#include <TH1F.h>
#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Use MC info"}};
  workflowOptions.push_back(optionDoMC);
}

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

  void processRec(soa::Filtered<aod::Collisions>::iterator const& collision, aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      registry.fill(HIST("etaRec"), track.eta());
      registry.fill(HIST("phiRec"), track.phi());
    }
  }

  void processGen(soa::Filtered<aod::McCollisions>::iterator const& mcCollision, aod::McParticles const& mcParticles)
  {
    for (auto& particle : mcParticles) {
      registry.fill(HIST("etaMC"), particle.eta());
      registry.fill(HIST("phiMC"), particle.phi());
    }
  }

  void processResolution(soa::Filtered<soa::Join<aod::Collisions, aod::McCollisionLabels>>::iterator const& collision, soa::Join<aod::Tracks, aod::McTrackLabels> const& tracks, aod::McParticles const& mcParticles, aod::McCollisions const& mcCollisions)
  {
    LOGF(info, "vtx-z (data) = %f | vtx-z (MC) = %f", collision.posZ(), collision.mcCollision().posZ());
    for (auto& track : tracks) {
      registry.fill(HIST("etaDiff"), track.mcParticle().eta() - track.eta());
      registry.fill(HIST("phiDiff"), normalize(track.mcParticle().phi() - track.phi()));
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  if (!cfgc.options().get<bool>("doMC")) {
    return WorkflowSpec{
      // only use rec-level process when MC info is not there
      adaptAnalysisTask<MultipleProcessExample>(cfgc, Processes{&MultipleProcessExample::processRec})};
  }
  return WorkflowSpec{
    // use additional process functions when MC info is present
    // functions will be executed in the sequence they are listed - allows to use, for example,
    // histograms that were filled previously
    // produced tables *cannot* be used
    // filters will be applied *to all* processes
    adaptAnalysisTask<MultipleProcessExample>(cfgc, Processes{&MultipleProcessExample::processRec,
                                                              &MultipleProcessExample::processGen,
                                                              &MultipleProcessExample::processResolution})};
}
