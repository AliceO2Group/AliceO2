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
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "AnalysisCore/MC.h"
#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Use MC information"}};
  workflowOptions.push_back(optionDoMC);
}
// always should be after customize() function
#include "Framework/runDataProcessing.h"

namespace o2::aod
{
namespace mcparticle
{
DECLARE_SOA_COLUMN(Primary, primary, bool);
} // namespace mcparticle
DECLARE_SOA_TABLE(SelPrimaries, "AOD", "SPRIM", aod::mcparticle::Primary);
} // namespace o2::aod

struct PseudorapidityDensity {
  Configurable<float> etaMax{"etaMax", 2.0, "max eta value"};
  Configurable<float> etaMin{"etaMin", -2.0, "min eta value"};
  Configurable<float> vtxZMax{"vtxZMax", 15, "max z vertex"};
  Configurable<float> vtxZMin{"vtxZMin", -15, "min z vertex"};
  Configurable<int> trackType{"trackType", o2::dataformats::GlobalTrackID::ITS, "types of tracks to select"};

  HistogramRegistry registry{
    "registry",
    {
      {"EventsNtrkZvtx", "; N_{trk}; Z_{vtx}; events", {HistType::kTH2F, {{301, -0.5, 300.5}, {201, -20.1, 20.1}}}}, //
      {"TracksEtaZvtx", "; #eta; Z_{vtx}; tracks", {HistType::kTH2F, {{21, -2.1, 2.1}, {201, -20.1, 20.1}}}},        //
      {"TracksPhiEta", "; #varphi; #eta; tracks", {HistType::kTH2F, {{600, 0, 2 * M_PI}, {21, -2.1, 2.1}}}},         //
      {"EventSelection", ";status;events", {HistType::kTH1F, {{3, 0.5, 3.5}}}},                                      //
    }                                                                                                                //
  };

  void init(InitContext&)
  {
    auto hstat = registry.get<TH1>(HIST("EventSelection"));
    auto x = hstat->GetXaxis();
    x->SetBinLabel(1, "All");
    x->SetBinLabel(2, "Selected");
    x->SetBinLabel(3, "Rejected");
  }

  Filter etaFilter = (aod::track::eta < etaMax) && (aod::track::eta > etaMin);
  Filter trackTypeFilter = (aod::track::trackType == static_cast<uint8_t>(trackType));
  Filter posZFilter = (aod::collision::posZ < vtxZMax) && (aod::collision::posZ > vtxZMin);

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision, soa::Filtered<aod::Tracks> const& tracks)
  {
    registry.fill(HIST("EventSelection"), 1.);
    if (collision.sel8()) {
      registry.fill(HIST("EventSelection"), 2.);
      auto z = collision.posZ();
      registry.fill(HIST("EventsNtrkZvtx"), tracks.size(), z);
      for (auto& track : tracks) {
          registry.fill(HIST("TracksEtaZvtx"), track.eta(), z);
          registry.fill(HIST("TracksPhiEta"), track.phi(), track.eta());
      }
    } else {
      registry.fill(HIST("EventSelection"), 3.);
    }
  }
};

struct SelectPhysicalPrimaries {
  Produces<aod::SelPrimaries> prims;
  void process(aod::McParticle const& particle)
  {
    prims(MC::isPhysicalPrimaryChargedRun3(particle));
  }
};

struct PseudorapidityDensityMc {
  Configurable<float> etaMax{"etaMax", 2.0, "max eta value"};
  Configurable<float> etaMin{"etaMin", -2.0, "min eta value"};
  Configurable<float> vtxZMax{"vtxZMax", 15, "max z vertex"};
  Configurable<float> vtxZMin{"vtxZMin", -15, "min z vertex"};
  Configurable<int> trackType{"trackType", o2::dataformats::GlobalTrackID::ITS, "types of tracks to select"};

  HistogramRegistry registry{
    "registry",
    {
      {"EventsNtrkZvtxGen", "; N_{trk}; Z_{vtx}; events", {HistType::kTH2F, {{301, -0.5, 300.5}, {201, -20.1, 20.1}}}}, //
      {"TracksEtaZvtxGen", "; #eta; Z_{vtx}; tracks", {HistType::kTH2F, {{21, -2.1, 2.1}, {201, -20.1, 20.1}}}},        //
      {"TracksPhiEtaGen", "; #varphi; #eta; tracks", {HistType::kTH2F, {{600, 0, 2 * M_PI}, {21, -2.1, 2.1}}}},         //
      {"EventEfficiency", "; status; events", {HistType::kTH1F, {{3, 0.5, 3.5}}}}                                       //
    }                                                                                                                   //
  };

  void init(InitContext&)
  {
    auto h = registry.get<TH1>(HIST("EventEfficiency"));
    auto x = h->GetXaxis();
    x->SetBinLabel(1, "Generated");
    x->SetBinLabel(2, "Reconstructed");
    x->SetBinLabel(3, "Selected");
  }

  Filter etaFilter = (aod::track::eta < etaMax) && (aod::track::eta > etaMin);
  Filter trackTypeFilter = (aod::track::trackType == static_cast<uint8_t>(trackType));
  Filter posZFilter = (aod::collision::posZ < vtxZMax) && (aod::collision::posZ > vtxZMin);
  Filter posZFilterMC = (aod::mccollision::posZ < vtxZMax) && (aod::mccollision::posZ > vtxZMin);

  using Particles = soa::Join<aod::McParticles, aod::SelPrimaries>;
  Filter primariesFilter = (aod::mcparticle::primary == true);

  void processGen(soa::Filtered<aod::McCollisions>::iterator const& collision, soa::Filtered<Particles> const& primaries)
  {
      registry.fill(HIST("EventEfficiency"), 1.);
      for (auto& particle : primaries) {
        if ((particle.eta() < etaMax) && (particle.eta() > etaMin)) {
          registry.fill(HIST("TracksEtaZvtxGen"), particle.eta(), collision.posZ());
          registry.fill(HIST("TracksPhiEtaGen"), particle.phi(), particle.eta());
        }
      }
  }

  void processMatching(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::McCollisionLabels>>::iterator const& collision, soa::Filtered<aod::Tracks> const& tracks, aod::McCollisions const&)
  {
    registry.fill(HIST("EventEfficiency"), 2.);
    auto z = collision.mcCollision().posZ();
    registry.fill(HIST("EventsNtrkZvtxGen"), tracks.size(), z);
    if (collision.sel8()) {
      registry.fill(HIST("EventEfficiency"), 3.);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  const bool doMC = cfgc.options().get<bool>("doMC");
  WorkflowSpec ws{adaptAnalysisTask<PseudorapidityDensity>(cfgc)};
  if (doMC) {
    ws.push_back(adaptAnalysisTask<SelectPhysicalPrimaries>(cfgc));
    ws.push_back(adaptAnalysisTask<PseudorapidityDensityMc>(cfgc, Processes{
                                                                    &PseudorapidityDensityMc::processGen,
                                                                    &PseudorapidityDensityMc::processMatching //
                                                                  }));
  }
  return ws;
}
