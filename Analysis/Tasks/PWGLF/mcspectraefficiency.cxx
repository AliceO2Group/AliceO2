// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// O2 includes
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisCore/MC.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

// ROOT includes
#include <TH1F.h>
#include "TPDGCode.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace MC;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"add-vertex", VariantType::Int, 1, {"Vertex plots"}},
    {"add-gen", VariantType::Int, 1, {"Generated plots"}},
    {"add-reco", VariantType::Int, 1, {"Reconstructed plots"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

#define PDGBINNING 100, 0, 100

// Simple access to collision
struct VertexTask {
  OutputObj<TH1F> vertex{TH1F("vertex", "vertex", 100, -10, 10)};

  void process(aod::McCollision const& mcCollision)
  {
    LOGF(info, "MC. vtx-z = %f", mcCollision.posZ());
    vertex->Fill(mcCollision.posZ());
  }
};

// Grouping between MC particles and collisions
struct GeneratedTask {
  OutputObj<TH2F> phiH{TH2F("phi", "phi;#phi;PDG code", 100, 0., 2. * M_PI, PDGBINNING)};
  OutputObj<TH2F> etaH{TH2F("eta", "eta;#eta;PDG code", 102, -2.01, 2.01, PDGBINNING)};
  OutputObj<TH2F> ptH{TH2F("pt", "pt;#it{p}_{T} (GeV/#it{c});PDG code", 500, 0, 20, PDGBINNING)};
  OutputObj<TH2F> pH{TH2F("p", "p;#it{p} (GeV/#it{c});PDG code", 500, 0, 20, PDGBINNING)};
  OutputObj<TH1F> pdgH{TH1F("pdg", "pdg;PDG code", PDGBINNING)};

  int events = 0;
  int particles = 0;
  int primaryparticles = 0;

  void process(aod::McCollision const& mcCollision, aod::McParticles& mcParticles)
  {
    LOGF(info, "MC. vtx-z = %f", mcCollision.posZ());
    for (auto& mcParticle : mcParticles) {
      if (abs(mcParticle.eta()) > 0.8) {
        continue;
      }
      if (isPhysicalPrimary(mcParticles, mcParticle)) {
        const auto pdg = Form("%i", mcParticle.pdgCode());
        pdgH->Fill(pdg, 1);
        const float pdgbin = pdgH->GetXaxis()->GetBinCenter(pdgH->GetXaxis()->FindBin(pdg));
        phiH->Fill(mcParticle.phi(), pdgbin);
        etaH->Fill(mcParticle.eta(), pdgbin);
        pH->Fill(sqrt(mcParticle.px() * mcParticle.px() + mcParticle.py() * mcParticle.py() + mcParticle.pz() * mcParticle.pz()), pdgbin);
        ptH->Fill(mcParticle.pt(), pdgbin);
        primaryparticles++;
      }
      particles++;
    }
    LOGF(info, "Events %i", events++ + 1);
    LOGF(info, "Particles %i", particles);
    LOGF(info, "Primaries %i", primaryparticles);
  }
};

// Access from tracks to MC particle
struct ReconstructedTask {
  OutputObj<TH2F> phiH{TH2F("phi", "phi;#phi;PDG code", 100, 0., 2. * M_PI, PDGBINNING)};
  OutputObj<TH2F> etaH{TH2F("eta", "eta;#eta;PDG code", 102, -2.01, 2.01, PDGBINNING)};
  OutputObj<TH2F> ptH{TH2F("pt", "pt;#it{p}_{T} (GeV/#it{c})", 500, 0, 20, PDGBINNING)};
  OutputObj<TH2F> pH{TH2F("p", "p;#it{p} (GeV/#it{c})", 500, 0, 20, PDGBINNING)};
  OutputObj<TH2F> dcaxyH{TH2F("dcaxy", "dcaxy;DCA_{xy} (cm)", 500, -10, 10, PDGBINNING)};
  OutputObj<TH2F> dcazH{TH2F("dcaz", "dcaz;DCA_{z} (cm)", 500, -10, 10, PDGBINNING)};
  OutputObj<TH1F> pdgH{TH1F("pdg", "pdg;PDG code", PDGBINNING)};
  OutputObj<TH2F> dcaxysecH{TH2F("dcaxysec", "dcaxysec;DCA_{xy} (cm)", 500, -10, 10, PDGBINNING)};
  OutputObj<TH2F> dcazsecH{TH2F("dcazsec", "dcazsec;DCA_{z} (cm)", 500, -10, 10, PDGBINNING)};
  OutputObj<TH1F> pdgsecH{TH1F("pdgsec", "pdgsec;PDG code", PDGBINNING)};

  OutputObj<TH2F> phiDiff{TH2F("phiDiff", ";phi_{MC} - phi_{Rec}", 100, -M_PI, M_PI, PDGBINNING)};
  OutputObj<TH2F> etaDiff{TH2F("etaDiff", ";eta_{MC} - eta_{Rec}", 100, -2, 2, PDGBINNING)};
  OutputObj<TH2F> ptDiff{TH2F("ptDiff", "ptDiff;#it{p}_{T}_{MC} #it{p}_{T}_{Rec} (GeV/#it{c})", 500, -2, 2, PDGBINNING)};
  OutputObj<TH2F> pDiff{TH2F("pDiff", "pDiff;#it{p}_{MC} #it{p}_{Rec} (GeV/#it{c})", 500, -2, 2, PDGBINNING)};

  Filter trackAcceptance = (nabs(aod::track::eta) < 0.8f);
  Filter trackCuts = ((aod::track::isGlobalTrack == (uint8_t) true) || (aod::track::isGlobalTrackSDD == (uint8_t) true));

  void process(soa::Join<aod::Collisions, aod::McCollisionLabels>::iterator const& collision,
               soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksExtended, aod::McTrackLabels, aod::TrackSelection>> const& tracks,
               aod::McParticles& mcParticles, aod::McCollisions const& mcCollisions)
  {
    LOGF(info, "vtx-z (data) = %f | vtx-z (MC) = %f", collision.posZ(), collision.label().posZ());
    for (auto& track : tracks) {
      const auto pdg = Form("%i", track.label().pdgCode());
      if (!isPhysicalPrimary(mcParticles, track.label())) {
        pdgsecH->Fill(pdg, 1);
        const float pdgbinsec = pdgH->GetXaxis()->GetBinCenter(pdgsecH->GetXaxis()->FindBin(pdg));
        dcaxysecH->Fill(track.dcaXY(), pdgbinsec);
        dcazsecH->Fill(track.dcaZ(), pdgbinsec);
        continue;
      }
      pdgH->Fill(pdg, 1); // Filling the first bin and check its bin
      const float pdgbin = pdgH->GetXaxis()->GetBinCenter(pdgH->GetXaxis()->FindBin(pdg));
      phiH->Fill(track.phi(), pdgbin);
      etaH->Fill(track.eta(), pdgbin);
      pH->Fill(track.p(), pdgbin);
      ptH->Fill(track.pt(), pdgbin);
      dcaxyH->Fill(track.dcaXY(), pdgbin);
      dcazH->Fill(track.dcaZ(), pdgbin);

      etaDiff->Fill(track.label().eta() - track.eta(), pdgbin);
      auto delta = track.label().phi() - track.phi();
      if (delta > M_PI) {
        delta -= 2 * M_PI;
      }
      if (delta < -M_PI) {
        delta += 2 * M_PI;
      }
      phiDiff->Fill(delta, pdgbin);
      pDiff->Fill(sqrt(track.label().px() * track.label().px() + track.label().py() * track.label().py() + track.label().pz() * track.label().pz()) - track.p(), pdgbin);
      ptDiff->Fill(track.label().pt() - track.pt(), pdgbin);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  const bool vertex = cfgc.options().get<int>("add-vertex");
  const bool gen = cfgc.options().get<int>("add-gen");
  const bool reco = cfgc.options().get<int>("add-reco");
  WorkflowSpec workflow{};
  if (vertex) {
    workflow.push_back(adaptAnalysisTask<VertexTask>(cfgc, "vertex-histogram"));
  }
  if (gen) {
    workflow.push_back(adaptAnalysisTask<GeneratedTask>(cfgc, "generator-histogram"));
  }
  if (reco) {
    workflow.push_back(adaptAnalysisTask<ReconstructedTask>(cfgc, "reconstructed-histogram"));
  }
  return workflow;
}
