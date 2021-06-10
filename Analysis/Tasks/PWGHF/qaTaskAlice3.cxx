// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Peter Hristov <Peter.Hristov@cern.ch>, CERN
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Henrique J C Zanoli <henrique.zanoli@cern.ch>, Utrecht University
/// \author Nicolo' Jacazio <nicolo.jacazio@cern.ch>, CERN

// O2 inlcudes
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "ReconstructionDataFormats/DCA.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisCore/MC.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"rej-el", VariantType::Int, 1, {"Efficiency for the Electron PDG code"}},
    {"rej-mu", VariantType::Int, 0, {"Efficiency for the Muon PDG code"}},
    {"rej-pi", VariantType::Int, 0, {"Efficiency for the Pion PDG code"}},
    {"rej-ka", VariantType::Int, 0, {"Efficiency for the Kaon PDG code"}},
    {"rej-pr", VariantType::Int, 0, {"Efficiency for the Proton PDG code"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

// ROOT includes
#include "TPDGCode.h"
#include "TEfficiency.h"
#include "TList.h"


/// Task to QA the efficiency of a particular particle defined by particlePDG
template <o2::track::pid_constants::ID particle>
struct QaTrackingRejection {
  static constexpr PDG_t PDGs[5] = {kElectron, kMuonMinus, kPiPlus, kKPlus, kProton};
  static_assert(particle < 5 && "Maximum of particles reached");
  static constexpr int particlePDG = PDGs[particle];
  // Particle selection
  Configurable<int> ptBins{"ptBins", 100, "Number of pT bins"};
  Configurable<float> ptMin{"ptMin", 0.f, "Lower limit in pT"};
  Configurable<float> ptMax{"ptMax", 5.f, "Upper limit in pT"};
  HistogramRegistry histos{"HistogramsRejection"};

  void init(InitContext&)
  {
    AxisSpec ptAxis{ptBins, ptMin, ptMax};

    TString commonTitle = "";
    if (particlePDG != 0) {
      commonTitle += Form("PDG %i", particlePDG);
    }

    const TString pt = "#it{p}_{T} [GeV/#it{c}]";
    const TString eta = "#it{#eta}";
    const TString phi = "#it{#varphi} [rad]";

    histos.add("tracking/pt", commonTitle + ";" + pt, kTH1D, {ptAxis});
    histos.add("trackingPrm/pt", commonTitle + " Primary;" + pt, kTH1D, {ptAxis});
    histos.add("trackingSec/pt", commonTitle + " Secondary;" + pt, kTH1D, {ptAxis});
  }

  void process(const o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>& collisions,
               const o2::soa::Join<o2::aod::Tracks, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McCollisions& mcCollisions,
               const o2::aod::McParticles& mcParticles)
  {
    std::vector<int64_t> recoEvt(collisions.size());
    std::vector<int64_t> recoTracks(tracks.size());

    for (const auto& track : tracks) {
      const auto mcParticle = track.mcParticle();
      if (particlePDG != 0 && mcParticle.pdgCode() != particlePDG) { // Checking PDG code
        continue;
      }
      if (MC::isPhysicalPrimary(mcParticles, mcParticle)) {
        histos.fill(HIST("trackingPrm/pt"), track.pt());
      } else {
        histos.fill(HIST("trackingSec/pt"), track.pt());
      }
      histos.fill(HIST("tracking/pt"), track.pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec w;
  if (cfgc.options().get<int>("rej-el")) {
    w.push_back(adaptAnalysisTask<QaTrackingRejection<o2::track::PID::Electron>>(cfgc, TaskName{"qa-tracking-rejection-electron"}));
  }
  return w;
}
