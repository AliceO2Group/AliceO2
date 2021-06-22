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
    {"eff-el", VariantType::Int, 0, {"Efficiency for the Electron PDG code"}},
    {"eff-mu", VariantType::Int, 0, {"Efficiency for the Muon PDG code"}},
    {"eff-pi", VariantType::Int, 1, {"Efficiency for the Pion PDG code"}},
    {"eff-ka", VariantType::Int, 0, {"Efficiency for the Kaon PDG code"}},
    {"eff-pr", VariantType::Int, 0, {"Efficiency for the Proton PDG code"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

// ROOT includes
#include "TPDGCode.h"
#include "TEfficiency.h"
#include "TList.h"

template <typename T>
void makelogaxis(T h)
{
  const int nbins = h->GetNbinsX();
  double binp[nbins + 1];
  double max = h->GetXaxis()->GetBinUpEdge(nbins);
  double min = h->GetXaxis()->GetBinLowEdge(1);
  if (min <= 0) {
    min = 0.01;
  }
  double lmin = TMath::Log10(min);
  double ldelta = (TMath::Log10(max) - lmin) / ((double)nbins);
  for (int i = 0; i < nbins; i++) {
    binp[i] = TMath::Exp(TMath::Log(10) * (lmin + i * ldelta));
  }
  binp[nbins] = max + 1;
  h->GetXaxis()->Set(nbins, binp);
}

/// Task to QA the efficiency of a particular particle defined by particlePDG
template <o2::track::pid_constants::ID particle>
struct QaTrackingEfficiency {
  static constexpr PDG_t PDGs[5] = {kElectron, kMuonMinus, kPiPlus, kKPlus, kProton};
  static_assert(particle < 5 && "Maximum of particles reached");
  static constexpr int particlePDG = PDGs[particle];
  // Particle selection
  Configurable<float> etaMin{"eta-min", -3.f, "Lower limit in eta"};
  Configurable<float> etaMax{"eta-max", 3.f, "Upper limit in eta"};
  Configurable<float> phiMin{"phi-min", 0.f, "Lower limit in phi"};
  Configurable<float> phiMax{"phi-max", 6.284f, "Upper limit in phi"};
  Configurable<float> ptMin{"pt-min", 0.f, "Lower limit in pT"};
  Configurable<float> ptMax{"pt-max", 5.f, "Upper limit in pT"};
  // Event selection
  Configurable<int> nMinNumberOfContributors{"nMinNumberOfContributors", 2, "Minimum required number of contributors to the vertex"};
  Configurable<float> vertexZMin{"vertex-z-min", -10.f, "Minimum position of the generated vertez in Z (cm)"};
  Configurable<float> vertexZMax{"vertex-z-max", 10.f, "Maximum position of the generated vertez in Z (cm)"};
  // Histogram configuration
  Configurable<int> ptBins{"pt-bins", 500, "Number of pT bins"};
  Configurable<int> logPt{"log-pt", 0, "Flag to use a logarithmic pT axis"};
  Configurable<int> etaBins{"eta-bins", 500, "Number of eta bins"};
  Configurable<int> phiBins{"phi-bins", 500, "Number of phi bins"};
  Configurable<int> selPrim{"sel-prim", 1, "1 select primaries, 0 select all particles"};
  // Task configuration
  Configurable<int> makeEff{"make-eff", 0, "Flag to produce the efficiency with TEfficiency"};

  OutputObj<TList> list{"Efficiency"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext&)
  {
    const TString tagPt = Form("%s #it{#eta} [%.2f,%.2f] #it{#varphi} [%.2f,%.2f] Prim %i",
                               o2::track::pid_constants::sNames[particle],
                               etaMin.value, etaMax.value,
                               phiMin.value, phiMax.value,
                               selPrim.value);
    const TString xPt = "#it{p}_{T} (GeV/#it{c})";
    AxisSpec axisPt{ptBins, ptMin, ptMax};

    const TString tagEta = Form("%s #it{p}_{T} [%.2f,%.2f] #it{#varphi} [%.2f,%.2f] Prim %i",
                                o2::track::pid_constants::sNames[particle],
                                ptMin.value, ptMax.value,
                                phiMin.value, phiMax.value,
                                selPrim.value);
    const TString xEta = "#it{#eta}";
    AxisSpec axisEta{etaBins, etaMin, etaMax};

    const TString tagPhi = Form("%s #it{#eta} [%.2f,%.2f] #it{p}_{T} [%.2f,%.2f] Prim %i",
                                o2::track::pid_constants::sNames[particle],
                                etaMin.value, etaMax.value,
                                ptMin.value, ptMax.value,
                                selPrim.value);
    const TString xPhi = "#it{#varphi} (rad)";
    AxisSpec axisPhi{phiBins, phiMin, phiMax};

    histos.add("pt/num", "Numerator " + tagPt + ";" + xPt, kTH1D, {axisPt});
    histos.add("pt/den", "Denominator " + tagPt + ";" + xPt, kTH1D, {axisPt});
    if (logPt) {
      makelogaxis(histos.get<TH1>(HIST("pt/num")));
      makelogaxis(histos.get<TH1>(HIST("pt/den")));
    }

    histos.add("eta/num", "Numerator " + tagEta + ";" + xEta, kTH1D, {axisEta});
    histos.add("eta/den", "Denominator " + tagEta + ";" + xEta, kTH1D, {axisEta});

    histos.add("phi/num", "Numerator " + tagPhi + ";" + xPhi, kTH1D, {axisPhi});
    histos.add("phi/den", "Denominator " + tagPhi + ";" + xPhi, kTH1D, {axisPhi});

    list.setObject(new TList);
    if (makeEff) {
      auto makeEfficiency = [&](TString effname, TString efftitle, auto templateHisto) {
        TAxis* axis = histos.get<TH1>(templateHisto)->GetXaxis();
        if (axis->IsVariableBinSize()) {
          list->Add(new TEfficiency(effname, efftitle, axis->GetNbins(), axis->GetXbins()->GetArray()));
        } else {
          list->Add(new TEfficiency(effname, efftitle, axis->GetNbins(), axis->GetXmin(), axis->GetXmax()));
        }
      };
      makeEfficiency("efficiencyVsPt", "Efficiency " + tagPt + ";" + xPt + ";Efficiency", HIST("pt/num"));
      makeEfficiency("efficiencyVsEta", "Efficiency " + tagEta + ";" + xEta + ";Efficiency", HIST("eta/num"));
      makeEfficiency("efficiencyVsPhi", "Efficiency " + tagPhi + ";" + xPhi + ";Efficiency", HIST("phi/num"));
    }
  }

  void process(const o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>& collisions,
               const o2::soa::Join<o2::aod::Tracks, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McCollisions& mcCollisions,
               const o2::aod::McParticles& mcParticles)
  {
    std::vector<int64_t> recoEvt(collisions.size());
    int nevts = 0;
    for (const auto& collision : collisions) {
      if (collision.numContrib() < nMinNumberOfContributors) {
        continue;
      }
      const auto mcCollision = collision.mcCollision();
      if ((mcCollision.posZ() < vertexZMin || mcCollision.posZ() > vertexZMax)) {
        continue;
      }
      recoEvt[nevts++] = mcCollision.globalIndex();
    }
    recoEvt.resize(nevts);

    std::vector<int64_t> recoTracks(tracks.size());
    int ntrks = 0;
    for (const auto& track : tracks) {
      const auto mcParticle = track.mcParticle();
      if ((mcParticle.pt() < ptMin || mcParticle.pt() > ptMax)) { // Check pt
        continue;
      }
      if ((mcParticle.eta() < etaMin || mcParticle.eta() > etaMax)) { // Check eta
        continue;
      }
      if ((mcParticle.phi() < phiMin || mcParticle.phi() > phiMax)) { // Check phi
        continue;
      }
      if ((selPrim == 1) && (!MC::isPhysicalPrimary(mcParticles, mcParticle))) { // Requiring is physical primary
        continue;
      }
      if (abs(mcParticle.pdgCode()) == particlePDG) { // Checking PDG code
        histos.fill(HIST("pt/num"), mcParticle.pt());
        histos.fill(HIST("eta/num"), mcParticle.eta());
        histos.fill(HIST("phi/num"), mcParticle.phi());
        recoTracks[ntrks++] = mcParticle.globalIndex();
      }
    }

    for (const auto& mcParticle : mcParticles) {
      const auto evtReconstructed = std::find(recoEvt.begin(), recoEvt.end(), mcParticle.mcCollision().globalIndex()) != recoEvt.end();
      if (!evtReconstructed) {
        continue;
      }
      if ((mcParticle.eta() < etaMin || mcParticle.eta() > etaMax)) { // Check eta
        continue;
      }
      if ((mcParticle.phi() < phiMin || mcParticle.phi() > phiMax)) { // Check phi
        continue;
      }
      if ((selPrim == 1) && (!MC::isPhysicalPrimary(mcParticles, mcParticle))) { // Requiring is physical primary
        continue;
      }
      if (abs(mcParticle.pdgCode()) == particlePDG) { // Checking PDG code
        if (makeEff) {
          const auto particleReconstructed = std::find(recoTracks.begin(), recoTracks.end(), mcParticle.globalIndex()) != recoTracks.end();
          static_cast<TEfficiency*>(list->At(0))->Fill(particleReconstructed, mcParticle.pt());
          static_cast<TEfficiency*>(list->At(1))->Fill(particleReconstructed, mcParticle.eta());
          static_cast<TEfficiency*>(list->At(2))->Fill(particleReconstructed, mcParticle.phi());
        }
        histos.fill(HIST("pt/den"), mcParticle.pt());
        histos.fill(HIST("eta/den"), mcParticle.eta());
        histos.fill(HIST("phi/den"), mcParticle.phi());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec w;
  if (cfgc.options().get<int>("eff-el")) {
    w.push_back(adaptAnalysisTask<QaTrackingEfficiency<o2::track::PID::Electron>>(cfgc, TaskName{"qa-tracking-efficiency-electron"}));
  }
  if (cfgc.options().get<int>("eff-mu")) {
    w.push_back(adaptAnalysisTask<QaTrackingEfficiency<o2::track::PID::Muon>>(cfgc, TaskName{"qa-tracking-efficiency-muon"}));
  }
  if (cfgc.options().get<int>("eff-pi")) {
    w.push_back(adaptAnalysisTask<QaTrackingEfficiency<o2::track::PID::Pion>>(cfgc, TaskName{"qa-tracking-efficiency-pion"}));
  }
  if (cfgc.options().get<int>("eff-ka")) {
    w.push_back(adaptAnalysisTask<QaTrackingEfficiency<o2::track::PID::Kaon>>(cfgc, TaskName{"qa-tracking-efficiency-kaon"}));
  }
  if (cfgc.options().get<int>("eff-pr")) {
    w.push_back(adaptAnalysisTask<QaTrackingEfficiency<o2::track::PID::Proton>>(cfgc, TaskName{"qa-tracking-efficiency-proton"}));
  }
  return w;
}
