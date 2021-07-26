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
/// \author Peter Hristov <Peter.Hristov@cern.ch>, CERN
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Henrique J C Zanoli <henrique.zanoli@cern.ch>, Utrecht University
/// \author Nicolo' Jacazio <nicolo.jacazio@cern.ch>, CERN

// O2 inlcudes
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/Track.h"
#include "AnalysisCore/MC.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"eff-el", VariantType::Int, 1, {"Efficiency for the Electron PDG code"}},
    {"eff-mu", VariantType::Int, 1, {"Efficiency for the Muon PDG code"}},
    {"eff-pi", VariantType::Int, 1, {"Efficiency for the Pion PDG code"}},
    {"eff-ka", VariantType::Int, 1, {"Efficiency for the Kaon PDG code"}},
    {"eff-pr", VariantType::Int, 1, {"Efficiency for the Proton PDG code"}},
    {"eff-de", VariantType::Int, 0, {"Efficiency for the Deuteron PDG code"}},
    {"eff-he", VariantType::Int, 0, {"Efficiency for the Helium3 PDG code"}}};
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

/// Task to QA the efficiency of a particular particle defined by its pdg code
template <o2::track::pid_constants::ID particle>
struct QaTrackingEfficiency {
  static constexpr int nSpecies = 8;
  static constexpr int PDGs[nSpecies] = {kElectron, kMuonMinus, kPiPlus, kKPlus, kProton, 1000010020, 1000010030, 1000020030};
  static_assert(particle < nSpecies && "Maximum of particles reached");
  static constexpr int pdg = PDGs[particle];
  // Particle selection
  Configurable<float> etaMin{"eta-min", -3.f, "Lower limit in eta"};
  Configurable<float> etaMax{"eta-max", 3.f, "Upper limit in eta"};
  Configurable<float> phiMin{"phi-min", 0.f, "Lower limit in phi"};
  Configurable<float> phiMax{"phi-max", 6.284f, "Upper limit in phi"};
  Configurable<float> ptMin{"pt-min", 0.f, "Lower limit in pT"};
  Configurable<float> ptMax{"pt-max", 5.f, "Upper limit in pT"};
  Configurable<int> selPrim{"sel-prim", 1, "1 select primaries, 0 select all particles"};
  Configurable<int> pdgSign{"pdgSign", 0, "Sign to give to the PDG. If 0 both signs are accepted."};
  Configurable<bool> noFakes{"noFakes", false, "Flag to reject tracks that have fake hits"};
  // Event selection
  Configurable<int> nMinNumberOfContributors{"nMinNumberOfContributors", 2, "Minimum required number of contributors to the primary vertex"};
  Configurable<float> vertexZMin{"vertex-z-min", -10.f, "Minimum position of the generated vertez in Z (cm)"};
  Configurable<float> vertexZMax{"vertex-z-max", 10.f, "Maximum position of the generated vertez in Z (cm)"};
  // Histogram configuration
  Configurable<int> ptBins{"pt-bins", 500, "Number of pT bins"};
  Configurable<int> logPt{"log-pt", 0, "Flag to use a logarithmic pT axis"};
  Configurable<int> etaBins{"eta-bins", 500, "Number of eta bins"};
  Configurable<int> phiBins{"phi-bins", 500, "Number of phi bins"};
  // Task configuration
  Configurable<int> makeEff{"make-eff", 0, "Flag to produce the efficiency with TEfficiency"};

  OutputObj<TList> list{"Efficiency"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext&)
  {
    if (pdgSign != 0 && pdgSign != 1 && pdgSign != -1) {
      LOG(FATAL) << "Provide pdgSign as 0, 1, -1. Provided: " << pdgSign.value;
    }
    const TString tagPt = Form("%s #it{#eta} [%.2f,%.2f] #it{#varphi} [%.2f,%.2f] Prim %i",
                               o2::track::pid_constants::sNames[particle],
                               etaMin.value, etaMax.value,
                               phiMin.value, phiMax.value,
                               selPrim.value);
    const AxisSpec axisPt{ptBins, ptMin, ptMax, "#it{p}_{T} (GeV/#it{c})"};

    const TString tagEta = Form("%s #it{p}_{T} [%.2f,%.2f] #it{#varphi} [%.2f,%.2f] Prim %i",
                                o2::track::pid_constants::sNames[particle],
                                ptMin.value, ptMax.value,
                                phiMin.value, phiMax.value,
                                selPrim.value);
    const AxisSpec axisEta{etaBins, etaMin, etaMax, "#it{#eta}"};

    const TString tagPhi = Form("%s #it{#eta} [%.2f,%.2f] #it{p}_{T} [%.2f,%.2f] Prim %i",
                                o2::track::pid_constants::sNames[particle],
                                etaMin.value, etaMax.value,
                                ptMin.value, ptMax.value,
                                selPrim.value);
    const AxisSpec axisPhi{phiBins, phiMin, phiMax, "#it{#varphi} (rad)"};

    const AxisSpec axisSel{9, 0.5, 9.5, "Selection"};
    histos.add("eventSelection", "Event Selection", kTH1D, {axisSel});
    histos.get<TH1>(HIST("eventSelection"))->GetXaxis()->SetBinLabel(1, "Events read");
    histos.get<TH1>(HIST("eventSelection"))->GetXaxis()->SetBinLabel(2, "Passed Contrib.");
    histos.get<TH1>(HIST("eventSelection"))->GetXaxis()->SetBinLabel(3, "Passed Position");

    histos.add("trackSelection", "Track Selection", kTH1D, {axisSel});
    histos.get<TH1>(HIST("trackSelection"))->GetXaxis()->SetBinLabel(1, "Tracks read");
    histos.get<TH1>(HIST("trackSelection"))->GetXaxis()->SetBinLabel(2, "Passed Ev. Reco.");
    histos.get<TH1>(HIST("trackSelection"))->GetXaxis()->SetBinLabel(3, "Passed #it{p}_{T}");
    histos.get<TH1>(HIST("trackSelection"))->GetXaxis()->SetBinLabel(4, "Passed #it{#eta}");
    histos.get<TH1>(HIST("trackSelection"))->GetXaxis()->SetBinLabel(5, "Passed #it{#varphi}");
    histos.get<TH1>(HIST("trackSelection"))->GetXaxis()->SetBinLabel(6, "Passed Prim.");
    histos.get<TH1>(HIST("trackSelection"))->GetXaxis()->SetBinLabel(7, Form("Passed PDG %i", pdg));
    histos.get<TH1>(HIST("trackSelection"))->GetXaxis()->SetBinLabel(8, "Passed Fake");

    histos.add("partSelection", "Particle Selection", kTH1D, {axisSel});
    histos.get<TH1>(HIST("partSelection"))->GetXaxis()->SetBinLabel(1, "Particles read");
    histos.get<TH1>(HIST("partSelection"))->GetXaxis()->SetBinLabel(2, "Passed Ev. Reco.");
    histos.get<TH1>(HIST("partSelection"))->GetXaxis()->SetBinLabel(3, "Passed #it{p}_{T}");
    histos.get<TH1>(HIST("partSelection"))->GetXaxis()->SetBinLabel(4, "Passed #it{#eta}");
    histos.get<TH1>(HIST("partSelection"))->GetXaxis()->SetBinLabel(5, "Passed #it{#varphi}");
    histos.get<TH1>(HIST("partSelection"))->GetXaxis()->SetBinLabel(6, "Passed Prim.");
    histos.get<TH1>(HIST("partSelection"))->GetXaxis()->SetBinLabel(7, Form("Passed PDG %i", pdg));

    histos.add("pt/num", "Numerator " + tagPt, kTH1D, {axisPt});
    histos.add("pt/den", "Denominator " + tagPt, kTH1D, {axisPt});
    if (logPt) {
      makelogaxis(histos.get<TH1>(HIST("pt/num")));
      makelogaxis(histos.get<TH1>(HIST("pt/den")));
    }

    histos.add("eta/num", "Numerator " + tagEta, kTH1D, {axisEta});
    histos.add("eta/den", "Denominator " + tagEta, kTH1D, {axisEta});

    histos.add("phi/num", "Numerator " + tagPhi, kTH1D, {axisPhi});
    histos.add("phi/den", "Denominator " + tagPhi, kTH1D, {axisPhi});

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
      auto makeEfficiency2D = [&](TString effname, TString efftitle, auto templateHistoX, auto templateHistoY) {
        TAxis* axisX = histos.get<TH1>(templateHistoX)->GetXaxis();
        TAxis* axisY = histos.get<TH1>(templateHistoY)->GetXaxis();
        if (axisX->IsVariableBinSize() || axisY->IsVariableBinSize()) {
          list->Add(new TEfficiency(effname, efftitle, axisX->GetNbins(), axisX->GetXbins()->GetArray(), axisY->GetNbins(), axisY->GetXbins()->GetArray()));
        } else {
          list->Add(new TEfficiency(effname, efftitle, axisX->GetNbins(), axisX->GetXmin(), axisX->GetXmax(), axisY->GetNbins(), axisY->GetXmin(), axisY->GetXmax()));
        }
      };
      makeEfficiency("efficiencyVsPt", "Efficiency " + tagPt + ";#it{p}_{T} (GeV/#it{c});Efficiency", HIST("pt/num"));
      makeEfficiency("efficiencyVsP", "Efficiency " + tagPt + ";#it{p} (GeV/#it{c});Efficiency", HIST("pt/num"));
      makeEfficiency("efficiencyVsEta", "Efficiency " + tagEta + ";#it{#eta};Efficiency", HIST("eta/num"));
      makeEfficiency("efficiencyVsPhi", "Efficiency " + tagPhi + ";#it{#varphi} (rad);Efficiency", HIST("phi/num"));

      makeEfficiency2D("efficiencyVsPtVsEta", Form("Efficiency %s #it{#varphi} [%.2f,%.2f] Prim %i;%s;%s;Efficiency", o2::track::pid_constants::sNames[particle], phiMin.value, phiMax.value, selPrim.value, "#it{p}_{T} (GeV/#it{c})", "#it{#eta}"), HIST("pt/num"), HIST("eta/num"));
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
      histos.fill(HIST("eventSelection"), 1);
      if (collision.numContrib() < nMinNumberOfContributors) {
        continue;
      }
      histos.fill(HIST("eventSelection"), 2);
      const auto mcCollision = collision.mcCollision();
      if ((mcCollision.posZ() < vertexZMin || mcCollision.posZ() > vertexZMax)) {
        continue;
      }
      histos.fill(HIST("eventSelection"), 3);
      recoEvt[nevts++] = mcCollision.globalIndex();
    }
    recoEvt.resize(nevts);

    auto rejectParticle = [&](auto p, auto h) {
      histos.fill(h, 1);
      const auto evtReconstructed = std::find(recoEvt.begin(), recoEvt.end(), p.mcCollision().globalIndex()) != recoEvt.end();
      if (!evtReconstructed) { // Check that the event is reconstructed
        return true;
      }

      histos.fill(h, 2);
      if ((p.pt() < ptMin || p.pt() > ptMax)) { // Check pt
        return true;
      }
      histos.fill(h, 3);
      if ((p.eta() < etaMin || p.eta() > etaMax)) { // Check eta
        return true;
      }
      histos.fill(h, 4);
      if ((p.phi() < phiMin || p.phi() > phiMax)) { // Check phi
        return true;
      }
      histos.fill(h, 5);
      if ((selPrim == 1) && (!MC::isPhysicalPrimary<o2::aod::McParticles>(p))) { // Requiring is physical primary
        return true;
      }
      histos.fill(h, 6);

      // Selecting PDG code
      switch ((int)pdgSign) {
        case 0:
          if (abs(p.pdgCode()) != pdg) {
            return true;
          }
          break;
        case 1:
          if (p.pdgCode() != pdg) {
            return true;
          }
          break;
        case -1:
          if (p.pdgCode() != -pdg) {
            return true;
          }
          break;
        default:
          LOG(FATAL) << "Provide pdgSign as 0, 1, -1. Provided: " << pdgSign.value;
          break;
      }
      histos.fill(h, 7);

      return false;
    };

    std::vector<int64_t> recoTracks(tracks.size());
    int ntrks = 0;
    for (const auto& track : tracks) {
      const auto mcParticle = track.mcParticle();
      if (rejectParticle(mcParticle, HIST("trackSelection"))) {
        continue;
      }

      if (noFakes) { // Selecting tracks with no fake hits
        bool hasFake = false;
        for (int i = 0; i < 10; i++) { // From ITS to TPC
          if (track.mcMask() & 1 << i) {
            hasFake = true;
            break;
          }
        }
        if (hasFake) {
          continue;
        }
      }

      histos.fill(HIST("trackSelection"), 8);
      histos.fill(HIST("pt/num"), mcParticle.pt());
      histos.fill(HIST("eta/num"), mcParticle.eta());
      histos.fill(HIST("phi/num"), mcParticle.phi());
      recoTracks[ntrks++] = mcParticle.globalIndex();
    }

    for (const auto& mcParticle : mcParticles) {
      if (rejectParticle(mcParticle, HIST("partSelection"))) {
        continue;
      }

      if (makeEff) {
        const auto particleReconstructed = std::find(recoTracks.begin(), recoTracks.end(), mcParticle.globalIndex()) != recoTracks.end();
        static_cast<TEfficiency*>(list->At(0))->Fill(particleReconstructed, mcParticle.pt());
        static_cast<TEfficiency*>(list->At(1))->Fill(particleReconstructed, mcParticle.p());
        static_cast<TEfficiency*>(list->At(2))->Fill(particleReconstructed, mcParticle.eta());
        static_cast<TEfficiency*>(list->At(3))->Fill(particleReconstructed, mcParticle.phi());
        static_cast<TEfficiency*>(list->At(4))->Fill(particleReconstructed, mcParticle.pt(), mcParticle.eta());
      }
      histos.fill(HIST("pt/den"), mcParticle.pt());
      histos.fill(HIST("eta/den"), mcParticle.eta());
      histos.fill(HIST("phi/den"), mcParticle.phi());
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
  if (cfgc.options().get<int>("eff-de")) {
    w.push_back(adaptAnalysisTask<QaTrackingEfficiency<o2::track::PID::Deuteron>>(cfgc, TaskName{"qa-tracking-efficiency-deuteron"}));
  }
  if (cfgc.options().get<int>("eff-he")) {
    w.push_back(adaptAnalysisTask<QaTrackingEfficiency<o2::track::PID::Helium3>>(cfgc, TaskName{"qa-tracking-efficiency-helium3"}));
  }
  return w;
}
