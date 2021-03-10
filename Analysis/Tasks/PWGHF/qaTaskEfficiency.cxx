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
#include "Framework/AnalysisDataModel.h"
#include "Framework/HistogramRegistry.h"
#include "ReconstructionDataFormats/DCA.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisCore/MC.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

namespace o2fw = o2::framework;

namespace o2exp = o2::framework::expressions;
namespace o2df = o2::dataformats;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2fw::ConfigParamSpec> options{
    {"eff-el", o2fw::VariantType::Int, 0, {"Efficiency for the Electron PDG code"}},
    {"eff-mu", o2fw::VariantType::Int, 0, {"Efficiency for the Muon PDG code"}},
    {"eff-pi", o2fw::VariantType::Int, 1, {"Efficiency for the Pion PDG code"}},
    {"eff-ka", o2fw::VariantType::Int, 0, {"Efficiency for the Kaon PDG code"}},
    {"eff-pr", o2fw::VariantType::Int, 0, {"Efficiency for the Proton PDG code"}}};
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
struct QATrackingEfficiencyPt {
  static constexpr PDG_t PDGs[5] = {kElectron, kMuonMinus, kPiPlus, kKPlus, kProton};
  static_assert(particle < 5 && "Maximum of particles reached");
  static constexpr int particlePDG = PDGs[particle];
  o2fw::Configurable<float> etaMin{"eta-min", -3.f, "Lower limit in eta"};
  o2fw::Configurable<float> etaMax{"eta-max", 3.f, "Upper limit in eta"};
  o2fw::Configurable<float> phiMin{"phi-min", 0.f, "Lower limit in phi"};
  o2fw::Configurable<float> phiMax{"phi-max", 2.f * M_PI, "Upper limit in phi"};
  o2fw::Configurable<float> ptMin{"pt-min", 0.f, "Lower limit in pT"};
  o2fw::Configurable<float> ptMax{"pt-max", 5.f, "Upper limit in pT"};
  o2fw::Configurable<int> ptBins{"pt-bins", 500, "Number of pT bins"};
  o2fw::Configurable<int> logPt{"log-pt", 0, "Flag to use a logarithmic pT axis"};
  o2fw::Configurable<int> etaBins{"eta-bins", 500, "Number of eta bins"};
  o2fw::Configurable<int> phiBins{"phi-bins", 500, "Number of phi bins"};
  o2fw::Configurable<int> selPrim{"sel-prim", 1, "1 select primaries, 0 select all particles"};
  o2fw::Configurable<int> makeEff{"make-eff", 0, "Flag to produce the efficiency with TEfficiency"};

  o2fw::OutputObj<TList> list{"Efficiency"};
  o2fw::HistogramRegistry histos{"Histos", {}, o2fw::OutputObjHandlingPolicy::AnalysisObject};

  void init(o2fw::InitContext&)
  {
    const TString tagPt = Form("%s #eta [%.2f,%.2f] #varphi [%.2f,%.2f] Prim %i",
                               o2::track::pid_constants::sNames[particle],
                               etaMin.value, etaMax.value,
                               phiMin.value, phiMax.value,
                               selPrim.value);
    const TString xPt = "#it{p}_{T} (GeV/#it{c})";
    o2fw::AxisSpec axisPt{ptBins.value, ptMin.value, ptMax.value};

    const TString tagEta = Form("%s #it{p}_{T} [%.2f,%.2f] #varphi [%.2f,%.2f] Prim %i",
                                o2::track::pid_constants::sNames[particle],
                                ptMin.value, ptMax.value,
                                phiMin.value, phiMax.value,
                                selPrim.value);
    const TString xEta = "#eta";
    o2fw::AxisSpec axisEta{etaBins.value, etaMin.value, etaMax.value};

    const TString tagPhi = Form("%s #eta [%.2f,%.2f] #it{p}_{T} [%.2f,%.2f] Prim %i",
                                o2::track::pid_constants::sNames[particle],
                                etaMin.value, etaMax.value,
                                ptMin.value, ptMax.value,
                                selPrim.value);
    const TString xPhi = "#varphi (rad)";
    o2fw::AxisSpec axisPhi{phiBins.value, phiMin.value, phiMax.value};

    histos.add("pt/num", "Numerator " + tagPt + ";" + xPt,
               o2fw::kTH1D, {axisPt});
    histos.add("pt/den", "Denominator " + tagPt + ";" + xPt,
               o2fw::kTH1D, {axisPt});
    if (logPt.value) {
      makelogaxis(histos.get<TH1>(HIST("pt/num")));
      makelogaxis(histos.get<TH1>(HIST("pt/den")));
    }

    histos.add("eta/num", "Numerator " + tagEta + ";" + xEta,
               o2fw::kTH1D, {axisEta});
    histos.add("eta/den", "Denominator " + tagEta + ";" + xEta,
               o2fw::kTH1D, {axisEta});

    histos.add("phi/num", "Numerator " + tagPhi + ";" + xPhi,
               o2fw::kTH1D, {axisPhi});
    histos.add("phi/den", "Denominator " + tagPhi + ";" + xPhi,
               o2fw::kTH1D, {axisPhi});

    list.setObject(new TList);
    if (makeEff.value) {
      auto makeEfficiency = [&](TString effname, TString efftitle, auto templateHisto) {
        TAxis* axis = histos.get<TH1>(templateHisto)->GetXaxis();
        if (axis->IsVariableBinSize()) {
          list->Add(new TEfficiency(effname, efftitle,
                                    axis->GetNbins(),
                                    axis->GetXbins()->GetArray()));
        } else {
          list->Add(new TEfficiency(effname, efftitle,
                                    axis->GetNbins(),
                                    axis->GetXmin(),
                                    axis->GetXmax()));
        }
      };
      makeEfficiency("efficiencyVsPt", "Efficiency " + tagPt + ";" + xPt + ";Efficiency", HIST("pt/num"));
      makeEfficiency("efficiencyVsEta", "Efficiency " + tagEta + ";" + xEta + ";Efficiency", HIST("eta/num"));
      makeEfficiency("efficiencyVsPhi", "Efficiency " + tagPhi + ";" + xPhi + ";Efficiency", HIST("phi/num"));
    }
  }

  void process(const o2::soa::Join<o2::aod::Tracks, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McParticles& mcParticles)
  {
    std::vector<int64_t> recoTracks(tracks.size());
    int ntrks = 0;
    for (const auto& track : tracks) {
      const auto mcParticle = track.mcParticle();
      if ((mcParticle.eta() < etaMin.value || mcParticle.eta() > etaMax.value)) { // Check eta
        continue;
      }
      if ((mcParticle.phi() < phiMin.value || mcParticle.phi() > phiMax.value)) { // Check phi
        continue;
      }
      if ((selPrim.value == 1) && (!MC::isPhysicalPrimary(mcParticles, mcParticle))) { // Requiring is physical primary
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
      if ((mcParticle.eta() < etaMin.value || mcParticle.eta() > etaMax.value)) { // Check eta
        continue;
      }
      if ((mcParticle.phi() < phiMin.value || mcParticle.phi() > phiMax.value)) { // Check phi
        continue;
      }
      if ((selPrim.value == 1) && (!MC::isPhysicalPrimary(mcParticles, mcParticle))) { // Requiring is physical primary
        continue;
      }
      if (abs(mcParticle.pdgCode()) == particlePDG) { // Checking PDG code
        if (makeEff.value) {
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

o2fw::WorkflowSpec defineDataProcessing(o2fw::ConfigContext const& cfgc)
{
  o2fw::WorkflowSpec w;
  if (cfgc.options().get<int>("eff-el")) {
    w.push_back(o2fw::adaptAnalysisTask<QATrackingEfficiencyPt<o2::track::PID::Electron>>(cfgc, o2fw::TaskName{"qa-tracking-efficiency-electron"}));
  }
  if (cfgc.options().get<int>("eff-mu")) {
    w.push_back(o2fw::adaptAnalysisTask<QATrackingEfficiencyPt<o2::track::PID::Muon>>(cfgc, o2fw::TaskName{"qa-tracking-efficiency-muon"}));
  }
  if (cfgc.options().get<int>("eff-pi")) {
    w.push_back(o2fw::adaptAnalysisTask<QATrackingEfficiencyPt<o2::track::PID::Pion>>(cfgc, o2fw::TaskName{"qa-tracking-efficiency-pion"}));
  }
  if (cfgc.options().get<int>("eff-ka")) {
    w.push_back(o2fw::adaptAnalysisTask<QATrackingEfficiencyPt<o2::track::PID::Kaon>>(cfgc, o2fw::TaskName{"qa-tracking-efficiency-kaon"}));
  }
  if (cfgc.options().get<int>("eff-pr")) {
    w.push_back(o2fw::adaptAnalysisTask<QATrackingEfficiencyPt<o2::track::PID::Proton>>(cfgc, o2fw::TaskName{"qa-tracking-efficiency-proton"}));
  }
  return w;
}
