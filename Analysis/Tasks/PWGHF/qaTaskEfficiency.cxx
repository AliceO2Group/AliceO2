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
  o2fw::Configurable<int> selPrim{"sel-prim", 1, "1 select primaries, 0 select all particles"};
  o2fw::Configurable<int> makeEff{"make-eff", 0, "Flag to produce the efficiency with TEfficiency"};

  o2fw::OutputObj<TList> list{"Efficiency"};
  o2fw::HistogramRegistry histos{"Histos", {}, o2fw::OutputObjHandlingPolicy::AnalysisObject};

  void init(o2fw::InitContext&)
  {
    const TString tag = Form("%s Eta [%.2f,%.2f] Phi [%.2f,%.2f] Prim %i",
                             o2::track::pid_constants::sNames[particle],
                             etaMin.value, etaMax.value,
                             phiMin.value, phiMax.value,
                             selPrim.value);
    const TString xaxis = "#it{p}_{T} GeV/#it{c}";

    histos.add("num", "Numerator " + tag + ";" + xaxis,
               o2fw::kTH1D, {{ptBins.value, ptMin.value, ptMax.value}});

    histos.add("den", "Denominator " + tag + ";" + xaxis,
               o2fw::kTH1D, {{ptBins.value, ptMin.value, ptMax.value}});

    list.setObject(new TList);
    if (makeEff.value) {
      list->Add(new TEfficiency("efficiency", "Efficiency " + tag + ";" + xaxis + ";Efficiency",
                                ptBins.value, ptMin.value, ptMax.value));
    }
  }

  void process(const o2::soa::Join<o2::aod::Tracks, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McParticles& mcParticles)
  {
    std::vector<int64_t> v_reco(tracks.size());
    int ntrks = 0;
    for (const auto& track : tracks) {
      const auto mcParticle = track.label();
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
        histos.fill(HIST("num"), mcParticle.pt());
        v_reco[ntrks++] = mcParticle.globalIndex();
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
          static_cast<TEfficiency*>(list->At(0))->Fill(std::find(v_reco.begin(), v_reco.end(), mcParticle.globalIndex()) != v_reco.end(), mcParticle.pt());
        }
        histos.fill(HIST("den"), mcParticle.pt());
      }
    }
  }
};

o2fw::WorkflowSpec defineDataProcessing(o2fw::ConfigContext const& cfgc)
{
  o2fw::WorkflowSpec w;
  if (cfgc.options().get<int>("eff-el")) {
    w.push_back(o2fw::adaptAnalysisTask<QATrackingEfficiencyPt<o2::track::PID::Electron>>("qa-tracking-efficiency-electron"));
  }
  if (cfgc.options().get<int>("eff-mu")) {
    w.push_back(o2fw::adaptAnalysisTask<QATrackingEfficiencyPt<o2::track::PID::Muon>>("qa-tracking-efficiency-muon"));
  }
  if (cfgc.options().get<int>("eff-pi")) {
    w.push_back(o2fw::adaptAnalysisTask<QATrackingEfficiencyPt<o2::track::PID::Pion>>("qa-tracking-efficiency-pion"));
  }
  if (cfgc.options().get<int>("eff-ka")) {
    w.push_back(o2fw::adaptAnalysisTask<QATrackingEfficiencyPt<o2::track::PID::Kaon>>("qa-tracking-efficiency-kaon"));
  }
  if (cfgc.options().get<int>("eff-pr")) {
    w.push_back(o2fw::adaptAnalysisTask<QATrackingEfficiencyPt<o2::track::PID::Proton>>("qa-tracking-efficiency-proton"));
  }
  return w;
}
