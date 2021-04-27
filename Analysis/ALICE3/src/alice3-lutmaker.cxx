// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author Nicolo' Jacazio <nicolo.jacazio@cern.ch>, CERN
/// \brief Task to extract LUTs for the fast simulation from full simulation
/// \since 27/04/2021

// O2 includes
#include "Framework/AnalysisTask.h"
#include "AnalysisCore/MC.h"
#include "ReconstructionDataFormats/Track.h"

using namespace o2;
using namespace framework;
using namespace framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"lut-el", VariantType::Int, 1, {"LUT input for the Electron PDG code"}},
    {"lut-mu", VariantType::Int, 1, {"LUT input for the Muon PDG code"}},
    {"lut-pi", VariantType::Int, 1, {"LUT input for the Pion PDG code"}},
    {"lut-ka", VariantType::Int, 1, {"LUT input for the Kaon PDG code"}},
    {"lut-pr", VariantType::Int, 1, {"LUT input for the Proton PDG code"}},
    {"lut-tr", VariantType::Int, 0, {"LUT input for the Triton PDG code"}},
    {"lut-de", VariantType::Int, 0, {"LUT input for the Deuteron PDG code"}},
    {"lut-he", VariantType::Int, 0, {"LUT input for the Helium3 PDG code"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

template <o2::track::pid_constants::ID particle>
struct Alice3LutMaker {
  static constexpr int nSpecies = 8;
  static constexpr int PDGs[nSpecies] = {kElectron, kMuonMinus, kPiPlus, kKPlus, kProton, 1000010020, 1000010030, 1000020030};
  static_assert(particle < nSpecies && "Maximum of particles reached");
  static constexpr int pdg = PDGs[particle];
  Configurable<bool> selPrim{"sel-prim", false, "If true selects primaries, if not select all particles"};
  Configurable<int> etaBins{"eta-bins", 80, "Number of eta bins"};
  Configurable<float> etaMin{"eta-min", -4.f, "Lower limit in eta"};
  Configurable<float> etaMax{"eta-max", 4.f, "Upper limit in eta"};
  Configurable<int> ptBins{"pt-bins", 200, "Number of pT bins"};
  Configurable<float> ptMin{"pt-min", -2.f, "Lower limit in pT"};
  Configurable<float> ptMax{"pt-max", 2.f, "Upper limit in pT"};
  Configurable<int> logPt{"log-pt", 0, "Flag to use a logarithmic pT axis, in this case the pT limits are the expontents"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext&)
  {
    const TString commonTitle = Form(" PDG %i", pdg);
    AxisSpec axisPt{ptBins, ptMin, ptMax, "#it{p}_{T} GeV/#it{c}"};
    if (logPt) {
      const double min = axisPt.binEdges[0];
      const double width = (axisPt.binEdges[1] - axisPt.binEdges[0]) / axisPt.nBins.value();
      axisPt.binEdges.clear();
      axisPt.binEdges.resize(0);
      for (int i = 0; i < axisPt.nBins.value() + 1; i++) {
        axisPt.binEdges.push_back(std::pow(10., min + i * width));
      }
      axisPt.nBins = std::nullopt;
    }
    const AxisSpec axisEta{etaBins, etaMin, etaMax, "#it{#eta}"};

    histos.add("multiplicity", "Track multiplicity;Tracks per event;Events", kTH1F, {{100, 0, 2000}});
    histos.add("pt", "pt" + commonTitle, kTH1F, {axisPt});
    histos.add("eta", "eta" + commonTitle, kTH1F, {axisEta});
    histos.add("CovMat_cYY", "cYY" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_cZY", "cZY" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_cZZ", "cZZ" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_cSnpY", "cSnpY" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_cSnpZ", "cSnpZ" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_cSnpSnp", "cSnpSnp" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_cTglY", "cTglY" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_cTglZ", "cTglZ" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_cTglSnp", "cTglSnp" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_cTglTgl", "cTglTgl" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_c1PtY", "c1PtY" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_c1PtZ", "c1PtZ" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_c1PtSnp", "c1PtSnp" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_c1PtTgl", "c1PtTgl" + commonTitle, kTProfile2D, {axisPt, axisEta});
    histos.add("CovMat_c1Pt21Pt2", "c1Pt21Pt2" + commonTitle, kTProfile2D, {axisPt, axisEta});

    histos.add("Efficiency", "Efficiency" + commonTitle, kTProfile2D, {axisPt, axisEta});
  }

  void process(const soa::Join<aod::Tracks, aod::TracksCov, aod::McTrackLabels>& tracks,
               const aod::McParticles& mcParticles)
  {
    std::vector<int64_t> recoTracks(tracks.size());
    int ntrks = 0;

    for (const auto& track : tracks) {
      const auto mcParticle = track.mcParticle();
      if (mcParticle.pdgCode() != pdg) {
        continue;
      }
      if (selPrim.value && !MC::isPhysicalPrimary(mcParticles, mcParticle)) { // Requiring is physical primary
        continue;
      }

      recoTracks[ntrks++] = mcParticle.globalIndex();

      histos.fill(HIST("pt"), mcParticle.pt());
      histos.fill(HIST("eta"), mcParticle.eta());
      histos.fill(HIST("CovMat_cYY"), mcParticle.pt(), mcParticle.eta(), track.cYY());
      histos.fill(HIST("CovMat_cZY"), mcParticle.pt(), mcParticle.eta(), track.cZY());
      histos.fill(HIST("CovMat_cZZ"), mcParticle.pt(), mcParticle.eta(), track.cZZ());
      histos.fill(HIST("CovMat_cSnpY"), mcParticle.pt(), mcParticle.eta(), track.cSnpY());
      histos.fill(HIST("CovMat_cSnpZ"), mcParticle.pt(), mcParticle.eta(), track.cSnpZ());
      histos.fill(HIST("CovMat_cSnpSnp"), mcParticle.pt(), mcParticle.eta(), track.cSnpSnp());
      histos.fill(HIST("CovMat_cTglY"), mcParticle.pt(), mcParticle.eta(), track.cTglY());
      histos.fill(HIST("CovMat_cTglZ"), mcParticle.pt(), mcParticle.eta(), track.cTglZ());
      histos.fill(HIST("CovMat_cTglSnp"), mcParticle.pt(), mcParticle.eta(), track.cTglSnp());
      histos.fill(HIST("CovMat_cTglTgl"), mcParticle.pt(), mcParticle.eta(), track.cTglTgl());
      histos.fill(HIST("CovMat_c1PtY"), mcParticle.pt(), mcParticle.eta(), track.c1PtY());
      histos.fill(HIST("CovMat_c1PtZ"), mcParticle.pt(), mcParticle.eta(), track.c1PtZ());
      histos.fill(HIST("CovMat_c1PtSnp"), mcParticle.pt(), mcParticle.eta(), track.c1PtSnp());
      histos.fill(HIST("CovMat_c1PtTgl"), mcParticle.pt(), mcParticle.eta(), track.c1PtTgl());
      histos.fill(HIST("CovMat_c1Pt21Pt2"), mcParticle.pt(), mcParticle.eta(), track.c1Pt21Pt2());
    }
    histos.fill(HIST("multiplicity"), ntrks);

    for (const auto& mcParticle : mcParticles) {
      if (mcParticle.pdgCode() != pdg) {
        continue;
      }
      if (!MC::isPhysicalPrimary(mcParticles, mcParticle)) { // Requiring is physical primary
        continue;
      }

      if (std::find(recoTracks.begin(), recoTracks.end(), mcParticle.globalIndex()) != recoTracks.end()) {
        histos.fill(HIST("Efficiency"), mcParticle.pt(), mcParticle.eta(), 1.);
      } else {
        histos.fill(HIST("Efficiency"), mcParticle.pt(), mcParticle.eta(), 0.);
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec w;
  if (cfgc.options().get<int>("lut-el")) {
    w.push_back(adaptAnalysisTask<Alice3LutMaker<o2::track::PID::Electron>>(cfgc, TaskName{"alice3-lutmaker-electron"}));
  }
  if (cfgc.options().get<int>("lut-mu")) {
    w.push_back(adaptAnalysisTask<Alice3LutMaker<o2::track::PID::Muon>>(cfgc, TaskName{"alice3-lutmaker-muon"}));
  }
  if (cfgc.options().get<int>("lut-pi")) {
    w.push_back(adaptAnalysisTask<Alice3LutMaker<o2::track::PID::Pion>>(cfgc, TaskName{"alice3-lutmaker-pion"}));
  }
  if (cfgc.options().get<int>("lut-ka")) {
    w.push_back(adaptAnalysisTask<Alice3LutMaker<o2::track::PID::Kaon>>(cfgc, TaskName{"alice3-lutmaker-kaon"}));
  }
  if (cfgc.options().get<int>("lut-pr")) {
    w.push_back(adaptAnalysisTask<Alice3LutMaker<o2::track::PID::Proton>>(cfgc, TaskName{"alice3-lutmaker-proton"}));
  }
  if (cfgc.options().get<int>("lut-de")) {
    w.push_back(adaptAnalysisTask<Alice3LutMaker<o2::track::PID::Deuteron>>(cfgc, TaskName{"alice3-lutmaker-deuteron"}));
  }
  if (cfgc.options().get<int>("lut-tr")) {
    w.push_back(adaptAnalysisTask<Alice3LutMaker<o2::track::PID::Triton>>(cfgc, TaskName{"alice3-lutmaker-triton"}));
  }
  if (cfgc.options().get<int>("lut-he")) {
    w.push_back(adaptAnalysisTask<Alice3LutMaker<o2::track::PID::Helium3>>(cfgc, TaskName{"alice3-lutmaker-helium3"}));
  }
  return w;
}
