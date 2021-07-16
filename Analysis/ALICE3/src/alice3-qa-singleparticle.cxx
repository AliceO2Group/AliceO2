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

/// \author Nicolo' Jacazio <nicolo.jacazio@cern.ch>, CERN

// O2 includes
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "AnalysisCore/MC.h"
#include "Framework/HistogramRegistry.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct Alice3SingleParticle {
  Configurable<int> PDG{"PDG", 2212, "PDG code of the particle of interest"};
  Configurable<int> IsStable{"IsStable", 0, "Flag to check stable particles"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};
  Configurable<int> ptBins{"pt-bins", 500, "Number of pT bins"};
  Configurable<float> ptMin{"pt-min", 0.f, "Lower limit in pT"};
  Configurable<float> ptMax{"pt-max", 5.f, "Upper limit in pT"};
  Configurable<int> etaBins{"eta-bins", 500, "Number of eta bins"};
  Configurable<float> etaMin{"eta-min", -3.f, "Lower limit in eta"};
  Configurable<float> etaMax{"eta-max", 3.f, "Upper limit in eta"};
  Configurable<float> yMin{"y-min", -3.f, "Lower limit in y"};
  Configurable<float> yMax{"y-max", 3.f, "Upper limit in y"};
  Configurable<int> prodBins{"prod-bins", 100, "Number of production vertex bins"};
  Configurable<float> prodMin{"prod-min", -1.f, "Lower limit in production vertex"};
  Configurable<float> prodMax{"prod-max", 1.f, "Upper limit in production vertex"};
  Configurable<float> charge{"charge", 1.f, "Particle charge to scale the reconstructed momentum"};
  Configurable<bool> doPrint{"doPrint", false, "Flag to print debug messages"};

  void init(InitContext&)
  {
    const TString tit = Form("%i", PDG.value);
    AxisSpec axisPt{ptBins, ptMin, ptMax};
    AxisSpec axisEta{etaBins, etaMin, etaMax};
    AxisSpec axisProd{prodBins, prodMin, prodMax};

    histos.add("particlePt", "Particle Pt " + tit + ";#it{p}_{T} (GeV/#it{c})", kTH1D, {axisPt});
    histos.add("prodVx", "Particle Prod. Vertex X " + tit + ";Prod. Vertex X (cm)", kTH1D, {axisProd});
    histos.add("prodVy", "Particle Prod. Vertex Y " + tit + ";Prod. Vertex Y (cm)", kTH1D, {axisProd});
    histos.add("prodVz", "Particle Prod. Vertex Z " + tit + ";Prod. Vertex Z (cm)", kTH1D, {axisProd});
    histos.add("prodRadius", "Particle Prod. Vertex Radius " + tit + ";Prod. Vertex Radius (cm)", kTH1D, {axisProd});
    histos.add("prodVxVsPt", "Particle Prod. Vertex X " + tit + ";#it{p}_{T} (GeV/#it{c});Prod. Vertex X (cm)", kTH2D, {axisPt, axisProd});
    histos.add("prodVyVsPt", "Particle Prod. Vertex Y " + tit + ";#it{p}_{T} (GeV/#it{c});Prod. Vertex Y (cm)", kTH2D, {axisPt, axisProd});
    histos.add("prodVzVsPt", "Particle Prod. Vertex Z " + tit + ";#it{p}_{T} (GeV/#it{c});Prod. Vertex Z (cm)", kTH2D, {axisPt, axisProd});
    histos.add("prodRadiusVsPt", "Particle Prod. Vertex Radius " + tit + ";#it{p}_{T} (GeV/#it{c});Prod. Vertex Radius (cm)", kTH2D, {axisPt, axisProd});
    histos.add("prodRadius3DVsPt", "Particle Prod. Vertex Radius XYZ " + tit + ";#it{p}_{T} (GeV/#it{c});Prod. Vertex Radius XYZ (cm)", kTH2D, {axisPt, axisProd});
    histos.add("trackPt", "Track Pt " + tit + ";#it{p}_{T} (GeV/#it{c})", kTH1D, {axisPt});
    histos.add("particleEta", "Particle Eta " + tit + ";#it{#eta}", kTH1D, {axisEta});
    histos.add("trackEta", "Track Eta " + tit + ";#it{#eta}", kTH1D, {axisEta});
    histos.add("particleY", "Particle Y " + tit + ";#it{y}", kTH1D, {axisEta});
    histos.add("primaries", "Source for primaries " + tit + ";PDG Code", kTH1D, {{100, 0.f, 100.f}});
    histos.add("secondaries", "Source for secondaries " + tit + ";PDG Code", kTH1D, {{100, 0.f, 100.f}});
  }

  void process(const soa::Join<o2::aod::Tracks, o2::aod::McTrackLabels>& tracks,
               const aod::McParticles& mcParticles)
  {

    std::vector<int64_t> ParticlesOfInterest;
    for (const auto& mcParticle : mcParticles) {
      if (mcParticle.pdgCode() != PDG) {
        continue;
      }
      if (mcParticle.y() < yMin || mcParticle.y() > yMax) {
        continue;
      }
      histos.fill(HIST("particlePt"), mcParticle.pt());
      histos.fill(HIST("particleEta"), mcParticle.eta());
      histos.fill(HIST("particleY"), mcParticle.y());
      histos.fill(HIST("prodVx"), mcParticle.vx());
      histos.fill(HIST("prodVy"), mcParticle.vy());
      histos.fill(HIST("prodVz"), mcParticle.vz());
      histos.fill(HIST("prodRadius"), std::sqrt(mcParticle.vx() * mcParticle.vx() + mcParticle.vy() * mcParticle.vy()));
      histos.fill(HIST("prodVxVsPt"), mcParticle.pt(), mcParticle.vx());
      histos.fill(HIST("prodVyVsPt"), mcParticle.pt(), mcParticle.vy());
      histos.fill(HIST("prodVzVsPt"), mcParticle.pt(), mcParticle.vz());
      histos.fill(HIST("prodRadiusVsPt"), mcParticle.pt(), std::sqrt(mcParticle.vx() * mcParticle.vx() + mcParticle.vy() * mcParticle.vy()));
      histos.fill(HIST("prodRadius3DVsPt"), mcParticle.pt(), std::sqrt(mcParticle.vx() * mcParticle.vx() + mcParticle.vy() * mcParticle.vy() + mcParticle.vz() * mcParticle.vz()));
      ParticlesOfInterest.push_back(mcParticle.globalIndex());
    }

    for (const auto& track : tracks) {
      const auto mcParticle = track.mcParticle();
      if (!IsStable) {
        if (mcParticle.mother0() < 0) {
          continue;
        }
        auto mother = mcParticles.iteratorAt(mcParticle.mother0());
        const auto ParticleIsInteresting = std::find(ParticlesOfInterest.begin(), ParticlesOfInterest.end(), mother.globalIndex()) != ParticlesOfInterest.end();
        if (!ParticleIsInteresting) {
          continue;
        }
        if (doPrint) {
          LOG(INFO) << "Track " << track.globalIndex() << " comes from a " << mother.pdgCode() << " and is a " << mcParticle.pdgCode();
        }
      } else {
        if (mcParticle.pdgCode() != PDG) {
          continue;
        }
        histos.fill(HIST("trackPt"), track.pt() * charge);
        histos.fill(HIST("trackEta"), track.eta());
        if (mcParticle.mother0() < 0) {
          if (doPrint) {
            LOG(INFO) << "Track " << track.globalIndex() << " is a " << mcParticle.pdgCode();
          }
          continue;
        }
        auto mother = mcParticles.iteratorAt(mcParticle.mother0());
        if (MC::isPhysicalPrimary(mcParticles, mcParticle)) {
          histos.get<TH1>(HIST("primaries"))->Fill(Form("%i", mother.pdgCode()), 1.f);
        } else {
          histos.get<TH1>(HIST("secondaries"))->Fill(Form("%i", mother.pdgCode()), 1.f);
        }
        if (doPrint) {
          LOG(INFO) << "Track " << track.globalIndex() << " is a " << mcParticle.pdgCode() << " and comes from a " << mother.pdgCode() << " and is " << (MC::isPhysicalPrimary(mcParticles, mcParticle) ? "" : "not") << " a primary";
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<Alice3SingleParticle>(cfgc)};
}
