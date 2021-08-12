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
    const AxisSpec axisVx{100, -1, 1, "Vtx_{x}"};
    const AxisSpec axisVy{100, -1, 1, "Vtx_{y}"};
    const AxisSpec axisVz{100, -20, 20, "Vtx_{z}"};
    const AxisSpec axisP{ptBins, ptMax, ptMax, "#it{p} (GeV/#it{c})"};
    const AxisSpec axisPt{ptBins, ptMin, ptMax, "#it{p}_{T} (GeV/#it{c})"};
    const AxisSpec axisPx{ptBins, -ptMax, ptMax, "#it{p}_{x} (GeV/#it{c})"};
    const AxisSpec axisPy{ptBins, -ptMax, ptMax, "#it{p}_{y} (GeV/#it{c})"};
    const AxisSpec axisPz{ptBins, -ptMax, ptMax, "#it{p}_{z} (GeV/#it{c})"};
    const AxisSpec axisEta{etaBins, etaMin, etaMax, "#it{#eta}"};
    const AxisSpec axisY{etaBins, etaMin, etaMax, "#it{y}"};
    const AxisSpec axisE{etaBins, 0, 1000, "E"};
    const AxisSpec axisProdx{prodBins, prodMin, prodMax, "Prod. Vertex X (cm)"};
    const AxisSpec axisPrody{prodBins, prodMin, prodMax, "Prod. Vertex Y (cm)"};
    const AxisSpec axisProdz{prodBins, prodMin, prodMax, "Prod. Vertex Z (cm)"};
    const AxisSpec axisProdRadius{prodBins, prodMin, prodMax, "Prod. Vertex Radius (cm)"};

    histos.add("event/VtxX", "Vertex X", kTH1D, {axisVx});
    histos.add("event/VtxY", "Vertex Y", kTH1D, {axisVy});
    histos.add("event/VtxZ", "Vertex Z", kTH1D, {axisVz});

    histos.add("particle/PDGs", "Particle PDGs;PDG Code", kTH1D, {{100, 0.f, 100.f}});
    histos.add("particle/Pt", "Particle Pt " + tit, kTH1D, {axisPt});
    histos.add("particle/prodVx", "Particle Prod. Vertex X " + tit, kTH1D, {axisProdx});
    histos.add("particle/prodVy", "Particle Prod. Vertex Y " + tit, kTH1D, {axisPrody});
    histos.add("particle/prodVz", "Particle Prod. Vertex Z " + tit, kTH1D, {axisProdz});
    histos.add("particle/prodRadius", "Particle Prod. Vertex Radius " + tit, kTH1D, {axisProdRadius});
    histos.add("particle/prodVxVsPt", "Particle Prod. Vertex X " + tit, kTH2D, {axisPt, axisProdx});
    histos.add("particle/prodVyVsPt", "Particle Prod. Vertex Y " + tit, kTH2D, {axisPt, axisPrody});
    histos.add("particle/prodVzVsPt", "Particle Prod. Vertex Z " + tit, kTH2D, {axisPt, axisProdz});
    histos.add("particle/prodRadiusVsPt", "Particle Prod. Vertex Radius " + tit, kTH2D, {axisPt, axisProdRadius});
    histos.add("particle/prodRadius3DVsPt", "Particle Prod. Vertex Radius XYZ " + tit, kTH2D, {axisPt, axisProdRadius});
    histos.add("particle/Eta", "Particle Eta " + tit, kTH1D, {axisEta});
    histos.add("particle/Y", "Particle Y " + tit, kTH1D, {axisY});
    histos.add("particle/EvsPz", "Particle E vs Pz " + tit, kTH2D, {axisE, axisPz});
    histos.add("particle/YvzPz", "Particle Y vs Pz " + tit, kTH2D, {axisY, axisPz});
    histos.add("particle/EtavzPz", "Particle Eta vs Pz " + tit, kTH2D, {axisEta, axisPz});
    histos.add("particle/PtvzPz", "Particle Pt vs Pz " + tit, kTH2D, {axisPt, axisPz});
    histos.add("particle/PvzPz", "Particle P vs Pz " + tit, kTH2D, {axisP, axisPz});
    histos.add("particle/Px", "Particle Px " + tit, kTH1D, {axisPx});
    histos.add("particle/Py", "Particle Py " + tit, kTH1D, {axisPy});
    histos.add("particle/Pz", "Particle Pz " + tit, kTH1D, {axisPz});

    histos.add("track/PDGs", "Track PDGs;PDG Code", kTH1D, {{100, 0.f, 100.f}});
    histos.add("track/tofPDGs", "Track wTOF PDGs;PDG Code", kTH1D, {{100, 0.f, 100.f}});
    histos.add("track/Pt", "Track Pt " + tit, kTH1D, {axisPt});
    histos.add("track/Eta", "Track Eta " + tit, kTH1D, {axisEta});
    histos.add("track/primaries", "Source for primaries " + tit + ";PDG Code", kTH1D, {{100, 0.f, 100.f}});
    histos.add("track/secondaries", "Source for secondaries " + tit + ";PDG Code", kTH1D, {{100, 0.f, 100.f}});
  }

  void process(const o2::aod::McCollisions& colls,
               const soa::Join<o2::aod::Tracks, o2::aod::McTrackLabels, o2::aod::TracksExtra>& tracks,
               const aod::McParticles& mcParticles)
  {
    for (const auto& col : colls) {
      histos.fill(HIST("event/VtxX"), col.posX());
      histos.fill(HIST("event/VtxY"), col.posY());
      histos.fill(HIST("event/VtxZ"), col.posZ());
    }
    std::vector<int64_t> ParticlesOfInterest;
    for (const auto& mcParticle : mcParticles) {
      histos.get<TH1>(HIST("particle/PDGs"))->Fill(Form("%i", mcParticle.pdgCode()), 1.f);
      if (mcParticle.pdgCode() != PDG) {
        continue;
      }
      if (mcParticle.y() < yMin || mcParticle.y() > yMax) {
        continue;
      }
      histos.fill(HIST("particle/Pt"), mcParticle.pt());
      histos.fill(HIST("particle/Eta"), mcParticle.eta());
      histos.fill(HIST("particle/EvsPz"), mcParticle.e(), mcParticle.pz());
      histos.fill(HIST("particle/Y"), mcParticle.y());
      histos.fill(HIST("particle/YvzPz"), mcParticle.y(), mcParticle.pz());
      histos.fill(HIST("particle/EtavzPz"), mcParticle.eta(), mcParticle.pz());
      histos.fill(HIST("particle/PvzPz"), mcParticle.p(), mcParticle.pz());
      histos.fill(HIST("particle/PtvzPz"), mcParticle.pt(), mcParticle.pz());
      histos.fill(HIST("particle/Px"), mcParticle.px());
      histos.fill(HIST("particle/Py"), mcParticle.py());
      histos.fill(HIST("particle/Pz"), mcParticle.pz());
      histos.fill(HIST("particle/prodVx"), mcParticle.vx());
      histos.fill(HIST("particle/prodVy"), mcParticle.vy());
      histos.fill(HIST("particle/prodVz"), mcParticle.vz());
      histos.fill(HIST("particle/prodRadius"), std::sqrt(mcParticle.vx() * mcParticle.vx() + mcParticle.vy() * mcParticle.vy()));
      histos.fill(HIST("particle/prodVxVsPt"), mcParticle.pt(), mcParticle.vx());
      histos.fill(HIST("particle/prodVyVsPt"), mcParticle.pt(), mcParticle.vy());
      histos.fill(HIST("particle/prodVzVsPt"), mcParticle.pt(), mcParticle.vz());
      histos.fill(HIST("particle/prodRadiusVsPt"), mcParticle.pt(), std::sqrt(mcParticle.vx() * mcParticle.vx() + mcParticle.vy() * mcParticle.vy()));
      histos.fill(HIST("particle/prodRadius3DVsPt"), mcParticle.pt(), std::sqrt(mcParticle.vx() * mcParticle.vx() + mcParticle.vy() * mcParticle.vy() + mcParticle.vz() * mcParticle.vz()));
      ParticlesOfInterest.push_back(mcParticle.globalIndex());
    }

    for (const auto& track : tracks) {
      const auto mcParticle = track.mcParticle();
      histos.get<TH1>(HIST("track/PDGs"))->Fill(Form("%i", mcParticle.pdgCode()), 1.f);
      if (track.hasTOF()) {
        histos.get<TH1>(HIST("track/tofPDGs"))->Fill(Form("%i", mcParticle.pdgCode()), 1.f);
      }
      if (!IsStable) {
        if (!mcParticle.has_mother0()) {
          continue;
        }
        auto mother = mcParticle.mother0_as<aod::McParticles>();
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
        histos.fill(HIST("track/Pt"), track.pt() * charge);
        histos.fill(HIST("track/Eta"), track.eta());
        if (!mcParticle.has_mother0()) {
          if (doPrint) {
            LOG(INFO) << "Track " << track.globalIndex() << " is a " << mcParticle.pdgCode();
          }
          continue;
        }
        auto mother = mcParticle.mother0_as<aod::McParticles>();
        if (MC::isPhysicalPrimary(mcParticle)) {
          histos.get<TH1>(HIST("track/primaries"))->Fill(Form("%i", mother.pdgCode()), 1.f);
        } else {
          histos.get<TH1>(HIST("track/secondaries"))->Fill(Form("%i", mother.pdgCode()), 1.f);
        }
        if (doPrint) {
          LOG(INFO) << "Track " << track.globalIndex() << " is a " << mcParticle.pdgCode() << " and comes from a " << mother.pdgCode() << " and is " << (MC::isPhysicalPrimary(mcParticle) ? "" : "not") << " a primary";
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<Alice3SingleParticle>(cfgc)};
}
