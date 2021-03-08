// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFMCValidation.cxx
/// \brief MonteCarlo Validation Code -- Gen and Rec Level validation
///
/// \author Antonio Palasciano <antonio.palasciano@cern.ch>, Università degli Studi di Bari & INFN, Sezione di Bari
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/runDataProcessing.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::aod::hf_cand_prong3;
using namespace o2::framework::expressions;

/// Gen Level Validation
///
/// - Number of HF quarks produced per collision
/// - Number of D±      → π± K∓ π±        per collision
///             D*±     → π± K∓ π±,
///             D0(bar) → π± K∓,
///             Λc±     → p(bar) K∓ π±
/// - Momentum Conservation for these particles
struct ValidationGenLevel {
  HistogramRegistry registry{
    "registry",
    {{"hMomentumCheck", "Mom. Conservation (1 = true, 0 = false) (#it{#epsilon} = 1 MeV/#it{c}); Mom. Conservation result; entries", {HistType::kTH1F, {{2, -0.5, +1.5}}}},
     {"hPtDiffMotherDaughterGen", "Pt Difference Mother-Daughters; #Delta#it{p}_{T}^{gen.} (GeV/#it{c}); entries", {HistType::kTH1F, {{100, -0.01, 0.01}}}},
     {"hPxDiffMotherDaughterGen", "Px Difference Mother-Daughters; #Delta#it{p}_{x}^{gen.} (GeV/#it{c}); entries", {HistType::kTH1F, {{100, -0.01, 0.01}}}},
     {"hPyDiffMotherDaughterGen", "Py Difference Mother-Daughters; #Delta#it{p}_{y}^{gen.} (GeV/#it{c}); entries", {HistType::kTH1F, {{100, -0.01, 0.01}}}},
     {"hPzDiffMotherDaughterGen", "Pz Difference Mother-Daughters; #Delta#it{p}_{z}^{gen.} (GeV/#it{c}); entries", {HistType::kTH1F, {{100, -0.01, 0.01}}}},
     {"hPdiffMotherDaughterGen", "P  Difference Mother-Daughters; #Delta#it{p}^{gen.} (GeV/#it{c}); entries", {HistType::kTH1F, {{100, -0.01, 0.01}}}},
     {"hCountAverageC", "Event counter - Average Number Charm quark; Events Per Collision; entries", {HistType::kTH1F, {{20, 0., 20.}}}},
     {"hCountAverageB", "Event counter - Average Number Beauty quark; Events Per Collision; entries", {HistType::kTH1F, {{20, 0., 20.}}}},
     {"hCountAverageCbar", "Event counter - Average Number Anti-Charm quark; Events Per Collision; entries", {HistType::kTH1F, {{20, 0., 20.}}}},
     {"hCountAverageBbar", "Event counter - Average Number Anti-Beauty quark; Events Per Collision; entries", {HistType::kTH1F, {{20, 0., 20.}}}},
     {"hCouterPerCollisionDzero", "Event counter - D0; Events Per Collision; entries", {HistType::kTH1F, {{10, -0.5, +9.5}}}},
     {"hCouterPerCollisionDplus", "Event counter - DPlus; Events Per Collision; entries", {HistType::kTH1F, {{10, -0.5, +9.5}}}},
     {"hCouterPerCollisionDstar", "Event counter - Dstar; Events Per Collision; entries", {HistType::kTH1F, {{10, -0.5, +9.5}}}},
     {"hCouterPerCollisionLambdaC", "Event counter - LambdaC; Events Per Collision; entries", {HistType::kTH1F, {{10, -0.5, +9.5}}}}}};

  void process(aod::McCollision const& mccollision, aod::McParticles const& particlesMC)
  {
    int cPerCollision = 0;
    int cBarPerCollision = 0;
    int bPerCollision = 0;
    int bBarPerCollision = 0;
    double sumPxDau, sumPyDau, sumPzDau;
    bool momentumCheck;
    double pxDiff, pyDiff, pzDiff;

    //Particles and their decay checked in the second part of the task
    std::array<int, 4> PDGArrayParticle = {411, 413, 421, 4122};
    std::array<std::array<int, 3>, 4> arrPDGFinal = {{{211, 211, -321}, {211, 211, -321}, {-321, 211, 0}, {2212, -321, 211}}};
    int counter[4] = {0, 0, 0, 0};
    std::vector<int> listDaughters;

    for (auto& particle : particlesMC) {
      int particlePdgCode = particle.pdgCode();
      if (particle.mother0() < 0) {
        continue;
      }
      auto mother = particlesMC.iteratorAt(particle.mother0());
      if (particlePdgCode != mother.pdgCode()) {
        switch (particlePdgCode) {
          case 4:
            cPerCollision++;
            break;
          case -4:
            cBarPerCollision++;
            break;
          case 5:
            bPerCollision++;
            break;
          case -5:
            bBarPerCollision++;
            break;
        }
      }

      sumPxDau = 0;
      sumPyDau = 0;
      sumPzDau = 0;
      momentumCheck = 1;
      listDaughters.clear();

      // Checking the decay of the particles and the momentum conservation
      for (int iD = 0; iD < PDGArrayParticle.size(); iD++) {
        if (std::abs(particlePdgCode) == PDGArrayParticle[iD]) {
          RecoDecay::getDaughters(particlesMC, particle.globalIndex(), &listDaughters, arrPDGFinal[iD], -1);
          int arrayPDGsize = arrPDGFinal[iD].size() - std::count(arrPDGFinal[iD].begin(), arrPDGFinal[iD].end(), 0);
          if (listDaughters.size() == arrayPDGsize) {
            counter[iD]++;
          }
          for (int i = 0; i < listDaughters.size(); i++) {
            auto daughter = particlesMC.iteratorAt(listDaughters.at(i));
            sumPxDau += daughter.px();
            sumPyDau += daughter.py();
            sumPzDau += daughter.pz();
          }
          pxDiff = particle.px() - sumPxDau;
          pyDiff = particle.py() - sumPyDau;
          pzDiff = particle.pz() - sumPzDau;
          if (std::abs(pxDiff) > 0.001 || std::abs(pyDiff) > 0.001 || std::abs(pzDiff) > 0.001) {
            momentumCheck = 0;
          }
          double pDiff = RecoDecay::P(pxDiff, pyDiff, pzDiff);
          double ptDiff = RecoDecay::Pt(pxDiff, pyDiff);
          //Filling histograms with per-component momentum conservation
          registry.fill(HIST("hMomentumCheck"), momentumCheck);
          registry.fill(HIST("hPxDiffMotherDaughterGen"), pxDiff);
          registry.fill(HIST("hPyDiffMotherDaughterGen"), pyDiff);
          registry.fill(HIST("hPzDiffMotherDaughterGen"), pzDiff);
          registry.fill(HIST("hPdiffMotherDaughterGen"), pDiff);
          registry.fill(HIST("hPtDiffMotherDaughterGen"), ptDiff);
        }
      }
    } //end particles
    registry.fill(HIST("hCountAverageC"), cPerCollision);
    registry.fill(HIST("hCountAverageB"), bPerCollision);
    registry.fill(HIST("hCountAverageCbar"), cBarPerCollision);
    registry.fill(HIST("hCountAverageBbar"), bBarPerCollision);
    registry.fill(HIST("hCouterPerCollisionDplus"), counter[0]);
    registry.fill(HIST("hCouterPerCollisionDstar"), counter[1]);
    registry.fill(HIST("hCouterPerCollisionDzero"), counter[2]);
    registry.fill(HIST("hCouterPerCollisionLambdaC"), counter[3]);
  }
};

/// Rec Level Validation
///
/// Only D0 matched candidates:
///   - Gen-Rec Level Momentum Difference per component;
///   - Gen-Rec Level Difference for secondary Vertex coordinates and decay length;
struct ValidationRecLevel {
  HistogramRegistry registry{
    "registry",
    {{"histPt", "Pt difference reco - MC; #it{p}_{T}^{reco} - #it{p}_{T}^{gen.} (GeV/#it{c}); entries", {HistType::kTH1F, {{2000, -1, 1}}}},
     {"histPx", "Px difference reco - MC; #it{p}_{x}^{reco} - #it{p}_{x}^{gen.} (GeV/#it{c}); entries", {HistType::kTH1F, {{2000, -1, 1}}}},
     {"histPy", "Py difference reco - MC; #it{p}_{y}^{reco} - #it{p}_{y}^{gen.} (GeV/#it{c}); entries", {HistType::kTH1F, {{2000, -1, 1}}}},
     {"histPz", "Pz difference reco - MC; #it{p}_{z}^{reco} - #it{p}_{z}^{gen.} (GeV/#it{c}); entries", {HistType::kTH1F, {{2000, -1, 1}}}},
     {"histSecVx", "Sec. Vertex difference reco - MC (MC matched); #Delta x (cm); entries", {HistType::kTH1F, {{200, -1, 1}}}},
     {"histSecVy", "Sec. Vertex difference reco - MC (MC matched); #Delta y (cm); entries", {HistType::kTH1F, {{200, -1, 1}}}},
     {"histSecVz", "Sec. Vertex difference reco - MC (MC matched); #Delta z (cm); entries", {HistType::kTH1F, {{200, -1, 1}}}},
     {"histDecLen", "Decay Length difference reco - MC (MC matched); #Delta L (cm); entries", {HistType::kTH1F, {{200, -1, 1}}}}}};

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 0, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 0, "Selection Flag for D0bar"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= d_selectionFlagD0bar);

  void process(soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate, aod::HfCandProng2MCRec>> const& candidates, aod::BigTracksMC const& tracks, aod::McParticles const& particlesMC)
  {
    int indexParticle = 0;
    double pxDiff, pyDiff, pzDiff, pDiff;
    double decayLength;
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      if (std::abs(candidate.flagMCMatchRec()) == 1 << D0ToPiK) {
        indexParticle = RecoDecay::getMother(particlesMC, candidate.index0_as<aod::BigTracksMC>().mcParticle(), 421, true);
        auto mother = particlesMC.iteratorAt(indexParticle);
        registry.fill(HIST("histPt"), candidate.pt() - mother.pt());
        registry.fill(HIST("histPx"), candidate.px() - mother.px());
        registry.fill(HIST("histPy"), candidate.py() - mother.py());
        registry.fill(HIST("histPz"), candidate.pz() - mother.pz());
        //Compare Secondary vertex and decay length with MC
        auto daughter0 = particlesMC.iteratorAt(mother.daughter0());
        double vertexDau[3] = {daughter0.vx(), daughter0.vy(), daughter0.vz()};
        double vertexMoth[3] = {mother.vx(), mother.vy(), mother.vz()};
        decayLength = RecoDecay::distance(vertexMoth, vertexDau);

        registry.fill(HIST("histSecVx"), candidate.xSecondaryVertex() - vertexDau[0]);
        registry.fill(HIST("histSecVy"), candidate.ySecondaryVertex() - vertexDau[1]);
        registry.fill(HIST("histSecVz"), candidate.zSecondaryVertex() - vertexDau[2]);
        registry.fill(HIST("histDecLen"), candidate.decayLength() - decayLength);
      }
    } //end loop on candidates
  }   //end process
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<ValidationGenLevel>(cfgc, "hf-mc-validation-gen"),
    adaptAnalysisTask<ValidationRecLevel>(cfgc, "hf-mc-validation-rec")};
  return workflow;
}
