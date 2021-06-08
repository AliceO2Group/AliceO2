// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFCandidateCreator3Prong.cxx
/// \brief Reconstruction of heavy-flavour 3-prong decay candidates
/// \note Extended from HFCandidateCreator2Prong
///
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/AnalysisTask.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisCore/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand;
using namespace o2::aod::hf_cand_prong3;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, true, {"Perform MC matching."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// Reconstruction of heavy-flavour 3-prong decay candidates
struct HFCandidateCreator3Prong {
  Produces<aod::HfCandProng3Base> rowCandidateBase;

  Configurable<double> magneticField{"d_bz", 5., "magnetic field"};
  Configurable<bool> b_propdca{"b_propdca", true, "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200., "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4., "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9, "stop iterations is chi2/chi2old > this"};
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do validation plots"};

  OutputObj<TH1F> hmass3{TH1F("hmass3", "3-prong candidates;inv. mass (#pi K #pi) (GeV/#it{c}^{2});entries", 500, 1.6, 2.1)};
  OutputObj<TH1F> hCovPVXX{TH1F("hCovPVXX", "3-prong candidates;XX element of cov. matrix of prim. vtx position (cm^{2});entries", 100, 0., 1.e-4)};
  OutputObj<TH1F> hCovSVXX{TH1F("hCovSVXX", "3-prong candidates;XX element of cov. matrix of sec. vtx position (cm^{2});entries", 100, 0., 0.2)};

  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massK = RecoDecay::getMassPDG(kKPlus);
  double massPiKPi{0.};

  void process(aod::Collisions const& collisions,
               aod::HfTrackIndexProng3 const& rowsTrackIndexProng3,
               aod::BigTracks const& tracks)
  {
    // 3-prong vertex fitter
    o2::vertexing::DCAFitterN<3> df;
    df.setBz(magneticField);
    df.setPropagateToPCA(b_propdca);
    df.setMaxR(d_maxr);
    df.setMaxDZIni(d_maxdzini);
    df.setMinParamChange(d_minparamchange);
    df.setMinRelChi2Change(d_minrelchi2change);
    df.setUseAbsDCA(true);

    // loop over triplets of track indices
    for (const auto& rowTrackIndexProng3 : rowsTrackIndexProng3) {
      auto track0 = rowTrackIndexProng3.index0_as<aod::BigTracks>();
      auto track1 = rowTrackIndexProng3.index1_as<aod::BigTracks>();
      auto track2 = rowTrackIndexProng3.index2_as<aod::BigTracks>();
      auto trackParVar0 = getTrackParCov(track0);
      auto trackParVar1 = getTrackParCov(track1);
      auto trackParVar2 = getTrackParCov(track2);
      auto collision = track0.collision();

      // reconstruct the 3-prong secondary vertex
      if (df.process(trackParVar0, trackParVar1, trackParVar2) == 0) {
        continue;
      }
      const auto& secondaryVertex = df.getPCACandidate();
      auto chi2PCA = df.getChi2AtPCACandidate();
      auto covMatrixPCA = df.calcPCACovMatrix().Array();
      hCovSVXX->Fill(covMatrixPCA[0]); // FIXME: Calculation of errorDecayLength(XY) gives wrong values without this line.
      trackParVar0 = df.getTrack(0);
      trackParVar1 = df.getTrack(1);
      trackParVar2 = df.getTrack(2);

      // get track momenta
      array<float, 3> pvec0;
      array<float, 3> pvec1;
      array<float, 3> pvec2;
      trackParVar0.getPxPyPzGlo(pvec0);
      trackParVar1.getPxPyPzGlo(pvec1);
      trackParVar2.getPxPyPzGlo(pvec2);

      // get track impact parameters
      // This modifies track momenta!
      auto primaryVertex = getPrimaryVertex(collision);
      auto covMatrixPV = primaryVertex.getCov();
      hCovPVXX->Fill(covMatrixPV[0]);
      o2::dataformats::DCA impactParameter0;
      o2::dataformats::DCA impactParameter1;
      o2::dataformats::DCA impactParameter2;
      trackParVar0.propagateToDCA(primaryVertex, magneticField, &impactParameter0);
      trackParVar1.propagateToDCA(primaryVertex, magneticField, &impactParameter1);
      trackParVar2.propagateToDCA(primaryVertex, magneticField, &impactParameter2);

      // get uncertainty of the decay length
      double phi, theta;
      getPointDirection(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex, phi, theta);
      auto errorDecayLength = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, theta) + getRotatedCovMatrixXX(covMatrixPCA, phi, theta));
      auto errorDecayLengthXY = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, 0.) + getRotatedCovMatrixXX(covMatrixPCA, phi, 0.));

      // fill candidate table rows
      rowCandidateBase(collision.globalIndex(),
                       collision.posX(), collision.posY(), collision.posZ(),
                       secondaryVertex[0], secondaryVertex[1], secondaryVertex[2],
                       errorDecayLength, errorDecayLengthXY,
                       chi2PCA,
                       pvec0[0], pvec0[1], pvec0[2],
                       pvec1[0], pvec1[1], pvec1[2],
                       pvec2[0], pvec2[1], pvec2[2],
                       impactParameter0.getY(), impactParameter1.getY(), impactParameter2.getY(),
                       std::sqrt(impactParameter0.getSigmaY2()), std::sqrt(impactParameter1.getSigmaY2()), std::sqrt(impactParameter2.getSigmaY2()),
                       rowTrackIndexProng3.index0Id(), rowTrackIndexProng3.index1Id(), rowTrackIndexProng3.index2Id(),
                       rowTrackIndexProng3.hfflag());

      // fill histograms
      if (b_dovalplots) {
        // calculate invariant mass
        auto arrayMomenta = array{pvec0, pvec1, pvec2};
        massPiKPi = RecoDecay::M(std::move(arrayMomenta), array{massPi, massK, massPi});
        hmass3->Fill(massPiKPi);
      }
    }
  }
};

/// Extends the base table with expression columns.
struct HFCandidateCreator3ProngExpressions {
  Spawns<aod::HfCandProng3Ext> rowCandidateProng3;
  void init(InitContext const&) {}
};

/// Performs MC matching.
struct HFCandidateCreator3ProngMC {
  Produces<aod::HfCandProng3MCRec> rowMCMatchRec;
  Produces<aod::HfCandProng3MCGen> rowMCMatchGen;

  void process(aod::HfCandProng3 const& candidates,
               aod::BigTracksMC const& tracks,
               aod::McParticles const& particlesMC)
  {
    int indexRec = -1;
    int8_t sign = 0;
    int8_t flag = 0;
    int8_t origin = 0;
    int8_t channel = 0;
    std::vector<int> arrDaughIndex;
    std::array<int, 2> arrPDGDaugh;
    std::array<int, 2> arrPDGResonant1 = {kProton, 313};  // Λc± → p± K*
    std::array<int, 2> arrPDGResonant2 = {2224, kKPlus};  // Λc± → Δ(1232)±± K∓
    std::array<int, 2> arrPDGResonant3 = {3124, kPiPlus}; // Λc± → Λ(1520) π±

    // Match reconstructed candidates.
    for (auto& candidate : candidates) {
      //Printf("New rec. candidate");
      flag = 0;
      origin = 0;
      channel = 0;
      arrDaughIndex.clear();
      auto arrayDaughters = array{candidate.index0_as<aod::BigTracksMC>(), candidate.index1_as<aod::BigTracksMC>(), candidate.index2_as<aod::BigTracksMC>()};

      // D± → π± K∓ π±
      //Printf("Checking D± → π± K∓ π±");
      indexRec = RecoDecay::getMatchedMCRec(particlesMC, arrayDaughters, pdg::Code::kDPlus, array{+kPiPlus, -kKPlus, +kPiPlus}, true, &sign);
      if (indexRec > -1) {
        flag = sign * (1 << DecayType::DPlusToPiKPi);
      }

      // Λc± → p± K∓ π±
      if (flag == 0) {
        //Printf("Checking Λc± → p± K∓ π±");
        indexRec = RecoDecay::getMatchedMCRec(particlesMC, arrayDaughters, pdg::Code::kLambdaCPlus, array{+kProton, -kKPlus, +kPiPlus}, true, &sign, 2);
        if (indexRec > -1) {
          flag = sign * (1 << DecayType::LcToPKPi);

          //Printf("Flagging the different Λc± → p± K∓ π± decay channels");
          RecoDecay::getDaughters(particlesMC, particlesMC.iteratorAt(indexRec), &arrDaughIndex, array{0}, 1);
          if (arrDaughIndex.size() == 2) {
            for (auto iProng = 0; iProng < arrDaughIndex.size(); ++iProng) {
              auto daughI = particlesMC.iteratorAt(arrDaughIndex[iProng]);
              arrPDGDaugh[iProng] = std::abs(daughI.pdgCode());
            }
            if (arrPDGDaugh[0] == arrPDGResonant1[0] && arrPDGDaugh[1] == arrPDGResonant1[1]) {
              channel = 1;
            } else if (arrPDGDaugh[0] == arrPDGResonant2[0] && arrPDGDaugh[1] == arrPDGResonant2[1]) {
              channel = 2;
            } else if (arrPDGDaugh[0] == arrPDGResonant3[0] && arrPDGDaugh[1] == arrPDGResonant3[1]) {
              channel = 3;
            }
          }
        }
      }

      // Ξc± → p± K∓ π±
      if (flag == 0) {
        //Printf("Checking Ξc± → p± K∓ π±");
        indexRec = RecoDecay::getMatchedMCRec(particlesMC, std::move(arrayDaughters), pdg::Code::kXiCPlus, array{+kProton, -kKPlus, +kPiPlus}, true, &sign);
        if (indexRec > -1) {
          flag = sign * (1 << DecayType::XicToPKPi);
        }
      }

      // Check whether the particle is non-prompt (from a b quark).
      if (flag != 0) {
        auto particle = particlesMC.iteratorAt(indexRec);
        origin = (RecoDecay::getMother(particlesMC, particle, kBottom, true) > -1 ? OriginType::NonPrompt : OriginType::Prompt);
      }

      rowMCMatchRec(flag, origin, channel);
    }

    // Match generated particles.
    for (auto& particle : particlesMC) {
      //Printf("New gen. candidate");
      flag = 0;
      origin = 0;
      channel = 0;
      arrDaughIndex.clear();

      // D± → π± K∓ π±
      //Printf("Checking D± → π± K∓ π±");
      if (RecoDecay::isMatchedMCGen(particlesMC, particle, pdg::Code::kDPlus, array{+kPiPlus, -kKPlus, +kPiPlus}, true, &sign)) {
        flag = sign * (1 << DecayType::DPlusToPiKPi);
      }

      // Λc± → p± K∓ π±
      if (flag == 0) {
        //Printf("Checking Λc± → p± K∓ π±");
        if (RecoDecay::isMatchedMCGen(particlesMC, particle, pdg::Code::kLambdaCPlus, array{+kProton, -kKPlus, +kPiPlus}, true, &sign, 2)) {
          flag = sign * (1 << DecayType::LcToPKPi);

          //Printf("Flagging the different Λc± → p± K∓ π± decay channels");
          RecoDecay::getDaughters(particlesMC, particle, &arrDaughIndex, array{0}, 1);
          if (arrDaughIndex.size() == 2) {
            for (auto jProng = 0; jProng < arrDaughIndex.size(); ++jProng) {
              auto daughJ = particlesMC.iteratorAt(arrDaughIndex[jProng]);
              arrPDGDaugh[jProng] = std::abs(daughJ.pdgCode());
            }
            if (arrPDGDaugh[0] == arrPDGResonant1[0] && arrPDGDaugh[1] == arrPDGResonant1[1]) {
              channel = 1;
            } else if (arrPDGDaugh[0] == arrPDGResonant2[0] && arrPDGDaugh[1] == arrPDGResonant2[1]) {
              channel = 2;
            } else if (arrPDGDaugh[0] == arrPDGResonant3[0] && arrPDGDaugh[1] == arrPDGResonant3[1]) {
              channel = 3;
            }
          }
        }
      }

      // Ξc± → p± K∓ π±
      if (flag == 0) {
        //Printf("Checking Ξc± → p± K∓ π±");
        if (RecoDecay::isMatchedMCGen(particlesMC, particle, pdg::Code::kXiCPlus, array{+kProton, -kKPlus, +kPiPlus}, true, &sign)) {
          flag = sign * (1 << DecayType::XicToPKPi);
        }
      }

      // Check whether the particle is non-prompt (from a b quark).
      if (flag != 0) {
        origin = (RecoDecay::getMother(particlesMC, particle, kBottom, true) > -1 ? OriginType::NonPrompt : OriginType::Prompt);
      }

      rowMCMatchGen(flag, origin, channel);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<HFCandidateCreator3Prong>(cfgc, TaskName{"hf-cand-creator-3prong"}),
    adaptAnalysisTask<HFCandidateCreator3ProngExpressions>(cfgc, TaskName{"hf-cand-creator-3prong-expressions"})};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<HFCandidateCreator3ProngMC>(cfgc, TaskName{"hf-cand-creator-3prong-mc"}));
  }
  return workflow;
}
