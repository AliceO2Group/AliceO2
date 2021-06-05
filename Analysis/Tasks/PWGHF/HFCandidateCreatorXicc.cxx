// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFCandidateCreatorXicc.cxx
/// \brief Reconstruction of Xiccplusplus candidates
/// \note Extended from HFCandidateCreator2Prong, HFCandidateCreator3Prong, HFCandidateCreatorX
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Luigi Dello Stritto <luigi.dello.stritto@cern.ch >, SALERNO
/// \author Mattia Faggin <mattia.faggin@cern.ch>, University and INFN PADOVA

#include "Framework/AnalysisTask.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisCore/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/V0.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand;
using namespace o2::aod::hf_cand_xicc;
using namespace o2::framework::expressions; //FIXME not sure if this is needed

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Perform MC matching."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// Reconstruction of xicc candidates
struct HfCandidateCreatorXicc {
  Produces<aod::HfCandXiccBase> rowCandidateBase;

  Configurable<double> magneticField{"d_bz", 5., "magnetic field"};
  Configurable<bool> b_propdca{"b_propdca", true, "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200., "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4., "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9, "stop iterations is chi2/chi2old > this"};
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do validation plots"};

  OutputObj<TH1F> hmassXic{TH1F("hmassXic", "xic candidates;inv. mass (#pi K #pi) (GeV/#it{c}^{2});entries", 500, 1.6, 2.6)};
  OutputObj<TH1F> hCovPVXX{TH1F("hCovPVXX", "3-prong candidates;XX element of cov. matrix of prim. vtx position (cm^{2});entries", 100, 0., 1.e-4)};
  OutputObj<TH1F> hCovSVXX{TH1F("hCovSVXX", "3-prong candidates;XX element of cov. matrix of sec. vtx position (cm^{2});entries", 100, 0., 0.2)};

  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massK = RecoDecay::getMassPDG(kKPlus);
  double massXic = RecoDecay::getMassPDG(int(pdg::Code::kXiCPlus));
  double massXicc{0.};

  Configurable<int> d_selectionFlagXic{"d_selectionFlagXic", 1, "Selection Flag for Xic"};
  Configurable<double> cutPtPionMin{"cutPtPionMin", 1., "min. pt pion track"};
  Filter filterSelectCandidates = (aod::hf_selcandidate_xic::isSelXicToPKPi >= d_selectionFlagXic || aod::hf_selcandidate_xic::isSelXicToPiKP >= d_selectionFlagXic);

  void process(aod::Collision const& collision,
               soa::Filtered<soa::Join<aod::HfCandProng3, aod::HFSelXicToPKPiCandidate>> const& xicCands,
               aod::BigTracks const& tracks)
  {
    // 3-prong vertex fitter to rebuild the Xic vertex
    o2::vertexing::DCAFitterN<3> df3;
    df3.setBz(magneticField);
    df3.setPropagateToPCA(b_propdca);
    df3.setMaxR(d_maxr);
    df3.setMaxDZIni(d_maxdzini);
    df3.setMinParamChange(d_minparamchange);
    df3.setMinRelChi2Change(d_minrelchi2change);
    df3.setUseAbsDCA(true);

    // 2-prong vertex fitter to build the Xicc vertex
    o2::vertexing::DCAFitterN<2> df2;
    df2.setBz(magneticField);
    df2.setPropagateToPCA(b_propdca);
    df2.setMaxR(d_maxr);
    df2.setMaxDZIni(d_maxdzini);
    df2.setMinParamChange(d_minparamchange);
    df2.setMinRelChi2Change(d_minrelchi2change);
    df2.setUseAbsDCA(true);

    for (auto& xicCand : xicCands) {
      if (!(xicCand.hfflag() & 1 << o2::aod::hf_cand_prong3::XicToPKPi)) {
        continue;
      }
      if (xicCand.isSelXicToPKPi() >= d_selectionFlagXic) {
        hmassXic->Fill(InvMassXicToPKPi(xicCand), xicCand.pt());
      }
      if (xicCand.isSelXicToPiKP() >= d_selectionFlagXic) {
        hmassXic->Fill(InvMassXicToPiKP(xicCand), xicCand.pt());
      }
      auto track0 = xicCand.index0_as<aod::BigTracks>();
      auto track1 = xicCand.index1_as<aod::BigTracks>();
      auto track2 = xicCand.index2_as<aod::BigTracks>();
      auto trackParVar0 = getTrackParCov(track0);
      auto trackParVar1 = getTrackParCov(track1);
      auto trackParVar2 = getTrackParCov(track2);
      auto collision = track0.collision();

      // reconstruct the 3-prong secondary vertex
      if (df3.process(trackParVar0, trackParVar1, trackParVar2) == 0) {
        continue;
      }
      const auto& secondaryVertex = df3.getPCACandidate();
      trackParVar0.propagateTo(secondaryVertex[0], magneticField);
      trackParVar1.propagateTo(secondaryVertex[0], magneticField);
      trackParVar2.propagateTo(secondaryVertex[0], magneticField);

      array<float, 3> pvecpK = {track0.px() + track1.px(), track0.py() + track1.py(), track0.pz() + track1.pz()};
      array<float, 3> pvecxic = {pvecpK[0] + track2.px(), pvecpK[1] + track2.py(), pvecpK[2] + track2.pz()};
      auto trackpK = o2::dataformats::V0(df3.getPCACandidatePos(), pvecpK, df3.calcPCACovMatrixFlat(),
                                         trackParVar0, trackParVar1, {0, 0}, {0, 0});
      auto trackxic = o2::dataformats::V0(df3.getPCACandidatePos(), pvecxic, df3.calcPCACovMatrixFlat(),
                                          trackpK, trackParVar2, {0, 0}, {0, 0});

      int index0Xic = track0.globalIndex();
      int index1Xic = track1.globalIndex();
      int index2Xic = track2.globalIndex();
      int charge = track0.sign() + track1.sign() + track2.sign();

      for (auto& trackpion : tracks) {
        if (trackpion.pt() < cutPtPionMin) {
          continue;
        }
        if (trackpion.sign() * charge < 0) {
          continue;
        }
        if (trackpion.globalIndex() == index0Xic || trackpion.globalIndex() == index1Xic || trackpion.globalIndex() == index2Xic) {
          continue;
        }
        array<float, 3> pvecpion;
        auto trackParVarPi = getTrackParCov(trackpion);

        // reconstruct the 3-prong X vertex
        if (df2.process(trackxic, trackParVarPi) == 0) {
          continue;
        }

        // calculate relevant properties
        const auto& secondaryVertexXicc = df2.getPCACandidate();
        auto chi2PCA = df2.getChi2AtPCACandidate();
        auto covMatrixPCA = df2.calcPCACovMatrix().Array();

        df2.propagateTracksToVertex();
        df2.getTrack(0).getPxPyPzGlo(pvecxic);
        df2.getTrack(1).getPxPyPzGlo(pvecpion);

        auto primaryVertex = getPrimaryVertex(collision);
        auto covMatrixPV = primaryVertex.getCov();
        o2::dataformats::DCA impactParameter0;
        o2::dataformats::DCA impactParameter1;
        trackxic.propagateToDCA(primaryVertex, magneticField, &impactParameter0);
        trackParVarPi.propagateToDCA(primaryVertex, magneticField, &impactParameter1);

        // get uncertainty of the decay length
        double phi, theta;
        getPointDirection(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertexXicc, phi, theta);
        auto errorDecayLength = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, theta) + getRotatedCovMatrixXX(covMatrixPCA, phi, theta));
        auto errorDecayLengthXY = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, 0.) + getRotatedCovMatrixXX(covMatrixPCA, phi, 0.));

        int hfFlag = 1 << DecayType::XiccToXicPi;

        rowCandidateBase(collision.globalIndex(),
                         collision.posX(), collision.posY(), collision.posZ(),
                         secondaryVertexXicc[0], secondaryVertexXicc[1], secondaryVertexXicc[2],
                         errorDecayLength, errorDecayLengthXY,
                         chi2PCA,
                         pvecxic[0], pvecxic[1], pvecxic[2],
                         pvecpion[0], pvecpion[1], pvecpion[2],
                         impactParameter0.getY(), impactParameter1.getY(),
                         std::sqrt(impactParameter0.getSigmaY2()), std::sqrt(impactParameter1.getSigmaY2()),
                         xicCand.globalIndex(), trackpion.globalIndex(),
                         hfFlag);
      } // if on selected Xicc
    }   // loop over candidates
  }     // end of process
};      //end of struct

/// Extends the base table with expression columns.
struct HfCandidateCreatorXiccExpressions {
  Spawns<aod::HfCandXiccExt> rowCandidateXicc;
  void init(InitContext const&) {}
};

/// Performs MC matching.
struct HfCandidateCreatorXiccMc {
  Produces<aod::HfCandXiccMCRec> rowMCMatchRec;
  Produces<aod::HfCandXiccMCGen> rowMCMatchGen;

  void process(aod::HfCandXicc const& candidates,
               aod::HfCandProng3,
               aod::BigTracksMC const& tracks,
               aod::McParticles const& particlesMC)
  {
    int indexRec = -1;
    int8_t sign = 0;
    int8_t flag = 0;
    int8_t origin = 0;

    // Match reconstructed candidates.
    for (auto& candidate : candidates) {
      //Printf("New rec. candidate");
      flag = 0;
      origin = 0;

      auto xicCand = candidate.index0();
      auto arrayDaughters = array{xicCand.index0_as<aod::BigTracksMC>(),
                                  xicCand.index1_as<aod::BigTracksMC>(),
                                  xicCand.index2_as<aod::BigTracksMC>(),
                                  candidate.index1_as<aod::BigTracksMC>()};

      // Ξcc±± → p± K∓ π± π±
      //Printf("Checking Ξcc±± → p± K∓ π± π±");
      indexRec = RecoDecay::getMatchedMCRec(particlesMC, arrayDaughters, pdg::Code::kXiCCPlusPlus, array{+kProton, -kKPlus, +kPiPlus, +kPiPlus}, true, &sign, 3);
      if (indexRec > -1) {
        flag = 1 << DecayType::XiccToXicPi;
      }

      // Check whether the particle is non-prompt (from a b quark).
      if (flag != 0) {
        auto particle = particlesMC.iteratorAt(indexRec);
        origin = (RecoDecay::getMother(particlesMC, particle, kBottom, true) > -1 ? OriginType::NonPrompt : OriginType::Prompt);
      }

      rowMCMatchRec(flag, origin);
    }

    // Match generated particles.
    for (auto& particle : particlesMC) {
      //Printf("New gen. candidate");
      flag = 0;
      origin = 0;
      // Xicc → Xic + π+
      if (RecoDecay::isMatchedMCGen(particlesMC, particle, pdg::Code::kXiCCPlusPlus, array{int(pdg::Code::kXiCPlus), +kPiPlus}, true)) {
        // Match Xic -> pKπ
        std::vector<int> arrDaughter;
        RecoDecay::getDaughters(particlesMC, particle, &arrDaughter, array{int(pdg::Code::kXiCPlus)}, 1);
        auto XicCandMC = particlesMC.iteratorAt(arrDaughter[0]);
        //Printf("Checking Ξc± → p± K∓ π±");
        if (RecoDecay::isMatchedMCGen(particlesMC, XicCandMC, int(pdg::Code::kXiCPlus), array{+kProton, -kKPlus, +kPiPlus}, true, &sign)) {
          flag = sign * (1 << DecayType::XiccToXicPi);
        }
      }
      // Check whether the particle is non-prompt (from a b quark).
      if (flag != 0) {
        origin = (RecoDecay::getMother(particlesMC, particle, kBottom, true) > -1 ? OriginType::NonPrompt : OriginType::Prompt);
      }
      rowMCMatchGen(flag, origin);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<HfCandidateCreatorXicc>(cfgc),
    adaptAnalysisTask<HfCandidateCreatorXiccExpressions>(cfgc)};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<HfCandidateCreatorXiccMc>(cfgc));
  }
  return workflow;
}
