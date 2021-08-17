// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFCandidateCreatorBPlus.cxx
/// \brief Reconstruction of B± → D0bar(D0) π± candidates
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN
/// \author Deepa Thomas <deepa.thomas@cern.ch>, UT Austin
/// \author Antonio Palasciano <antonio.palasciano@cern.ch>, Università degli Studi di Bari & INFN, Sezione di Bari

#include "Framework/AnalysisTask.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisCore/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"
#include "AnalysisCore/HFSelectorCuts.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "ReconstructionDataFormats/V0.h"

using namespace o2;
using namespace o2::aod;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::analysis;
using namespace o2::aod::hf_cand;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::aod::hf_cand_bplus;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Perform MC matching."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// Reconstruction of B± → D0bar(D0) π± → (K± π∓) π±
struct HfCandidateCreatorBplus {
  Produces<aod::HfCandBPlusBase> rowCandidateBase;
  // vertexing parameters
  Configurable<double> bz{"bz", 5., "magnetic field"};
  Configurable<bool> propdca{"propdca", true, "create tracks version propagated to PCA"};
  Configurable<double> maxr{"maxr", 5., "reject PCA's above this radius"};
  Configurable<double> maxdzini{"maxdzini", 999, "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> minparamchange{"minparamchange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> minrelchi2change{"minrelchi2change", 0.9, "stop iterations if chi2/chi2old > this"};
  Configurable<bool> UseAbsDCA{"UseAbsDCA", true, "Use Abs DCAs"};

  OutputObj<TH1F> hCovPVXX{TH1F("hCovPVXX", "2-prong candidates;XX element of cov. matrix of prim. vtx. position (cm^{2});entries", 100, 0., 1.e-4)};
  OutputObj<TH1F> hCovSVXX{TH1F("hCovSVXX", "2-prong candidates;XX element of cov. matrix of sec. vtx. position (cm^{2});entries", 100, 0., 0.2)};
  OutputObj<TH1F> hNevents{TH1F("hNevents", "Number of events;Nevents;entries", 1, 0., 1)};
  OutputObj<TH1F> hD0Rapidity{TH1F("hD0Rapidity", "D0 candidates;#it{y};entries", 100, -2, 2)};
  OutputObj<TH1F> hPiEta{TH1F("hPiEta", "Pion track;#it{#eta};entries", 400, 2, 2)};

  Configurable<int> selectionFlagD0{"selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> selectionFlagD0bar{"selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutEtaTrkMax{"cutEtaTrkMax", -1, "max. bach track. pseudorapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= selectionFlagD0bar);

  void process(aod::Collision const& collisions,
               soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>> const& candidates, aod::BigTracks const& tracks)
  {
    hNevents->Fill(0);

    //Initialise fitter for B vertex
    o2::vertexing::DCAFitterN<2> bfitter;
    bfitter.setBz(bz);
    bfitter.setPropagateToPCA(propdca);
    bfitter.setMaxR(maxr);
    bfitter.setMinParamChange(minparamchange);
    bfitter.setMinRelChi2Change(minrelchi2change);
    bfitter.setUseAbsDCA(UseAbsDCA);

    //Initial fitter to redo D-vertex to get extrapolated daughter tracks
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(bz);
    df.setPropagateToPCA(propdca);
    df.setMaxR(maxr);
    df.setMinParamChange(minparamchange);
    df.setMinRelChi2Change(minrelchi2change);
    df.setUseAbsDCA(UseAbsDCA);

    // loop over pairs of track indices
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << hf_cand_prong2::DecayType::D0ToPiK)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YD0(candidate)) > cutYCandMax) {
        continue;
      }

      hD0Rapidity->Fill(YD0(candidate));

      const std::array<float, 3> vertexD0 = {candidate.xSecondaryVertex(), candidate.ySecondaryVertex(), candidate.zSecondaryVertex()};
      const std::array<float, 3> momentumD0 = {candidate.px(), candidate.py(), candidate.pz()};

      auto prong0 = candidate.index0_as<aod::BigTracks>();
      auto prong1 = candidate.index1_as<aod::BigTracks>();
      auto prong0TrackParCov = getTrackParCov(prong0);
      auto prong1TrackParCov = getTrackParCov(prong1);
      auto collision = prong0.collision();

      // LOGF(INFO, "All track: %d (prong0); %d (prong1)", candidate.index0().globalIndex(), candidate.index1().globalIndex());
      // LOGF(INFO, "All track pT: %f (prong0); %f (prong1)", prong0.pt(), prong1.pt());

      // reconstruct D0 secondary vertex
      if (df.process(prong0TrackParCov, prong1TrackParCov) == 0) {
        continue;
      }

      prong0TrackParCov.propagateTo(candidate.xSecondaryVertex(), bz);
      prong1TrackParCov.propagateTo(candidate.xSecondaryVertex(), bz);
      //std::cout << df.getTrack(0).getX() << " "<< "secVx=" << candidate.xSecondaryVertex() << std::endl;

      const std::array<float, 6> pCovMatrixD0 = df.calcPCACovMatrixFlat();
      // build a D0 neutral track
      auto trackD0 = o2::dataformats::V0(vertexD0, momentumD0, pCovMatrixD0, prong0TrackParCov, prong1TrackParCov, {0, 0}, {0, 0});

      //loop over tracks for pi selection
      auto count = 0;
      for (auto& track : tracks) {
        //if(count % 100 == 0){
        //LOGF(INFO, "Col: %d (cand); %d (track)", candidate.collisionId(), track.collisionId());
        //  count++;
        // }

        if (cutEtaTrkMax >= 0. && std::abs(track.eta()) > cutEtaTrkMax) {
          continue;
        }

        hPiEta->Fill(track.eta());

        if (candidate.index0Id() == track.globalIndex() || candidate.index1Id() == track.globalIndex()) {
          continue; //daughter track id and bachelor track id not the same
        }

        //Select D0pi- and D0(bar)pi+ pairs only
        if (!((candidate.isSelD0() >= selectionFlagD0 && track.sign() < 0) || (candidate.isSelD0bar() >= selectionFlagD0bar && track.sign() > 0))) {
          //Printf("D0: %d, D0bar%d, sign: %d", candidate.isSelD0(), candidate.isSelD0bar(), track.sign());
          continue;
        }

        auto trackBach = getTrackParCov(track);
        std::array<float, 3> pVecD0 = {0., 0., 0.};
        std::array<float, 3> pVecBach = {0., 0., 0.};
        std::array<float, 3> pVecBCand = {0., 0., 0.};

        //find the DCA between the D0 and the bachelor track, for B+
        if (bfitter.process(trackD0, trackBach) == 0) {
          continue;
        }

        bfitter.propagateTracksToVertex();          // propagate the bachelor and D0 to the B+ vertex
        bfitter.getTrack(0).getPxPyPzGlo(pVecD0);   //momentum of D0 at the B+ vertex
        bfitter.getTrack(1).getPxPyPzGlo(pVecBach); //momentum of pi+ at the B+ vertex
        const auto& BSecVertex = bfitter.getPCACandidate();
        auto chi2PCA = bfitter.getChi2AtPCACandidate();
        auto covMatrixPCA = bfitter.calcPCACovMatrix().Array();
        hCovSVXX->Fill(covMatrixPCA[0]); //FIXME: Calculation of errorDecayLength(XY) gives wrong values without this line.

        pVecBCand = RecoDecay::PVec(pVecD0, pVecBach);

        // get track impact parameters
        // This modifies track momenta!
        auto primaryVertex = getPrimaryVertex(collision);
        auto covMatrixPV = primaryVertex.getCov();
        hCovPVXX->Fill(covMatrixPV[0]);
        o2::dataformats::DCA impactParameter0;
        o2::dataformats::DCA impactParameter1;

        bfitter.getTrack(0).propagateToDCA(primaryVertex, bz, &impactParameter0);
        bfitter.getTrack(1).propagateToDCA(primaryVertex, bz, &impactParameter1);

        // get uncertainty of the decay length
        double phi, theta;
        getPointDirection(array{collision.posX(), collision.posY(), collision.posZ()}, BSecVertex, phi, theta);
        auto errorDecayLength = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, theta) + getRotatedCovMatrixXX(covMatrixPCA, phi, theta));
        auto errorDecayLengthXY = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, 0.) + getRotatedCovMatrixXX(covMatrixPCA, phi, 0.));

        int hfFlag = 1 << hf_cand_bplus::DecayType::BPlusToD0Pi;

        // fill candidate table rows
        rowCandidateBase(collision.globalIndex(),
                         collision.posX(), collision.posY(), collision.posZ(),
                         BSecVertex[0], BSecVertex[1], BSecVertex[2],
                         errorDecayLength, errorDecayLengthXY,
                         chi2PCA,
                         pVecD0[0], pVecD0[1], pVecD0[2],
                         pVecBach[0], pVecBach[1], pVecBach[2],
                         impactParameter0.getY(), impactParameter1.getY(),
                         std::sqrt(impactParameter0.getSigmaY2()), std::sqrt(impactParameter1.getSigmaY2()),
                         candidate.globalIndex(), track.globalIndex(), //index D0 and bachelor
                         hfFlag);
      } //track loop
    }   //D0 cand loop
  }     //process
};      //struct

/// Extends the base table with expression columns.
struct HfCandidateCreatorBplusExpressions {
  Spawns<aod::HfCandBPlusExt> rowCandidateBPlus;
  void init(InitContext const&) {}
};

/// Performs MC matching.
struct HfCandidateCreatorBplusMc {
  Produces<aod::HfCandBPMCRec> rowMCMatchRec;
  Produces<aod::HfCandBPMCGen> rowMCMatchGen;

  void process(aod::HfCandBPlus const& candidates,
               aod::HfCandProng2,
               aod::BigTracksMC const& tracks,
               aod::McParticles const& particlesMC)
  {
    int indexRec = -1, indexRecD0 = -1;
    int8_t signB = 0, signD0 = 0;
    int8_t flag = 0;
    int kD0pdg = pdg::Code::kD0;

    // Match reconstructed candidates.
    for (auto& candidate : candidates) {
      //Printf("New rec. candidate");

      flag = 0;
      auto candDaughterD0 = candidate.index0_as<aod::HfCandProng2>();
      auto arrayDaughtersD0 = array{candDaughterD0.index0_as<aod::BigTracksMC>(), candDaughterD0.index1_as<aod::BigTracksMC>()};
      auto arrayDaughters = array{candidate.index1_as<aod::BigTracksMC>(), candDaughterD0.index0_as<aod::BigTracksMC>(), candDaughterD0.index1_as<aod::BigTracksMC>()};

      // B± → D0bar(D0) π± → (K± π∓) π±
      //Printf("Checking B± → D0(bar) π±");
      indexRec = RecoDecay::getMatchedMCRec(particlesMC, arrayDaughters, pdg::Code::kBPlus, array{+kPiPlus, +kKPlus, -kPiPlus}, true, &signB, 2);
      indexRecD0 = RecoDecay::getMatchedMCRec(particlesMC, arrayDaughtersD0, pdg::Code::kD0, array{-kKPlus, +kPiPlus}, true, &signD0, 1);

      if (indexRecD0 > -1 && indexRec > -1) {
        flag = signB * (1 << hf_cand_bplus::DecayType::BPlusToD0Pi);
      }
      rowMCMatchRec(flag);
    }

    // Match generated particles.
    for (auto& particle : particlesMC) {
      //Printf("New gen. candidate");
      flag = 0;
      signB = 0;
      signD0 = 0;
      int indexGenD0 = -1;

      // B± → D0bar(D0) π± → (K± π∓) π±
      //Printf("Checking B± → D0(bar) π±");
      std::vector<int> arrayDaughterB;
      if (RecoDecay::isMatchedMCGen(particlesMC, particle, pdg::Code::kBPlus, array{-kD0pdg, +kPiPlus}, true, &signB, 1, &arrayDaughterB)) {
        // D0(bar) → π± K∓
        //Printf("Checking D0(bar) → π± K∓");
        for (auto iD : arrayDaughterB) {
          auto candDaughterMC = particlesMC.iteratorAt(iD);
          if (std::abs(candDaughterMC.pdgCode()) == kD0pdg) {
            indexGenD0 = RecoDecay::isMatchedMCGen(particlesMC, candDaughterMC, pdg::Code::kD0, array{-kKPlus, +kPiPlus}, true, &signD0, 1);
          }
        }
        if (indexGenD0 > -1) {
          flag = signB * (1 << hf_cand_bplus::DecayType::BPlusToD0Pi);
        }
      }
      rowMCMatchGen(flag);
    } //B candidate
  }   // process
};    // struct

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{};
  const bool doMC = cfgc.options().get<bool>("doMC");

  workflow.push_back(adaptAnalysisTask<HfCandidateCreatorBplus>(cfgc));
  workflow.push_back(adaptAnalysisTask<HfCandidateCreatorBplusExpressions>(cfgc));
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<HfCandidateCreatorBplusMc>(cfgc));
  }
  return workflow;
}
