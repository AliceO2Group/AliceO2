// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFTreeCreator2Prong.cxx
/// \brief Writer of the 2 prong candidates in the form of flat tables to be stored in TTrees.
///        Intended for debug or for the local optimization of analysis on small samples.
///        In this file are defined and filled the output tables
///
/// \author Nicolo' Jacazio <nicolo.jacazio@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "AnalysisCore/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;

namespace o2::aod
{
namespace full
{
DECLARE_SOA_COLUMN(RSecondaryVertex, rSecondaryVertex, float);
DECLARE_SOA_COLUMN(PtProng0, ptProng0, float);
DECLARE_SOA_COLUMN(PProng0, pProng0, float);
DECLARE_SOA_COLUMN(ImpactParameterNormalised0, impactParameterNormalised0, float);
DECLARE_SOA_COLUMN(PtProng1, ptProng1, float);
DECLARE_SOA_COLUMN(PProng1, pProng1, float);
DECLARE_SOA_COLUMN(ImpactParameterNormalised1, impactParameterNormalised1, float);
DECLARE_SOA_COLUMN(CandidateSelFlag, candidateSelFlag, int8_t);
DECLARE_SOA_COLUMN(M, m, float);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(P, p, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Y, y, float);
DECLARE_SOA_COLUMN(E, e, float);
DECLARE_SOA_COLUMN(NSigTPCPi0, nsigTPCPi0, float);
DECLARE_SOA_COLUMN(NSigTPCKa0, nsigTPCKa0, float);
DECLARE_SOA_COLUMN(NSigTOFPi0, nsigTOFPi0, float);
DECLARE_SOA_COLUMN(NSigTOFKa0, nsigTOFKa0, float);
DECLARE_SOA_COLUMN(NSigTPCPi1, nsigTPCPi1, float);
DECLARE_SOA_COLUMN(NSigTPCKa1, nsigTPCKa1, float);
DECLARE_SOA_COLUMN(NSigTOFPi1, nsigTOFPi1, float);
DECLARE_SOA_COLUMN(NSigTOFKa1, nsigTOFKa1, float);
DECLARE_SOA_COLUMN(DecayLength, decayLength, float);
DECLARE_SOA_COLUMN(DecayLengthXY, decayLengthXY, float);
DECLARE_SOA_COLUMN(DecayLengthNormalised, decayLengthNormalised, float);
DECLARE_SOA_COLUMN(DecayLengthXYNormalised, decayLengthXYNormalised, float);
DECLARE_SOA_COLUMN(CPA, cpa, float);
DECLARE_SOA_COLUMN(CPAXY, cpaXY, float);
DECLARE_SOA_COLUMN(Ct, ct, float);
DECLARE_SOA_COLUMN(ImpactParameterProduct, impactParameterProduct, float);
DECLARE_SOA_COLUMN(CosThetaStar, cosThetaStar, float);
DECLARE_SOA_COLUMN(MCflag, mcflag, int8_t);
// Events
DECLARE_SOA_COLUMN(IsEventReject, isEventReject, int);
DECLARE_SOA_COLUMN(RunNumber, runNumber, int);
} // namespace full

DECLARE_SOA_TABLE(HfCandProng2Full, "AOD", "HFCANDP2Full",
                  collision::BCId,
                  collision::NumContrib,
                  collision::PosX,
                  collision::PosY,
                  collision::PosZ,
                  hf_cand::XSecondaryVertex,
                  hf_cand::YSecondaryVertex,
                  hf_cand::ZSecondaryVertex,
                  hf_cand::ErrorDecayLength,
                  hf_cand::ErrorDecayLengthXY,
                  hf_cand::Chi2PCA,
                  full::RSecondaryVertex,
                  full::DecayLength,
                  full::DecayLengthXY,
                  full::DecayLengthNormalised,
                  full::DecayLengthXYNormalised,
                  full::ImpactParameterNormalised0,
                  full::PtProng0,
                  full::PProng0,
                  full::ImpactParameterNormalised1,
                  full::PtProng1,
                  full::PProng1,
                  hf_cand::PxProng0,
                  hf_cand::PyProng0,
                  hf_cand::PzProng0,
                  hf_cand::PxProng1,
                  hf_cand::PyProng1,
                  hf_cand::PzProng1,
                  hf_cand::ImpactParameter0,
                  hf_cand::ImpactParameter1,
                  hf_cand::ErrorImpactParameter0,
                  hf_cand::ErrorImpactParameter1,
                  full::NSigTPCPi0,
                  full::NSigTPCKa0,
                  full::NSigTOFPi0,
                  full::NSigTOFKa0,
                  full::NSigTPCPi1,
                  full::NSigTPCKa1,
                  full::NSigTOFPi1,
                  full::NSigTOFKa1,
                  full::CandidateSelFlag,
                  full::M,
                  full::ImpactParameterProduct,
                  full::CosThetaStar,
                  full::Pt,
                  full::P,
                  full::CPA,
                  full::CPAXY,
                  full::Ct,
                  full::Eta,
                  full::Phi,
                  full::Y,
                  full::E,
                  full::MCflag);

DECLARE_SOA_TABLE(HfCandProng2FullEvents, "AOD", "HFCANDP2FullE",
                  collision::BCId,
                  collision::NumContrib,
                  collision::PosX,
                  collision::PosY,
                  collision::PosZ,
                  full::IsEventReject,
                  full::RunNumber);

DECLARE_SOA_TABLE(HfCandProng2FullParticles, "AOD", "HFCANDP2FullP",
                  collision::BCId,
                  full::Pt,
                  full::Eta,
                  full::Phi,
                  full::Y,
                  full::MCflag);

} // namespace o2::aod

/// Writes the full information in an output TTree
struct CandidateTreeWriter {
  Produces<o2::aod::HfCandProng2Full> rowCandidateFull;
  Produces<o2::aod::HfCandProng2FullEvents> rowCandidateFullEvents;
  Produces<o2::aod::HfCandProng2FullParticles> rowCandidateFullParticles;
  void init(InitContext const&)
  {
  }
  using CandType = soa::Join<aod::HfCandProng2, aod::HfCandProng2MCRec, aod::HFSelD0Candidate>;
  void process(aod::Collisions const& collisions,
               aod::McCollisions const& mccollisions,
               CandType const& candidates,
               soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particles,
               aod::BigTracksPID const& tracks)
  {

    // Filling event properties
    rowCandidateFullEvents.reserve(collisions.size());
    for (auto& collision : collisions) {
      rowCandidateFullEvents(
        collision.bcId(),
        collision.numContrib(),
        collision.posX(),
        collision.posY(),
        collision.posZ(),
        0,
        1);
    }

    // Filling candidate properties
    rowCandidateFull.reserve(candidates.size());
    for (auto& candidate : candidates) {
      auto fillTable = [&](int CandFlag,
                           int FunctionSelection,
                           double FunctionInvMass,
                           double FunctionCosThetaStar,
                           double FunctionCt,
                           double FunctionY,
                           double FunctionE) {
        if (FunctionSelection >= 1) {
          rowCandidateFull(
            candidate.index0_as<aod::BigTracksPID>().collision().bcId(),
            candidate.index0_as<aod::BigTracksPID>().collision().numContrib(),
            candidate.posX(),
            candidate.posY(),
            candidate.posZ(),
            candidate.xSecondaryVertex(),
            candidate.ySecondaryVertex(),
            candidate.zSecondaryVertex(),
            candidate.errorDecayLength(),
            candidate.errorDecayLengthXY(),
            candidate.chi2PCA(),
            candidate.rSecondaryVertex(),
            candidate.decayLength(),
            candidate.decayLengthXY(),
            candidate.decayLengthNormalised(),
            candidate.decayLengthXYNormalised(),
            candidate.impactParameterNormalised0(),
            candidate.ptProng0(),
            RecoDecay::P(candidate.pxProng0(), candidate.pyProng0(), candidate.pzProng0()),
            candidate.impactParameterNormalised1(),
            candidate.ptProng1(),
            RecoDecay::P(candidate.pxProng1(), candidate.pyProng1(), candidate.pzProng1()),
            candidate.pxProng0(),
            candidate.pyProng0(),
            candidate.pzProng0(),
            candidate.pxProng1(),
            candidate.pyProng1(),
            candidate.pzProng1(),
            candidate.impactParameter0(),
            candidate.impactParameter1(),
            candidate.errorImpactParameter0(),
            candidate.errorImpactParameter1(),
            candidate.index0_as<aod::BigTracksPID>().tpcNSigmaPi(),
            candidate.index0_as<aod::BigTracksPID>().tpcNSigmaKa(),
            candidate.index0_as<aod::BigTracksPID>().tofNSigmaPi(),
            candidate.index0_as<aod::BigTracksPID>().tofNSigmaKa(),
            candidate.index1_as<aod::BigTracksPID>().tpcNSigmaPi(),
            candidate.index1_as<aod::BigTracksPID>().tpcNSigmaKa(),
            candidate.index1_as<aod::BigTracksPID>().tofNSigmaPi(),
            candidate.index1_as<aod::BigTracksPID>().tofNSigmaKa(),
            1 << CandFlag,
            FunctionInvMass,
            candidate.impactParameterProduct(),
            FunctionCosThetaStar,
            candidate.pt(),
            candidate.p(),
            candidate.cpa(),
            candidate.cpaXY(),
            FunctionCt,
            candidate.eta(),
            candidate.phi(),
            FunctionY,
            FunctionE,
            candidate.flagMCMatchRec());
        }
      };

      fillTable(0, candidate.isSelD0(), InvMassD0(candidate), CosThetaStarD0(candidate), CtD0(candidate), YD0(candidate), ED0(candidate));
      fillTable(1, candidate.isSelD0bar(), InvMassD0bar(candidate), CosThetaStarD0bar(candidate), CtD0(candidate), YD0(candidate), ED0(candidate));
    }

    // Filling particle properties
    rowCandidateFullParticles.reserve(particles.size());
    for (auto& particle : particles) {
      if (std::abs(particle.flagMCMatchGen()) == 1 << D0ToPiK) {
        rowCandidateFullParticles(
          particle.mcCollision().bcId(),
          particle.pt(),
          particle.eta(),
          particle.phi(),
          RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode())),
          particle.flagMCMatchGen());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow;
  workflow.push_back(adaptAnalysisTask<CandidateTreeWriter>(cfgc, TaskName{"hf-tree-creator-d0-tokpi"}));
  return workflow;
}
