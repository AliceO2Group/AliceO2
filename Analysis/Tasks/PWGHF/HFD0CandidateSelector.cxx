// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFD0CandidateSelector.cxx
/// \brief D0 selection task
///
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::analysis::hf_cuts_d0_topik;

/// Struct for applying D0 selection cuts

struct HFD0CandidateSelector {

  Produces<aod::HFSelD0Candidate> hfSelD0Candidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 50., "Upper bound of candidate pT"};

  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 5., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_pidTOFMinpT{"d_pidTOFMinpT", 0.15, "Lower bound of track pT for TOF PID"};
  Configurable<double> d_pidTOFMaxpT{"d_pidTOFMaxpT", 5., "Upper bound of track pT for TOF PID"};

  Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 50., "Lower bound of TPC findable clusters for good PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC only"};
  Configurable<double> d_nSigmaTPCCombined{"d_nSigmaTPCCombined", 5., "Nsigma cut on TPC combined with TOF"};
  Configurable<double> d_nSigmaTOF{"d_nSigmaTOF", 3., "Nsigma cut on TOF only"};
  Configurable<double> d_nSigmaTOFCombined{"d_nSigmaTOFCombined", 5., "Nsigma cut on TOF combined with TPC"};

  Configurable<std::vector<double>> pTBins{"pTBins", std::vector<double>{hf_cuts_d0_topik::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"D0_to_pi_K_cuts", {hf_cuts_d0_topik::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "D0 candidate selection per pT bin"};

  /// Selection on goodness of daughter tracks
  /// \note should be applied at candidate selection
  /// \param track is daughter track
  /// \return true if track is good
  template <typename T>
  bool daughterSelection(const T& track)
  {
    if (track.sign() == 0) {
      return false;
    }
    /* if (track.tpcNClsFound() == 0) {
      return false; //is it clusters findable or found - need to check
      }*/
    return true;
  }

  /// Conjugate-independent topological cuts
  /// \param hfCandProng2 is candidate
  /// \return true if candidate passes all cuts
  template <typename T>
  bool selectionTopol(const T& hfCandProng2)
  {
    auto candpT = hfCandProng2.pt();
    auto pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      return false;
    }

    if (candpT < d_pTCandMin || candpT >= d_pTCandMax) {
      return false; //check that the candidate pT is within the analysis range
    }
    if (hfCandProng2.impactParameterProduct() > cuts->get(pTBin, "d0d0")) {
      return false; //product of daughter impact parameters
    }
    if (hfCandProng2.cpa() < cuts->get(pTBin, "cos pointing angle")) {
      return false; //cosine of pointing angle
    }
    if (hfCandProng2.cpaXY() < cuts->get(pTBin, "cos pointing angle xy")) {
      return false; //cosine of pointing angle XY
    }
    if (hfCandProng2.decayLengthXYNormalised() < cuts->get(pTBin, "normalized decay length XY")) {
      return false; //normalised decay length in XY plane
    }
    // if (hfCandProng2.dca() > cuts[pTBin][1]) return false; //candidate DCA
    //if (hfCandProng2.chi2PCA() > cuts[pTBin][1]) return false; //candidate DCA

    //decay exponentail law, with tau = beta*gamma*ctau
    //decay length > ctau retains (1-1/e)
    if (std::abs(hfCandProng2.impactParameterNormalised0()) < 0.5 || std::abs(hfCandProng2.impactParameterNormalised1()) < 0.5) {
      return false;
    }
    double decayLengthCut = std::min((hfCandProng2.p() * 0.0066) + 0.01, 0.06);
    if (hfCandProng2.decayLength() * hfCandProng2.decayLength() < decayLengthCut * decayLengthCut) {
      return false;
    }
    if (hfCandProng2.decayLengthNormalised() * hfCandProng2.decayLengthNormalised() < 1.0) {
      //return false; //add back when getter fixed
    }
    return true;
  }

  /// Conjugate-dependent topological cuts
  /// \param hfCandProng2 is candidate
  /// \param trackPion is the track with the pion hypothesis
  /// \param trackKaon is the track with the kaon hypothesis
  /// \note trackPion = positive and trackKaon = negative for D0 selection and inverse for D0bar
  /// \return true if candidate passes all cuts for the given Conjugate
  template <typename T1, typename T2>
  bool selectionTopolConjugate(const T1& hfCandProng2, const T2& trackPion, const T2& trackKaon)
  {
    auto candpT = hfCandProng2.pt();
    auto pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      return false;
    }

    if (trackPion.sign() > 0) { //invariant mass cut
      if (std::abs(InvMassD0(hfCandProng2) - RecoDecay::getMassPDG(pdg::Code::kD0)) > cuts->get(pTBin, "m")) {
        return false;
      }
    } else {
      if (std::abs(InvMassD0bar(hfCandProng2) - RecoDecay::getMassPDG(pdg::Code::kD0)) > cuts->get(pTBin, "m")) {
        return false;
      }
    }

    if (trackPion.pt() < cuts->get(pTBin, "pT Pi") || trackKaon.pt() < cuts->get(pTBin, "pT K")) {
      return false; //cut on daughter pT
    }
    if (std::abs(trackPion.dcaPrim0()) > cuts->get(pTBin, "d0pi") || std::abs(trackKaon.dcaPrim0()) > cuts->get(pTBin, "d0K")) {
      return false; //cut on daughter dca - need to add secondary vertex constraint here
    }

    if (trackPion.sign() > 0) { //cut on cos(theta *)
      if (std::abs(CosThetaStarD0(hfCandProng2)) > cuts->get(pTBin, "cos theta*")) {
        return false;
      }
    } else {
      if (std::abs(CosThetaStarD0bar(hfCandProng2)) > cuts->get(pTBin, "cos theta*")) {
        return false;
      }
    }

    return true;
  }

  /// Check if track is ok for TPC PID
  /// \param track is the track
  /// \note function to be expanded
  /// \return true if track is ok for TPC PID
  template <typename T>
  bool validTPCPID(const T& track)
  {
    if (track.pt() < d_pidTPCMinpT || track.pt() >= d_pidTPCMaxpT) {
      return false;
    }
    //if (track.TPCNClsFindable() < d_TPCNClsFindablePIDCut) return false;
    return true;
  }

  /// Check if track is ok for TOF PID
  /// \param track is the track
  /// \note function to be expanded
  /// \return true if track is ok for TOF PID
  template <typename T>
  bool validTOFPID(const T& track)
  {
    if (track.pt() < d_pidTOFMinpT || track.pt() >= d_pidTOFMaxpT) {
      return false;
    }
    return true;
  }

  /// Check if track is compatible with given TPC Nsigma cut for a given flavour hypothesis
  /// \param track is the track
  /// \param nPDG is the flavour hypothesis PDG number
  /// \param nSigmaCut is the nsigma threshold to test against
  /// \return true if track satisfies TPC PID hypothesis for given Nsigma cut
  template <typename T>
  bool selectionPIDTPC(const T& track, int nPDG, double nSigmaCut)
  {
    double nSigma = 100.0; //arbitarily large value
    nPDG = std::abs(nPDG);
    if (nPDG == kPiPlus) {
      nSigma = track.tpcNSigmaPi();
    } else if (nPDG == kKPlus) {
      nSigma = track.tpcNSigmaKa();
    } else {
      return false;
    }
    return std::abs(nSigma) < nSigmaCut;
  }

  /// Check if track is compatible with given TOF NSigma cut for a given flavour hypothesis
  /// \param track is the track
  /// \param nPDG is the flavour hypothesis PDG number
  /// \param nSigmaCut is the nSigma threshold to test against
  /// \return true if track satisfies TOF PID hypothesis for given NSigma cut
  template <typename T>
  bool selectionPIDTOF(const T& track, int nPDG, double nSigmaCut)
  {
    double nSigma = 100.0; //arbitarily large value
    nPDG = std::abs(nPDG);
    if (nPDG == kPiPlus) {
      nSigma = track.tofNSigmaPi();
    } else if (nPDG == kKPlus) {
      nSigma = track.tofNSigmaKa();
    } else {
      return false;
    }
    return std::abs(nSigma) < nSigmaCut;
  }

  /// PID selection on daughter track
  /// \param track is the daughter track
  /// \param nPDG is the PDG code of the flavour hypothesis
  /// \return 1 if successful PID match, 0 if successful PID rejection, -1 if no PID info
  template <typename T>
  int selectionPID(const T& track, int nPDG)
  {
    int statusTPC = -1;
    int statusTOF = -1;

    if (validTPCPID(track)) {
      if (!selectionPIDTPC(track, nPDG, d_nSigmaTPC)) {
        if (!selectionPIDTPC(track, nPDG, d_nSigmaTPCCombined)) {
          statusTPC = 0; //rejected by PID
        } else {
          statusTPC = 1; //potential to be acceepted if combined with TOF
        }
      } else {
        statusTPC = 2; //positive PID
      }
    } else {
      statusTPC = -1; //no PID info
    }

    if (validTOFPID(track)) {
      if (!selectionPIDTOF(track, nPDG, d_nSigmaTOF)) {
        if (!selectionPIDTOF(track, nPDG, d_nSigmaTOFCombined)) {
          statusTOF = 0; //rejected by PID
        } else {
          statusTOF = 1; //potential to be acceepted if combined with TOF
        }
      } else {
        statusTOF = 2; //positive PID
      }
    } else {
      statusTOF = -1; //no PID info
    }

    if (statusTPC == 2 || statusTOF == 2) {
      return 1; //what if we have 2 && 0 ?
    } else if (statusTPC == 1 && statusTOF == 1) {
      return 1;
    } else if (statusTPC == 0 || statusTOF == 0) {
      return 0;
    } else {
      return -1;
    }
  }

  void process(aod::HfCandProng2 const& hfCandProng2s, aod::BigTracksPID const&)
  {
    for (auto& hfCandProng2 : hfCandProng2s) { //looping over 2 prong candidates

      // final selection flag : 0-rejected  1-accepted
      int statusD0 = 0;
      int statusD0bar = 0;

      if (!(hfCandProng2.hfflag() & 1 << DecayType::D0ToPiK)) {
        hfSelD0Candidate(statusD0, statusD0bar);
        continue;
      }

      auto trackPos = hfCandProng2.index0_as<aod::BigTracksPID>(); //positive daughter
      if (!daughterSelection(trackPos)) {
        hfSelD0Candidate(statusD0, statusD0bar);
        continue;
      }
      auto trackNeg = hfCandProng2.index1_as<aod::BigTracksPID>(); //negative daughter
      if (!daughterSelection(trackNeg)) {
        hfSelD0Candidate(statusD0, statusD0bar);
        continue;
      }

      //conjugate-independent topological selection
      if (!selectionTopol(hfCandProng2)) {
        hfSelD0Candidate(statusD0, statusD0bar);
        continue;
      }

      //implement filter bit 4 cut - should be done before this task at the track selection level
      //need to add special cuts (additional cuts on decay length and d0 norm)

      //conjugate-dependent topological selection for D0
      bool topolD0 = selectionTopolConjugate(hfCandProng2, trackPos, trackNeg);
      //conjugate-dependent topological selection for D0bar
      bool topolD0bar = selectionTopolConjugate(hfCandProng2, trackNeg, trackPos);

      if (!topolD0 && !topolD0bar) {
        hfSelD0Candidate(statusD0, statusD0bar);
        continue;
      }

      int pionPlus = selectionPID(trackPos, kPiPlus);
      int kaonMinus = selectionPID(trackNeg, kKPlus);
      int pionMinus = selectionPID(trackNeg, kPiPlus);
      int kaonPlus = selectionPID(trackPos, kKPlus);

      int pidD0 = -1;
      int pidD0bar = -1;

      if (pionPlus == 0 || kaonMinus == 0 || pionMinus == 1 || kaonPlus == 1) {
        pidD0 = 0; //exclude D0
      }
      if (pionPlus == 1 || kaonMinus == 1 || pionMinus == 0 || kaonPlus == 0) {
        pidD0bar = 0; //exclude D0bar
      }
      if (pionPlus == 1 && kaonMinus == 1) {
        pidD0 = 1; //accept D0
      }
      if (pionMinus == 1 && kaonPlus == 1) {
        pidD0bar = 1; //accept D0bar
      }

      if (pidD0 == 0 && pidD0bar == 0) {
        hfSelD0Candidate(statusD0, statusD0bar);
        continue;
      }

      if ((pidD0 == -1 || pidD0 == 1) && topolD0) {
        statusD0 = 1; //identified as D0
      }
      if ((pidD0bar == -1 || pidD0bar == 1) && topolD0bar) {
        statusD0bar = 1; //identified as D0bar
      }

      hfSelD0Candidate(statusD0, statusD0bar);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFD0CandidateSelector>(cfgc, TaskName{"hf-d0-candidate-selector"})};
}
