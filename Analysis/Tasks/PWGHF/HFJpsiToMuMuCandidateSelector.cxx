// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFJpsiToMuMuCandidateSelector.cxx
/// \brief Jpsi selection task.
/// \author Biao Zhang <biao.zhang@cern.ch>, CCNU
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN
/// \author Antonio Uras <antonio.uras@cern.ch>, IP2I-Lyon

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisCore/HFSelectorCuts.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::analysis;
using namespace o2::analysis::hf_cuts_jpsi_toMuMu;

/// Struct for applying Jpsi selection cuts

struct HFJpsiToMuMuCandidateSelector {

  Produces<aod::HFSelJpsiToMuMuCandidate> hfSelJpsiToMuMuCandidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 50., "Upper bound of candidate pT"};

  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 10., "Upper bound of track pT for TPC PID"};

  Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 70., "Lower bound of TPC findable clusters for good PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC only"};
  Configurable<std::vector<double>> pTBins{"pTBins", std::vector<double>{hf_cuts_jpsi_toMuMu::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"Jpsi_toMuMu_cuts", {hf_cuts_jpsi_toMuMu::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "Jpsi candidate selection per pT bin"};

  /// Selection on goodness of daughter tracks
  /// \note should be applied at candidate selection
  /// \param track is daughter track
  /// \return true if track is good
  template <typename T>
  bool daughterSelection(const T& track)
  {
    /*if (track.tpcNClsFound() == 0) {
      return false; //is it clusters findable or found - need to check
      }*/
    return true;
  }

  /// Conjugate independent toplogical cuts
  /// \param hfCandProng2 is candidate
  /// \param trackMuPlus is the track with the positive muon hypothesis
  /// \param trackMuMinus is the track with the negative muon hypothesis
  /// \return true if candidate passes all cuts
  template <typename T1, typename T2>
  bool selectionTopol(const T1& hfCandProng2, const T2& trackMuPlus, const T2& trackMuMinus)
  {
    auto candpT = hfCandProng2.pt();
    auto pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      return false;
    }

    if (candpT < d_pTCandMin || candpT >= d_pTCandMax) {
      return false; //check that the candidate pT is within the analysis range
    }

    if (TMath::Abs(InvMassJpsiToMuMu(hfCandProng2) - RecoDecay::getMassPDG(pdg::Code::kJpsi)) > cuts->get(pTBin, "m")) {
      return false;
    }

    if (trackMuMinus.pt() < cuts->get(pTBin, "pT Mu") || trackMuPlus.pt() < cuts->get(pTBin, "pT Mu")) {
      return false; //cut on daughter pT
    }
    if (TMath::Abs(trackMuMinus.dcaPrim0()) > cuts->get(pTBin, "DCA_xy") || TMath::Abs(trackMuPlus.dcaPrim0()) > cuts->get(pTBin, "DCA_xy")) {
      return false; //cut on daughter dca - need to add secondary vertex constraint here
    }
    if (TMath::Abs(trackMuMinus.dcaPrim1()) > cuts->get(pTBin, "DCA_z") || TMath::Abs(trackMuPlus.dcaPrim1()) > cuts->get(pTBin, "DCA_z")) {
      return false; //cut on daughter dca - need to add secondary vertex constraint here
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
    if (TMath::Abs(track.pt()) < d_pidTPCMinpT || TMath::Abs(track.pt()) >= d_pidTPCMaxpT) {
      return false;
    }
    //if (track.TPCNClsFindable() < d_TPCNClsFindablePIDCut) return false;
    return true;
  }

  /// Check if track is compatible with given TPC Nsigma cut for the muon hypothesis
  /// \param track is the track
  /// \param nSigmaCut is the nsigma threshold to test against
  /// \return true if track satisfies TPC PID hypothesis for given Nsigma cut
  template <typename T>
  bool selectionPIDTPC(const T& track, int nSigmaCut)
  {
    if (nSigmaCut > 999.) {
      return true;
    }
    //    return track.tpcNSigmaMu() < nSigmaCut;
    return track.tpcNSigmaEl() < nSigmaCut;
  }
  /// PID selection on daughter track
  /// \param track is the daughter track
  /// \return 1 if successful PID match, 0 if successful PID rejection, -1 if no PID info
  template <typename T>
  int selectionPID(const T& track)
  {

    if (validTPCPID(track)) {
      if (!selectionPIDTPC(track, d_nSigmaTPC)) {

        return 0; //rejected by PID
      } else {
        return 1; //positive PID
      }
    } else {
      return -1; //no PID info
    }
  }
  void process(aod::HfCandProng2 const& hfCandProng2s, aod::BigTracksPID const&)
  {

    for (auto& hfCandProng2 : hfCandProng2s) { //looping over 2 prong candidates

      auto trackPos = hfCandProng2.index0_as<aod::BigTracksPID>(); //positive daughter
      auto trackNeg = hfCandProng2.index1_as<aod::BigTracksPID>(); //negative daughter

      if (!(hfCandProng2.hfflag() & 1 << JpsiToMuMu)) {
        hfSelJpsiToMuMuCandidate(0);
        continue;
      }

      // daughter track validity selection
      if (!daughterSelection(trackPos) || !daughterSelection(trackNeg)) {
        hfSelJpsiToMuMuCandidate(0);
        continue;
      }

      //implement filter bit 4 cut - should be done before this task at the track selection level
      //need to add special cuts (additional cuts on decay length and d0 norm)

      if (!selectionTopol(hfCandProng2, trackPos, trackNeg)) {
        hfSelJpsiToMuMuCandidate(0);
        continue;
      }

      if (selectionPID(trackPos) == 0 || selectionPID(trackNeg) == 0) {
        hfSelJpsiToMuMuCandidate(0);
        continue;
      }

      hfSelJpsiToMuMuCandidate(1);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFJpsiToMuMuCandidateSelector>(cfgc, TaskName{"hf-jpsi-toMuMu-candidate-selector"})};
}
