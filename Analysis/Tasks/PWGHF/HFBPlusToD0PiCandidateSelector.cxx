// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFBPlusToD0PiCandidateSelector.cxx
/// \brief B+ to D0barPi candidate selector
///
/// \author Antonio Palasciano <antonio.palasciano@cern.ch>, Università degli Studi di Bari & INFN, Sezione di Bari
/// \author Deepa Thomas <deepa.thomas@cern.ch>, UT Austin

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisCore/HFSelectorCuts.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "AnalysisCore/HFSelectorCuts.h"

using namespace o2;
using namespace o2::aod;
using namespace o2::framework;
using namespace o2::aod::hf_cand_bplus;
using namespace o2::analysis;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::analysis::hf_cuts_bplus_tod0pi;

struct HFBPlusToD0PiCandidateSelector {

  Produces<aod::HFSelBPlusToD0PiCandidate> hfSelBPlusToD0PiCandidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 50., "Upper bound of candidate pT"};

  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 10., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_pidTOFMinpT{"d_pidTOFMinpT", 0.15, "Lower bound of track pT for TOF PID"};
  Configurable<double> d_pidTOFMaxpT{"d_pidTOFMaxpT", 10., "Upper bound of track pT for TOF PID"};

  Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 70., "Lower bound of TPC findable clusters for good PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 5., "Nsigma cut on TPC only"};
  Configurable<double> d_nSigmaTPCCombined{"d_nSigmaTPCCombined", 5., "Nsigma cut on TPC combined with TOF"};
  Configurable<double> d_nSigmaTOF{"d_nSigmaTOF", 2., "Nsigma cut on TOF only"};
  Configurable<double> d_nSigmaTOFCombined{"d_nSigmaTOFCombined", 5., "Nsigma cut on TOF combined with TPC"};

  Configurable<std::vector<double>> pTBins{"pTBins", std::vector<double>{hf_cuts_bplus_tod0pi::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"BPlus_to_d0pi_cuts", {hf_cuts_bplus_tod0pi::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "B+ candidate selection per pT bin"};
  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};

  // return true if track is good
  template <typename T>
  bool daughterSelection(const T& track)
  {
    if (track.sign() == 0) {
      return false;
    }
    /*if (track.tpcNClsFound() == 0) {
      return false; //is it clusters findable or found - need to check
      }*/
    return true;
  }

  // Apply topological cuts as defined in HFSelectorCuts.h; return true if candidate passes all cuts
  template <typename T1, typename T2, typename T3>
  bool selectionTopol(const T1& hfCandBPlus, const T2& hfCandD0, const T3& trackPos)
  {
    auto candpT = hfCandBPlus.pt();
    int pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      // Printf("B+ topol selection failed at getpTBin");
      return false;
    }

    //pi pt
    if (trackPos.pt() < cuts->get(pTBin, "pT Pi")) {
      return false;
    }

    //d0(D0)xd0(pi)
    if (hfCandBPlus.impactParameterProduct() > cuts->get(pTBin, "Imp. Par. Product")) {
      return false;
    }

    //D0 mass
    if (trackPos.sign() > 0) {
      if (TMath::Abs(InvMassD0bar(hfCandD0) - RecoDecay::getMassPDG(421)) > cuts->get(pTBin, "DeltaMD0")) {
        return false;
      }
    }
    if (trackPos.sign() < 0) {
      if (TMath::Abs(InvMassD0(hfCandD0) - RecoDecay::getMassPDG(421)) > cuts->get(pTBin, "DeltaMD0")) {
        return false;
      }
    }

    //B Decay length
    if (TMath::Abs(hfCandBPlus.decayLength()) < cuts->get(pTBin, "B decLen")) {
      return false;
    }

    //B+ CPA cut
    if (hfCandBPlus.cpa() < cuts->get(pTBin, "CPA")) {
      return false;
    }

    //if (candpT < d_pTCandMin || candpT >= d_pTCandMax) {
    // Printf("B+ topol selection failed at cand pT check");
    // return false;
    // }

    //B+ mass cut
    //if (TMath::Abs(InvMassBplus(hfCandBPlus) - RecoDecay::getMassPDG(521)) > cuts->get(pTBin, "m")) {
    // Printf("B+ topol selection failed at mass diff check");
    //  return false;
    // }

    //d0 of D0 and pi
    //if ((TMath::Abs(hfCandBPlus.impactParameter0()) > cuts->get(pTBin, "d0 D0")) ||
    //    (TMath::Abs(hfCandBPlus.impactParameter1()) > cuts->get(pTBin, "d0 Pi"))){
    //  return false;
    //}
    //D0 CPA
    // if (TMath::Abs(hfCandD0.cpa()) < cuts->get(pTBin, "CPA D0")){
    //  return false;
    //}
    return true;
  }

  /// Check if track is ok for TPC PID
  template <typename T>
  bool validTPCPID(const T& track)
  {
    if (TMath::Abs(track.pt()) < d_pidTPCMinpT || TMath::Abs(track.pt()) >= d_pidTPCMaxpT) {
      return false;
    }
    //if (track.TPCNClsFindable() < d_TPCNClsFindablePIDCut) return false;
    return true;
  }

  /// Check if track is ok for TOF PID
  template <typename T>
  bool validTOFPID(const T& track)
  {
    if (TMath::Abs(track.pt()) < d_pidTOFMinpT || TMath::Abs(track.pt()) >= d_pidTOFMaxpT) {
      return false;
    }
    return true;
  }

  /// Check if track is compatible with given TPC Nsigma cut for the pion hypothesis
  template <typename T>
  bool selectionPIDTPC(const T& track, int nPDG, double nSigmaCut)
  {
    double nSigma = 100.0; //arbitarily large value
    nPDG = std::abs(nPDG);
    if (nPDG == kPiPlus) {
      nSigma = track.tpcNSigmaPi();
    } else {
      return false;
    }
    if (std::abs(nSigma) < nSigmaCut)
      return true;
    else
      return false;
    //return std::abs(nSigma) < nSigmaCut;
  }

  /// Check if track is compatible with given TOF NSigma cut for a given flavour hypothesis
  template <typename T>
  bool selectionPIDTOF(const T& track, int nPDG, double nSigmaCut)
  {
    double nSigma = 100.0; //arbitarily large value
    nPDG = std::abs(nPDG);
    if (nPDG == kPiPlus) {
      nSigma = track.tofNSigmaPi();
    } else {
      return false;
    }
    if (std::abs(nSigma) < nSigmaCut)
      return true;
    else
      return false;
    //  return std::abs(nSigma) < nSigmaCut;
  }

  /// TPC and TOF PID selection on daughter track
  /// return 1 if successful PID match, 0 if successful PID rejection, -1 if no PID info
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

    //question on the logic of this??
    /*
    if (statusTPC == 2 || statusTOF == 2) {
      return 1; //what if we have 2 && 0 ?
    } else if (statusTPC == 1 && statusTOF == 1) {
      return 1;
    } else if (statusTPC == 0 || statusTOF == 0) {
      return 0;
    } else {
      return -1;
    }
    */
    //temporary till I understand
    if (statusTPC == 2 || statusTOF == 2) {
      return 1;
    } else {
      return -1;
    }
  }

  void process(aod::HfCandBPlus const& hfCandBs, soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>, aod::BigTracksPID const& tracks)
  {
    for (auto& hfCandB : hfCandBs) { //looping over Bplus candidates
      // D0 is always index0 and pi is index1 by default
      auto candD0 = hfCandB.index0_as<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>>();
      auto trackPos = hfCandB.index1_as<aod::BigTracksPID>();

      int statusBplus = 0;

      // check if flagged as B+ --> D0bar Pi
      if (!(hfCandB.hfflag() & 1 << DecayType::BPlusToD0Pi)) {
        hfSelBPlusToD0PiCandidate(statusBplus);
        // Printf("B+ candidate selection failed at hfflag check");
        continue;
      }

      // daughter track validity selection
      if (!daughterSelection(trackPos)) {
        hfSelBPlusToD0PiCandidate(statusBplus);
        // Printf("B+ candidate selection failed at daughter selection");
        continue;
      }

      //topological cuts
      if (!selectionTopol(hfCandB, candD0, trackPos)) {
        hfSelBPlusToD0PiCandidate(statusBplus);
        // Printf("B+ candidate selection failed at selection topology");
        continue;
      }

      /*      //Pion PID
      //if (selectionPID(trackPos, kPiPlus) == 0) {
      if (selectionPID(trackPos, kPiPlus) != 1) {
        hfSelBPlusToD0PiCandidate(statusBplus);
        // Printf("B+ candidate selection failed at selection PID");
        continue;
      }
*/
      hfSelBPlusToD0PiCandidate(1);
      // Printf("B+ candidate selection successful, candidate should be selected");
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFBPlusToD0PiCandidateSelector>(cfgc, TaskName{"hf-bplus-tod0pi-candidate-selector"})};
}
