// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFLcCandidateSelector.cxx
/// \brief Lc->pKpi selection task.
///
/// \author Luigi Dello Stritto <luigi.dello.stritto@cern.ch>, University and INFN SALERNO
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "AnalysisCore/HFSelectorCuts.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;
using namespace o2::analysis;
using namespace o2::analysis::hf_cuts_lc_topkpi;

/// Struct for applying Lc selection cuts
struct HFLcCandidateSelector {

  Produces<aod::HFSelLcCandidate> hfSelLcCandidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 36., "Upper bound of candidate pT"};

  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.1, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 1., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_pidTOFMinpT{"d_pidTOFMinpT", 0.5, "Lower bound of track pT for TOF PID"};
  Configurable<double> d_pidTOFMaxpT{"d_pidTOFMaxpT", 2.5, "Upper bound of track pT for TOF PID"};

  Configurable<bool> d_FilterPID{"d_FilterPID", true, "Bool to use or not the PID at filtering level"};
  Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 70., "Lower bound of TPC findable clusters for good PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC only"};
  Configurable<double> d_nSigmaTPCCombined{"d_nSigmaTPCCombined", 5., "Nsigma cut on TPC combined with TOF"};
  Configurable<double> d_nSigmaTOF{"d_nSigmaTOF", 3., "Nsigma cut on TOF only"};
  Configurable<double> d_nSigmaTOFCombined{"d_nSigmaTOFCombined", 5., "Nsigma cut on TOF combined with TPC"};

  Configurable<std::vector<double>> ptBins{"ptBins", std::vector<double>{hf_cuts_lc_topkpi::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"Lc_to_p_K_pi_cuts", {hf_cuts_lc_topkpi::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "Lc candidate selection per pT bin"};

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
    /*if (track.tpcNClsFound() == 0) {
      return false; //is it clusters findable or found - need to check
      }*/
    return true;
  }

  /// Conjugate independent toplogical cuts
  /// \param hfCandProng3 is candidate
  /// \return true if candidate passes all cuts
  template <typename T>
  bool selectionTopol(const T& hfCandProng3)
  {
    auto candpT = hfCandProng3.pt();

    int pTBin = findBin(ptBins, candpT);
    if (pTBin == -1) {
      return false;
    }

    if (candpT < d_pTCandMin || candpT >= d_pTCandMax) {
      return false; //check that the candidate pT is within the analysis range
    }

    if (hfCandProng3.cpa() <= cuts->get(pTBin, "cos pointing angle")) {
      return false; //cosine of pointing angle
    }

    /*  if (hfCandProng3.chi2PCA() > cuts[pTBin][5]) { //candidate DCA
      return false;
      }*/

    if (hfCandProng3.decayLength() <= cuts->get(pTBin, "decay length")) {
      return false;
    }
    return true;
  }

  /// Conjugate dependent toplogical cuts
  /// \param hfCandProng3 is candidate
  /// \param trackProton is the track with the proton hypothesis
  /// \param trackPion is the track with the pion hypothesis
  /// \param trackKaon is the track with the kaon hypothesis
  /// \return true if candidate passes all cuts for the given Conjugate
  template <typename T1, typename T2>
  bool selectionTopolConjugate(const T1& hfCandProng3, const T2& trackProton, const T2& trackKaon, const T2& trackPion)
  {

    auto candpT = hfCandProng3.pt();
    int pTBin = findBin(ptBins, candpT);
    if (pTBin == -1) {
      return false;
    }

    if (trackProton.pt() < cuts->get(pTBin, "pT p") || trackKaon.pt() < cuts->get(pTBin, "pT K") || trackPion.pt() < cuts->get(pTBin, "pT Pi")) {
      return false; //cut on daughter pT
    }

    if (trackProton.globalIndex() == hfCandProng3.index0Id()) {
      if (TMath::Abs(InvMassLcpKpi(hfCandProng3) - RecoDecay::getMassPDG(4122)) > cuts->get(pTBin, "m")) {
        return false;
      }
    } else {
      if (TMath::Abs(InvMassLcpiKp(hfCandProng3) - RecoDecay::getMassPDG(4122)) > cuts->get(pTBin, "m")) {
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
    if (TMath::Abs(track.pt()) < d_pidTPCMinpT || TMath::Abs(track.pt()) >= d_pidTPCMaxpT) {
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
    if (TMath::Abs(track.pt()) < d_pidTOFMinpT || TMath::Abs(track.pt()) >= d_pidTOFMaxpT) {
      return false;
    }
    return true;
  }

  /// Check if track is compatible with given TPC Nsigma cut for a given flavour hypothesis
  /// \param track is the track
  /// \param nPDG is the flavour hypothesis PDG number
  /// \param nSigmaCut is the nsigma threshold to test against
  /// \note nPDG=2212 proton  nPDG=211 pion  nPDG=321 kaon
  /// \return true if track satisfies TPC PID hypothesis for given Nsigma cut
  template <typename T>
  bool selectionPIDTPC(const T& track, int nPDG, int nSigmaCut)
  {
    double nSigma = 100.; //arbitarily large value
    nPDG = TMath::Abs(nPDG);
    if (nPDG == kProton) {
      nSigma = track.tpcNSigmaPr();
    } else if (nPDG == kKPlus) {
      nSigma = track.tpcNSigmaKa();
    } else if (nPDG == kPiPlus) {
      nSigma = track.tpcNSigmaPi();
    } else {
      return false;
    }
    return nSigma < nSigmaCut;
  }

  /// Check if track is compatible with given TOF NSigma cut for a given flavour hypothesis
  /// \param track is the track
  /// \param nPDG is the flavour hypothesis PDG number
  /// \param nSigmaCut is the nSigma threshold to test against
  /// \note nPDG=2212 proton  nPDG=211 pion  nPDG=321 kaon
  /// \return true if track satisfies TOF PID hypothesis for given NSigma cut
  template <typename T>
  bool selectionPIDTOF(const T& track, int nPDG, int nSigmaCut)
  {
    double nSigma = 100.; //arbitarily large value
    nPDG = TMath::Abs(nPDG);
    if (nPDG == kProton) {
      nSigma = track.tofNSigmaPr();
    } else if (nPDG == kKPlus) {
      nSigma = track.tofNSigmaKa();
    } else if (nPDG == kPiPlus) {
      nSigma = track.tofNSigmaPi();
    } else {
      return false;
    }
    return nSigma < nSigmaCut;
  }

  /// PID selection on daughter track
  /// \param track is the daughter track
  /// \param nPDG is the PDG code of the flavour hypothesis
  /// \note nPDG=2212  nPDG=211 pion  nPDG=321 kaon
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

  void process(aod::HfCandProng3 const& hfCandProng3s, aod::BigTracksPID const&)
  {
    for (auto& hfCandProng3 : hfCandProng3s) {
      // final selection flag : 0-rejected  1-accepted
      auto statusLcpKpi = 0;
      auto statusLcpiKp = 0;

      if (!(hfCandProng3.hfflag() & 1 << LcToPKPi)) {
        hfSelLcCandidate(statusLcpKpi, statusLcpiKp);
        continue;
      }
      auto trackPos1 = hfCandProng3.index0_as<aod::BigTracksPID>(); //positive daughter (negative for the antiparticles)
      auto trackNeg1 = hfCandProng3.index1_as<aod::BigTracksPID>(); //negative daughter (positive for the antiparticles)
      auto trackPos2 = hfCandProng3.index2_as<aod::BigTracksPID>(); //positive daughter (negative for the antiparticles)

      // daughter track validity selection
      if (!daughterSelection(trackPos1) || !daughterSelection(trackNeg1) || !daughterSelection(trackPos2)) {
        hfSelLcCandidate(statusLcpKpi, statusLcpiKp);
        continue;
      }

      auto pidLcpKpi = -1;
      auto pidLcpiKp = -1;

      //implement filter bit 4 cut - should be done before this task at the track selection level

      //conjugate independent topological selection
      if (!selectionTopol(hfCandProng3)) {
        hfSelLcCandidate(statusLcpKpi, statusLcpiKp);
        continue;
      }

      //conjugate dependent toplogical selection for Lc

      bool topolLcpKpi = selectionTopolConjugate(hfCandProng3, trackPos1, trackNeg1, trackPos2);
      bool topolLcpiKp = selectionTopolConjugate(hfCandProng3, trackPos2, trackNeg1, trackPos1);

      if (!topolLcpKpi && !topolLcpiKp) {
        hfSelLcCandidate(statusLcpKpi, statusLcpiKp);
        continue;
      }

      if (!d_FilterPID) {
        // PID non applied
        pidLcpKpi = 1;
        pidLcpiKp = 1;
      } else {
        int proton1 = selectionPID(trackPos1, kProton);
        int proton2 = selectionPID(trackPos2, kProton);
        int kaonMinus = selectionPID(trackNeg1, kKPlus);
        int pionPlus1 = selectionPID(trackPos1, kPiPlus);
        int pionPlus2 = selectionPID(trackPos2, kPiPlus);

        if (proton1 == 0 || kaonMinus == 0 || pionPlus2 == 0) {
          pidLcpKpi = 0; //exclude LcpKpi
        }
        if (proton1 == 1 && kaonMinus == 1 && pionPlus2 == 1) {
          pidLcpKpi = 1; //accept LcpKpi
        }
        if (proton2 == 0 || kaonMinus == 0 || pionPlus1 == 0) {
          pidLcpiKp = 0; //exclude LcpiKp
        }
        if (proton2 == 1 && kaonMinus == 1 && pionPlus1 == 1) {
          pidLcpiKp = 1; //accept LcpiKp
        }
      }

      if (pidLcpKpi == 0 && pidLcpiKp == 0) {
        hfSelLcCandidate(statusLcpKpi, statusLcpiKp);
        continue;
      }

      if ((pidLcpKpi == -1 || pidLcpKpi == 1) && topolLcpKpi) {
        statusLcpKpi = 1; //identified as LcpKpi
      }
      if ((pidLcpiKp == -1 || pidLcpiKp == 1) && topolLcpiKp) {
        statusLcpiKp = 1; //identified as LcpiKp
      }

      hfSelLcCandidate(statusLcpKpi, statusLcpiKp);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFLcCandidateSelector>(cfgc, TaskName{"hf-lc-candidate-selector"})};
}
