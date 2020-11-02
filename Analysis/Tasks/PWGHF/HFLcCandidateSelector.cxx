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
/// \brief Lc->pKpi selection task.
///
/// \author Luigi Dello Stritto <luigi.dellostritto@cern.ch>, CERN
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Analysis/HFSecondaryVertex.h"
#include "Analysis/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;

static const int npTBins = 10;
static const int nCutVars = 8;
//temporary until 2D array in configurable is solved - then move to json
//m  ptp  ptk  ptpi DCA sigmavtx dlenght cosp
constexpr double cuts[npTBins][nCutVars] = {{0.5, 0.2, 0.2, 0.2, 0.05, 0.09, 0.005, 0.},  /* pt<1     */
                                            {0.5, 0.2, 0.2, 0.2, 0.05, 0.09, 0.005, 0.},  /* 1<pt<2   */
                                            {0.5, 0.2, 0.2, 0.2, 0.05, 0.09, 0.005, 0.},  /* 2<pt<3   */
                                            {0.5, 0.2, 0.2, 0.2, 0.05, 0.09, 0.005, 0.},  /* 3<pt<4   */
                                            {0.5, 0.2, 0.2, 0.2, 0.05, 0.09, 0.005, 0.},  /* 4<pt<5   */
                                            {0.5, 0.2, 0.2, 0.2, 0.05, 0.09, 0.005, 0.},  /* 5<pt<6   */
                                            {0.5, 0.2, 0.2, 0.2, 0.05, 0.09, 0.005, 0.},  /* 6<pt<8   */
                                            {0.5, 0.2, 0.2, 0.2, 0.05, 0.09, 0.005, 0.},  /* 8<pt<12  */
                                            {0.5, 0.2, 0.2, 0.2, 0.05, 0.09, 0.005, 0.},  /* 12<pt<24 */
                                            {0.5, 0.2, 0.2, 0.2, 0.05, 0.09, 0.005, 0.}}; /* 24<pt<36 */

/// Struct for applying Lc selection cuts

struct HFLcCandidateSelector {

  Produces<aod::HFSelLcCandidate> hfSelLcCandidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 36., "Upper bound of candidate pT"};

  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 1., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_pidTOFMinpT{"d_pidTOFMinpT", 0.5, "Lower bound of track pT for TOF PID"};
  Configurable<double> d_pidTOFMaxpT{"d_pidTOFMaxpT", 4., "Upper bound of track pT for TOF PID"};

  Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 70., "Lower bound of TPC findable clusters for good PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC only"};
  Configurable<double> d_nSigmaTPCCombined{"d_nSigmaTPCCombined", 5., "Nsigma cut on TPC combined with TOF"};
  Configurable<double> d_nSigmaTOF{"d_nSigmaTOF", 3., "Nsigma cut on TOF only"};
  Configurable<double> d_nSigmaTOFCombined{"d_nSigmaTOFCombined", 5., "Nsigma cut on TOF combined with TPC"};

  /// Gets corresponding pT bin from cut file array
  /// \param candpT is the pT of the candidate
  /// \return corresponding bin number of array
  template <typename T>
  int getpTBin(T candpT)
  {
    double pTBins[npTBins + 1] = {0, 1., 2., 3., 4., 5., 6., 8., 12., 24., 36.};
    if (candpT < pTBins[0] || candpT >= pTBins[npTBins]) {
      return -1;
    }
    for (int i = 0; i < npTBins; i++) {
      if (candpT < pTBins[i + 1]) {
        return i;
      }
    }
    return -1;
  }

  /// Selection on goodness of daughter tracks
  /// \note should be applied at candidate selection
  /// \param track is daughter track
  /// \return true if track is good
  template <typename T>
  bool daughterSelection(const T& track)
  {
    if (track.charge() == 0) {
      return false;
    }
    if (track.tpcNClsFound() == 0) {
      return false; //is it clusters findable or found - need to check
    }
    return true;
  }

  /// Conjugate independent toplogical cuts
  /// \param hfCandProng3 is candidate
  /// \return true if candidate passes all cuts
  template <typename T>
  bool selectionTopol(const T& hfCandProng3)
  {
    auto candpT = hfCandProng3.pt();
    int pTBin = getpTBin(candpT);
    if (pTBin == -1) {
      return false;
    }

    if (candpT < d_pTCandMin || candpT >= d_pTCandMax) {
      return false; //check that the candidate pT is within the analysis range
    }
    if (hfCandProng3.cpa() <= cuts[pTBin][7]) {
      return false; //cosine of pointing angle
    }
    // if (hfCandProng3.dca() > cuts[pTBin][4]) return false; //candidate DCA
    if (hfCandProng3.chi2PCA() > cuts[pTBin][5]) { //candidate DCA
      return false;
    }

    if (hfCandProng3.decayLength() * hfCandProng3.decayLength() < cuts[pTBin][6] * cuts[pTBin][6]) {
      return false;
    }
    return true;
  }

  /// Conjugate dependent toplogical cuts
  /// \param hfCandProng3 is candidate
  /// \param trackProton is the track with the proton hypothesis
  /// \param trackPion is the track with the pion hypothesis
  /// \param trackKaon is the track with the kaon hypothesis
  /// \note trackPion = positive and trackKaon = negative for D0 selection and inverse for D0bar
  /// \return true if candidate passes all cuts for the given Conjugate
  template <typename T1, typename T2>
  bool selectionTopolConjugate(const T1& hfCandProng3, const T2& trackProton, const T2& trackKaon, const T2& trackPion)
  {

    auto candpT = hfCandProng3.pt();
    int pTBin = getpTBin(candpT);
    if (pTBin == -1) {
      return false;
    }

    //invariant mass cut
    if (TMath::Abs(InvMassLc(hfCandProng3) - RecoDecay::getMassPDG(4122)) > cuts[pTBin][0]) {
      return false;
    }

    if (TMath::Abs(trackProton.pt()) < TMath::Abs(cuts[pTBin][1]) || TMath::Abs(trackKaon.pt()) < TMath::Abs(cuts[pTBin][2]) || TMath::Abs(trackPion.pt()) < TMath::Abs(cuts[pTBin][3])) {
      return false; //cut on daughter pT
    }

    /* if (TMath::Sqrt( trackProton.dcaPrim0()*trackProton.dcaPrim0() + trackKaon.dcaPrim0()*trackKaon.dcaPrim0() + trackPion.dcaPrim0()*trackPion.dcaPrim0() ) > cuts[pTBin][5]) {
      return false; //cut on daughter dca - need to add secondary vertex constraint here
    }*/

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
    double nSigma = 100.0; //arbitarily large value
    nPDG = TMath::Abs(nPDG);
    if (nPDG == 2212) {
      nSigma = track.tpcNSigmaPr();
    } else if (nPDG == 321) {
      nSigma = track.tpcNSigmaKa();
    } else if (nPDG == 111) {
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
    double nSigma = 100.0; //arbitarily large value
    nPDG = TMath::Abs(nPDG);
    if (nPDG == 2212) {
      nSigma = track.tofNSigmaPr();
    } else if (nPDG == 321) {
      nSigma = track.tofNSigmaKa();
    } else if (nPDG == 321) {
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

  void process(aod::HfCandProng3 const& hfCandProng3s, aod::BigTracksPID const& tracks)
  {
    int statusLc; // final selection flag : 0-rejected  1-accepted
    bool topolLc;
    int pidLc, proton, kaonMinus, pionPlus;

    for (auto& hfCandProng3 : hfCandProng3s) { //looping over 3 prong candidates

      auto trackPos1 = hfCandProng3.index0_as<aod::BigTracksPID>(); //positive daughter
      auto trackNeg1 = hfCandProng3.index1_as<aod::BigTracksPID>(); //negative daughter
      auto trackPos2 = hfCandProng3.index2_as<aod::BigTracksPID>(); //positive daughter

      statusLc = 0;
      topolLc = true;
      pidLc = -1;
      proton = -1;
      kaonMinus = -1;
      pionPlus = -1;

      // daughter track validity selection
      if (!daughterSelection(trackPos1) || !daughterSelection(trackNeg1) || !daughterSelection(trackPos2)) {
        hfSelLcCandidate(statusLc);
        continue;
      }

      //implement filter bit 4 cut - should be done before this task at the track selection level

      //conjugate independent topological selection
      if (!selectionTopol(hfCandProng3)) {
        hfSelLcCandidate(statusLc);
        continue;
      }

      //conjugate dependent toplogical selection for Lc
      topolLc = selectionTopolConjugate(hfCandProng3, trackPos1, trackNeg1, trackPos2);

      if (!topolLc) {
        hfSelLcCandidate(statusLc);
        continue;
      }

      proton = selectionPID(trackPos1, 2212);
      kaonMinus = selectionPID(trackNeg1, 321);
      pionPlus = selectionPID(trackPos2, 211);

      if (proton == 0 || kaonMinus == 0 || pionPlus == 0) {
        pidLc = 0; //exclude Lc
      }
      if (proton == 1 && kaonMinus == 1 && pionPlus == 1) {
        pidLc = 1; //accept Lc
      }

      if (pidLc == 0) {
        hfSelLcCandidate(statusLc);
        continue;
      }

      if ((pidLc == -1 || pidLc == 1) && topolLc) {
        statusLc = 1; //identified as Lc
      }

      hfSelLcCandidate(statusLc);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFLcCandidateSelector>("hf-lc-candidate-selector")};
}
