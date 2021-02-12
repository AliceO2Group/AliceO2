// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFLcK0spCandidateSelector.cxx
/// \brief Lc --> K0s+p selection task.
///
/// \author Chiara Zampolli <Chiara.Zampolli@cern.ch>, CERN

/// based on HFD0CandidateSelector.cxx

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_casc;

static const int npTBins = 8;
static const int nCutVars = 8;
//temporary until 2D array in configurable is solved - then move to json
//mK0s(MeV)     mLambdas(MeV)    mGammas(MeV)    ptp     ptK0sdau     pTLc     d0p     d0K0sdau
constexpr double cuts[npTBins][nCutVars] = {{8.0, 5., 100, 0.5, 0.3, 0.6, 0.05, 999999},   // 1 < pt < 2
                                            {8.0, 5., 100, 0.5, 0.4, 1.3, 0.05, 999999},   // 2 < pt < 3
                                            {9.0, 5., 100, 0.6, 0.4, 1.3, 0.05, 999999},   // 3 < pt < 4
                                            {11.0, 5., 100, 0.6, 0.4, 1.4, 0.05, 999999},  // 4 < pt < 5
                                            {13.0, 5., 100, 0.6, 0.4, 1.4, 0.06, 999999},  // 5 < pt < 6
                                            {13.0, 5., 100, 0.9, 0.4, 1.6, 0.09, 999999},  // 6 < pt < 8
                                            {16.0, 5., 100, 0.9, 0.4, 1.7, 0.10, 999999},  // 8 < pt < 12
                                            {19.0, 5., 100, 1.0, 0.4, 1.9, 0.20, 999999}}; // 12 < pt < 24
/// Struct for applying D0 selection cuts

struct HFLcK0spCandidateSelector {

  Produces<aod::HFSelLcK0spCandidate> hfSelLcK0spCandidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 50., "Upper bound of candidate pT"};

  // PID
  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0., "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 100., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_pidCombMaxp{"d_pidCombMaxp", 4., "Upper bound of track p to use TOF + TPC Bayes PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC only"};

  // track quality
  Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 50., "Lower bound of TPC findable clusters for good PID"};
  Configurable<bool> b_requireTPC{"b_requireTPC", true, "Flag to require a positive Number of found clusters in TPC"};

  /// Gets corresponding pT bin from cut file array
  /// \param candpT is the pT of the candidate
  /// \return corresponding bin number of array
  template <typename T>
  int getpTBin(T candpT) // This should be taken out of the selector, since it is something in common to everyone;
                         // it should become parameterized with the pt intervals, and also the pt intervals
                         // should be configurable from outside
  {
    double pTBins[npTBins + 1] = {1., 2., 3., 4., 5., 6., 8., 12., 24.};
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
  bool daughterSelection(const T& track) // aren't these checks already in the indexskimscreator?
  {
    if (track.charge() == 0) {
      return false;
    }
    if (b_requireTPC.value && track.tpcNClsFound() == 0) {
      return false; //is it clusters findable or found - need to check
    }
    return true;
  }

  /// Conjugate independent toplogical cuts
  /// \param hfCandCascade is candidate
  /// \return true if candidate passes all cuts
  template <typename T>
  bool selectionTopol(const T& hfCandCascade)
  {
    auto candpT = hfCandCascade.pt();
    int pTBin = getpTBin(candpT);
    if (pTBin == -1) {
      return false;
    }

    if (candpT < d_pTCandMin || candpT >= d_pTCandMax) {
      return false; //check that the candidate pT is within the analysis range
    }

    if (std::abs(hfCandCascade.mK0Short() - RecoDecay::getMassPDG(kK0Short)) > cuts[pTBin][0]) {
      return false; // mass of the K0s
    }

    if ((std::abs(hfCandCascade.mLambda() - RecoDecay::getMassPDG(kLambda0)) < cuts[pTBin][1]) || (std::abs(hfCandCascade.mAntiLambda() - RecoDecay::getMassPDG(kLambda0)) < cuts[pTBin][1])) {
      return false; // mass of the Lambda
    }

    if (std::abs(InvMassGamma(hfCandCascade) - RecoDecay::getMassPDG(kGamma)) < cuts[pTBin][2]) {
      return false; // mass of the Gamma
    }

    if (hfCandCascade.ptProng0() > cuts[pTBin][3]) {
      return false; // pt of the K0
    }

    if (hfCandCascade.ptProng1() > cuts[pTBin][4]) {
      return false; // pt of the p
    }

    if (hfCandCascade.pt() > cuts[pTBin][5]) {
      return false; // pt of the Lc
    }

    if (std::abs(hfCandCascade.impactParameter1()) > cuts[pTBin][6]) {
      return false; // d0 of the bachelor
    }

    if ((std::abs(hfCandCascade.dcapostopv()) > cuts[pTBin][7]) || (std::abs(hfCandCascade.dcanegtopv()) > cuts[pTBin][7])) {
      return false; // d0 of the K0s daughters
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
  bool validCombPID(const T& track)
  {
    if (TMath::Abs(track.pt()) > d_pidCombMaxp) { // is the pt sign used for the charge? If it is always positive, we should remove the abs
      return false;
    }
    return true;
  }

  /// Check if track is compatible with given TPC Nsigma cut for a given flavour hypothesis
  /// \param track is the track
  /// \param nPDG is the flavour hypothesis PDG number
  /// \param nSigmaCut is the nsigma threshold to test against
  /// \note nPDG=211 pion  nPDG=321 kaon
  /// \return true if track satisfies TPC PID hypothesis for given Nsigma cut
  template <typename T>
  bool selectionPIDTPC(const T& track, double nSigmaCut)
  {
    double nSigma = 100.0; //arbitarily large value
    nSigma = track.tpcNSigmaPr();
    return nSigma > nSigmaCut;
  }

  /*
  /// Check if track is compatible with given TOF NSigma cut for a given flavour hypothesis
  /// \param track is the track
  /// \param nPDG is the flavour hypothesis PDG number
  /// \param nSigmaCut is the nSigma threshold to test against
  /// \note nPDG=211 pion  nPDG=321 kaon
  /// \return true if track satisfies TOF PID hypothesis for given NSigma cut
  template <typename T>
  bool selectionPIDTOF(const T& track, int nPDG, double nSigmaCut)
  {
    double nSigma = 100.0; //arbitarily large value
    nPDG = TMath::Abs(nPDG);
    if (nPDG == 111) {
      nSigma = track.tofNSigmaPi();
    } else if (nPDG == 321) {
      nSigma = track.tofNSigmaKa();
    } else {
      return false;
    }
    return nSigma < nSigmaCut;
  }
  */

  /// PID selection on daughter track
  /// \param track is the daughter track
  /// \param nPDG is the PDG code of the flavour hypothesis
  /// \note nPDG=211 pion  nPDG=321 kaon
  /// \return 1 if successful PID match, 0 if successful PID rejection, -1 if no PID info
  template <typename T>
  int selectionPID(const T& track)
  {
    int statusTPC = -1;
    //    int statusTOF = -1;

    if (validTPCPID(track)) {
      if (!selectionPIDTPC(track, d_nSigmaTPC)) {
        statusTPC = 0;
        /*
        if (!selectionPIDTPC(track, nPDG, d_nSigmaTPCCombined)) {
          statusTPC = 0; //rejected by PID
        } else {
          statusTPC = 1; //potential to be acceepted if combined with TOF
        }
      } else {
        statusTPC = 2; //positive PID
      }
	*/
      } else {
        statusTPC = 1;
      }
    }

    return statusTPC;
    /*
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
      */
  }

  void process(aod::HfCandCascade const& hfCandCascades, aod::BigTracksPID const& tracks)
  {
    int statusLc = 0; // final selection flag : 0-rejected  1-accepted
    bool topolLc = 0;
    int pidProton = -1;
    int pidLc = -1;

    for (auto& hfCandCasc : hfCandCascades) { //looping over 2 prong candidates

      statusLc = 0;
      /* // not needed for the Lc
      if (!(hfCandCasc.hfflag() & 1 << D0ToPiK)) {
        hfSelD0Candidate(statusLc);
        continue;
      }
      */

      const auto& bach = hfCandCasc.index0_as<aod::BigTracksPID>(); //bachelor track

      topolLc = true;
      pidProton = -1;

      // daughter track validity selection
      if (!daughterSelection(bach)) {
        hfSelLcK0spCandidate(statusLc);
        continue;
      }

      //implement filter bit 4 cut - should be done before this task at the track selection level
      //need to add special cuts (additional cuts on decay length and d0 norm)

      if (!selectionTopol(hfCandCasc)) {
        hfSelLcK0spCandidate(statusLc);
        continue;
      }

      pidProton = selectionPID(bach);

      if (pidProton == 1) {
        statusLc = 1;
      }

      hfSelLcK0spCandidate(statusLc);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFLcK0spCandidateSelector>("hf-lck0sp-candidate-selector")};
}
