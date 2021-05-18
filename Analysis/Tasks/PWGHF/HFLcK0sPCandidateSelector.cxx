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
#include "AnalysisTasksUtils/UtilsDebugLcK0Sp.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_casc;

static const int nPtBins = 8;
static const int nCutVars = 8;
//temporary until 2D array in configurable is solved - then move to json
//mK0s(GeV)     mLambdas(GeV)    mGammas(GeV)    ptp     ptK0sdau     pTLc     d0p     d0K0
constexpr double cuts[nPtBins][nCutVars] = {{0.008, 0.005, 0.1, 0.5, 0.3, 0.6, 0.05, 999999.},  // 1 < pt < 2
                                            {0.008, 0.005, 0.1, 0.5, 0.4, 1.3, 0.05, 999999.},  // 2 < pt < 3
                                            {0.009, 0.005, 0.1, 0.6, 0.4, 1.3, 0.05, 999999.},  // 3 < pt < 4
                                            {0.011, 0.005, 0.1, 0.6, 0.4, 1.4, 0.05, 999999.},  // 4 < pt < 5
                                            {0.013, 0.005, 0.1, 0.6, 0.4, 1.4, 0.06, 999999.},  // 5 < pt < 6
                                            {0.013, 0.005, 0.1, 0.9, 0.4, 1.6, 0.09, 999999.},  // 6 < pt < 8
                                            {0.016, 0.005, 0.1, 0.9, 0.4, 1.7, 0.10, 999999.},  // 8 < pt < 12
                                            {0.019, 0.005, 0.1, 1.0, 0.4, 1.9, 0.20, 999999.}}; // 12 < pt < 24
/// Struct for applying D0 selection cuts

//#define MY_DEBUG

#ifdef MY_DEBUG
#define MY_DEBUG_MSG(condition, cmd) \
  if (condition) {                   \
    cmd;                             \
  }
using MyBigTracks = soa::Join<aod::BigTracksPID, aod::McTrackLabels>;
#else
#define MY_DEBUG_MSG(condition, cmd)
using MyBigTracks = aod::BigTracksPID;
#endif

struct HFLcK0sPCandidateSelector {

  Produces<aod::HFSelLcK0sPCandidate> hfSelLcK0sPCandidate;

  Configurable<double> pTCandMin{"pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> pTCandMax{"pTCandMax", 50., "Upper bound of candidate pT"};

  // PID
  Configurable<double> applyPidTPCMinPt{"applyPidTPCMinPt", 4., "Lower bound of track pT to apply TPC PID"};
  Configurable<double> pidTPCMinPt{"pidTPCMinPt", 0., "Lower bound of track pT for TPC PID"};
  Configurable<double> pidTPCMaxPt{"pidTPCMaxPt", 100., "Upper bound of track pT for TPC PID"};
  Configurable<double> pidCombMaxP{"pidCombMaxP", 4., "Upper bound of track p to use TOF + TPC Bayes PID"};
  Configurable<double> nSigmaTPC{"nSigmaTPC", 3., "Nsigma cut on TPC only"};

  // track quality
  Configurable<double> TPCNClsFindablePIDCut{"TPCNClsFindablePIDCut", 50., "Lower bound of TPC findable clusters for good PID"};
  Configurable<bool> requireTPC{"requireTPC", true, "Flag to require a positive Number of found clusters in TPC"};

  // for debugging
#ifdef MY_DEBUG
  Configurable<std::vector<int>> indexK0Spos{"indexK0Spos", {729, 2866, 4754, 5457, 6891, 7824, 9243, 9810}, "indices of K0S positive daughters, for debug"};
  Configurable<std::vector<int>> indexK0Sneg{"indexK0Sneg", {730, 2867, 4755, 5458, 6892, 7825, 9244, 9811}, "indices of K0S negative daughters, for debug"};
  Configurable<std::vector<int>> indexProton{"indexProton", {717, 2810, 4393, 5442, 6769, 7793, 9002, 9789}, "indices of protons, for debug"};
#endif

  /// Gets corresponding pT bin from cut file array
  /// \param candPt is the pT of the candidate
  /// \return corresponding bin number of array
  template <typename T>
  int getPtBin(T candPt) // This should be taken out of the selector, since it is something in common to everyone;
                         // it should become parameterized with the pt intervals, and also the pt intervals
                         // should be configurable from outside
  {
    double ptBins[nPtBins + 1] = {1., 2., 3., 4., 5., 6., 8., 12., 24.};
    if (candPt < ptBins[0] || candPt >= ptBins[nPtBins]) {
      return -1;
    }
    for (int i = 0; i < nPtBins; i++) {
      if (candPt < ptBins[i + 1]) {
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
  bool daughterSelection(const T& track) // there could be more checks done on the bachelor here
  {
    // this is for now just a placeholder, in case we want to add extra checks
    return true;
  }

  /// Conjugate independent toplogical cuts
  /// \param hfCandCascade is candidate
  /// \return true if candidate passes all cuts
  template <typename T>
  bool selectionTopol(const T& hfCandCascade)
  {
    auto candPt = hfCandCascade.pt();
    int ptBin = getPtBin(candPt);
    if (ptBin == -1) {
      return false;
    }

    if (candPt < pTCandMin || candPt >= pTCandMax) {
      LOG(DEBUG) << "cand pt (first check) cut failed: from cascade --> " << candPt << ", cut --> " << pTCandMax;
      return false; //check that the candidate pT is within the analysis range
    }

    if (std::abs(hfCandCascade.mK0Short() - RecoDecay::getMassPDG(kK0Short)) > cuts[ptBin][0]) {
      LOG(DEBUG) << "massK0s cut failed: from v0 in cascade, K0s --> " << hfCandCascade.mK0Short() << ", in PDG K0s --> " << RecoDecay::getMassPDG(kK0Short) << ", cut --> " << cuts[ptBin][0];
      return false; // mass of the K0s
    }

    if ((std::abs(hfCandCascade.mLambda() - RecoDecay::getMassPDG(kLambda0)) < cuts[ptBin][1]) || (std::abs(hfCandCascade.mAntiLambda() - RecoDecay::getMassPDG(kLambda0)) < cuts[ptBin][1])) {
      LOG(DEBUG) << "mass L cut failed: from v0 in cascade, Lambda --> " << hfCandCascade.mLambda() << ", AntiLambda --> " << hfCandCascade.mAntiLambda() << ", in PDG, Lambda --> " << RecoDecay::getMassPDG(kLambda0) << ", cut --> " << cuts[ptBin][1];
      return false; // mass of the Lambda
    }

    if (std::abs(InvMassGamma(hfCandCascade) - RecoDecay::getMassPDG(kGamma)) < cuts[ptBin][2]) {
      LOG(DEBUG) << "mass gamma cut failed: from v0 in cascade, gamma --> " << InvMassGamma(hfCandCascade) << ", cut --> " << cuts[ptBin][2];
      return false; // mass of the Gamma
    }

    if (hfCandCascade.ptProng0() < cuts[ptBin][3]) {
      LOG(DEBUG) << "bach pt cut failed, from cascade --> " << hfCandCascade.ptProng0() << " , cut --> " << cuts[ptBin][3];
      return false; // pt of the p
    }

    if (hfCandCascade.ptV0Pos() < cuts[ptBin][4]) {
      LOG(DEBUG) << "v0 pos daugh pt cut failed, from cascade --> " << hfCandCascade.ptV0Pos() << ", cut --> " << cuts[ptBin][4];
      return false; // pt of the K0
    }

    if (hfCandCascade.ptV0Neg() < cuts[ptBin][4]) {
      LOG(DEBUG) << "v0 neg daugh pt cut failed, from cascade --> " << hfCandCascade.ptV0Neg() << ", cut --> " << cuts[ptBin][4];
      return false; // pt of the K0
    }

    if (hfCandCascade.pt() < cuts[ptBin][5]) {
      LOG(DEBUG) << "cand pt cut failed, from cascade --> " << hfCandCascade.pt() << ", cut --> " << cuts[ptBin][5];
      return false; // pt of the Lc
    }

    if (std::abs(hfCandCascade.impactParameter0()) > cuts[ptBin][6]) {
      LOG(DEBUG) << "d0 bach cut failed, in cascade --> " << hfCandCascade.impactParameter0() << ", cut --> " << cuts[ptBin][6];
      return false; // d0 of the bachelor
    }

    /*
    if ((std::abs(hfCandCascade.dcapostopv()) > cuts[ptBin][7]) || (std::abs(hfCandCascade.dcanegtopv()) > cuts[ptBin][7])) {
      LOG(DEBUG) << "v0 daugh cut failed, positive v0 daugh --> " << hfCandCascade.dcapostopv() << ", negative v0 daugh --> " << hfCandCascade.dcanegtopv() << " , cut --> " << cuts[ptBin][7];
      return false; // d0 of the K0s daughters
    }
    */

    if (std::abs(hfCandCascade.impactParameter1()) > cuts[ptBin][7]) {
      LOG(DEBUG) << "d0 v0 cut failed, in cascade --> " << hfCandCascade.impactParameter1() << ", cut --> " << cuts[ptBin][7];
      return false; // d0 of the v0
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
    if (track.pt() < pidTPCMinPt || track.pt() >= pidTPCMaxPt) {
      LOG(DEBUG) << "Bachelor pt is " << track.pt() << ", we trust TPC PID in [" << pidTPCMinPt << ", " << pidTPCMaxPt << "]";
      return false;
    }
    return true;
  }

  /// Check if we will use TPC PID
  /// \param track is the track
  /// \note function to be expanded
  /// \return true if track is ok for TPC PID
  template <typename T>
  bool applyTPCPID(const T& track)
  {
    if (track.pt() < applyPidTPCMinPt) {
      LOG(DEBUG) << "Bachelor pt is " << track.pt() << ", we apply TPC PID from " << applyPidTPCMinPt;
      return false;
    }
    LOG(DEBUG) << "Bachelor pt is " << track.pt() << ", we apply TPC PID from " << applyPidTPCMinPt;
    return true;
  }

  /// Check if track is ok for TOF PID
  /// \param track is the track
  /// \note function to be expanded
  /// \return true if track is ok for TOF PID
  template <typename T>
  bool validCombPID(const T& track)
  {
    if (track.pt() > pidCombMaxP) { // is the pt sign used for the charge? If it is always positive, we should remove the abs
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
    LOG(DEBUG) << "nSigma for bachelor = " << nSigma << ", cut at " << nSigmaCut;
    return std::abs(nSigma) < nSigmaCut;
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

    if (!applyTPCPID(track)) {
      // we do not apply TPC PID in this range
      return 1;
    }

    if (validTPCPID(track)) {
      LOG(DEBUG) << "We check the TPC PID now";
      if (!selectionPIDTPC(track, nSigmaTPC)) {
        statusTPC = 0;
        /*
        if (!selectionPIDTPC(track, nPDG, nSigmaTPCCombined)) {
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
      if (!selectionPIDTOF(track, nPDG, nSigmaTOF)) {
        if (!selectionPIDTOF(track, nPDG, nSigmaTOFCombined)) {
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

  void process(aod::HfCandCascade const& candidates, MyBigTracks const& tracks)
  {
    int statusLc = 0; // final selection flag : 0-rejected  1-accepted
    bool topolLc = 0;
    int pidProton = -1;
    int pidLc = -1;

    for (auto& candidate : candidates) { //looping over cascade candidates

      const auto& bach = candidate.index0_as<MyBigTracks>(); //bachelor track
#ifdef MY_DEBUG
      auto indexV0DaughPos = candidate.posTrack_as<MyBigTracks>().mcParticleId();
      auto indexV0DaughNeg = candidate.negTrack_as<MyBigTracks>().mcParticleId();
      auto indexBach = bach.mcParticleId();
      bool isLc = isLcK0SpFunc(indexBach, indexV0DaughPos, indexV0DaughNeg, indexProton, indexK0Spos, indexK0Sneg);
      bool isK0SfromLc = isK0SfromLcFunc(indexV0DaughPos, indexV0DaughNeg, indexK0Spos, indexK0Sneg);
#endif
      MY_DEBUG_MSG(isLc, printf("\n"); LOG(INFO) << "In selector: correct Lc found: proton --> " << indexBach << ", posTrack --> " << indexV0DaughPos << ", negTrack --> " << indexV0DaughNeg);
      //MY_DEBUG_MSG(isLc != 1, printf("\n"); LOG(INFO) << "In selector: wrong Lc found: proton --> " << indexBach << ", posTrack --> " << indexV0DaughPos << ", negTrack --> " << indexV0DaughNeg);

      statusLc = 0;
      /* // not needed for the Lc
      if (!(candidate.hfflag() & 1 << D0ToPiK)) {
        hfSelD0Candidate(statusLc);
        continue;
      }
      */

      topolLc = true;
      pidProton = -1;

      // daughter track validity selection
      LOG(DEBUG) << "daughterSelection(bach) = " << daughterSelection(bach);
      if (!daughterSelection(bach)) {
        MY_DEBUG_MSG(isLc, LOG(INFO) << "In selector: Lc rejected due to selections on bachelor");
        hfSelLcK0sPCandidate(statusLc);
        continue;
      }

      //implement filter bit 4 cut - should be done before this task at the track selection level
      //need to add special cuts (additional cuts on decay length and d0 norm)
      LOG(DEBUG) << "selectionTopol(candidate) = " << selectionTopol(candidate);
      if (!selectionTopol(candidate)) {
        MY_DEBUG_MSG(isLc, LOG(INFO) << "In selector: Lc rejected due to topological selections");
        hfSelLcK0sPCandidate(statusLc);
        continue;
      }

      pidProton = selectionPID(bach);

      LOG(DEBUG) << "pidProton = " << pidProton;

      if (pidProton == 1) {
        statusLc = 1;
      }

      MY_DEBUG_MSG(isLc && pidProton != 1, LOG(INFO) << "In selector: Lc rejected due to PID selections on bachelor");
      MY_DEBUG_MSG(isLc && pidProton == 1, LOG(INFO) << "In selector: Lc ACCEPTED");

      hfSelLcK0sPCandidate(statusLc);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfcg)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFLcK0sPCandidateSelector>(cfcg, TaskName{"hf-lc-tok0sp-candidate-selector"})};
}
