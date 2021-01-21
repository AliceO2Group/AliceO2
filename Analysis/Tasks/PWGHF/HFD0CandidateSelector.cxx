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
/// \brief D0 selection task.
///
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;

static const int npTBins = 25;
static const int nCutVars = 11;
//temporary until 2D array in configurable is solved - then move to json
//m     dca   cost* ptk  ptpi  d0k            d0pi         d0d0     cosp cosxy normdxy
constexpr double cuts[npTBins][nCutVars] = {{0.400, 350. * 1E-4, 0.8, 0.5, 0.5, 1000. * 1E-4, 1000. * 1E-4, -5000. * 1E-8, 0.80, 0., 0.},   /* pt<0.5*/
                                            {0.400, 350. * 1E-4, 0.8, 0.5, 0.5, 1000. * 1E-4, 1000. * 1E-4, -5000. * 1E-8, 0.80, 0., 0.},   /* 0.5<pt<1*/
                                            {0.400, 300. * 1E-4, 0.8, 0.4, 0.4, 1000. * 1E-4, 1000. * 1E-4, -25000. * 1E-8, 0.80, 0., 0.},  /* 1<pt<1.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.4, 0.4, 1000. * 1E-4, 1000. * 1E-4, -25000. * 1E-8, 0.80, 0., 0.},  /* 1.5<pt<2 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -20000. * 1E-8, 0.90, 0., 0.},  /* 2<pt<2.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -20000. * 1E-8, 0.90, 0., 0.},  /* 2.5<pt<3 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -12000. * 1E-8, 0.85, 0., 0.},  /* 3<pt<3.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -12000. * 1E-8, 0.85, 0., 0.},  /* 3.5<pt<4 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 4<pt<4.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 4.5<pt<5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 5<pt<5.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 5.5<pt<6 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 6<pt<6.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 6.5<pt<7 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -7000. * 1E-8, 0.85, 0., 0.},   /* 7<pt<7.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -7000. * 1E-8, 0.85, 0., 0.},   /* 7.5<pt<8 */
                                            {0.400, 300. * 1E-4, 0.9, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -5000. * 1E-8, 0.85, 0., 0.},   /* 8<pt<9 */
                                            {0.400, 300. * 1E-4, 0.9, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -5000. * 1E-8, 0.85, 0., 0.},   /* 9<pt<10 */
                                            {0.400, 300. * 1E-4, 0.9, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -5000. * 1E-8, 0.85, 0., 0.},   /* 10<pt<12 */
                                            {0.400, 300. * 1E-4, 1.0, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, 10000. * 1E-8, 0.85, 0., 0.},   /* 12<pt<16 */
                                            {0.400, 300. * 1E-4, 1.0, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, 999999. * 1E-8, 0.85, 0., 0.},  /* 16<pt<20 */
                                            {0.400, 300. * 1E-4, 1.0, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, 999999. * 1E-8, 0.85, 0., 0.},  /* 20<pt<24 */
                                            {0.400, 300. * 1E-4, 1.0, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, 999999. * 1E-8, 0.85, 0., 0.},  /* 24<pt<36 */
                                            {0.400, 300. * 1E-4, 1.0, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, 999999. * 1E-8, 0.85, 0., 0.},  /* 36<pt<50 */
                                            {0.400, 300. * 1E-4, 1.0, 0.6, 0.6, 1000. * 1E-4, 1000. * 1E-4, 999999. * 1E-8, 0.80, 0., 0.}}; /* pt>50 */

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

  /// Gets corresponding pT bin from cut file array
  /// \param candpT is the pT of the candidate
  /// \return corresponding bin number of array
  template <typename T>
  int getpTBin(T candpT)
  {
    double pTBins[npTBins + 1] = {0, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 9., 10., 12., 16., 20., 24., 36., 50., 100.};
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
  /// \param hfCandProng2 is candidate
  /// \return true if candidate passes all cuts
  template <typename T>
  bool selectionTopol(const T& hfCandProng2)
  {
    auto candpT = hfCandProng2.pt();
    int pTBin = getpTBin(candpT);
    if (pTBin == -1) {
      return false;
    }

    if (candpT < d_pTCandMin || candpT >= d_pTCandMax) {
      return false; //check that the candidate pT is within the analysis range
    }
    if (hfCandProng2.impactParameterProduct() > cuts[pTBin][7]) {
      return false; //product of daughter impact parameters
    }
    if (hfCandProng2.cpa() < cuts[pTBin][8]) {
      return false; //cosine of pointing angle
    }
    if (hfCandProng2.cpaXY() < cuts[pTBin][9]) {
      return false; //cosine of pointing angle XY
    }
    if (hfCandProng2.decayLengthXYNormalised() < cuts[pTBin][10]) {
      return false; //normalised decay length in XY plane
    }
    // if (hfCandProng2.dca() > cuts[pTBin][1]) return false; //candidate DCA
    //if (hfCandProng2.chi2PCA() > cuts[pTBin][1]) return false; //candidate DCA

    //decay exponentail law, with tau = beta*gamma*ctau
    //decay length > ctau retains (1-1/e)
    double decayLengthCut = TMath::Min((hfCandProng2.p() * 0.0066) + 0.01, 0.06);
    if (TMath::Abs(hfCandProng2.impactParameterNormalised0()) < 0.5 || TMath::Abs(hfCandProng2.impactParameterNormalised1()) < 0.5) {
      return false;
    }
    if (hfCandProng2.decayLength() * hfCandProng2.decayLength() < decayLengthCut * decayLengthCut) {
      return false;
    }
    if (hfCandProng2.decayLengthNormalised() * hfCandProng2.decayLengthNormalised() < 1.0) {
      //return false; //add back when getter fixed
    }
    return true;
  }

  /// Conjugate dependent toplogical cuts
  /// \param hfCandProng2 is candidate
  /// \param trackPion is the track with the pion hypothesis
  /// \param trackKaon is the track with the kaon hypothesis
  /// \note trackPion = positive and trackKaon = negative for D0 selection and inverse for D0bar
  /// \return true if candidate passes all cuts for the given Conjugate
  template <typename T1, typename T2>
  bool selectionTopolConjugate(const T1& hfCandProng2, const T2& trackPion, const T2& trackKaon)
  {

    auto candpT = hfCandProng2.pt();
    int pTBin = getpTBin(candpT);
    if (pTBin == -1) {
      return false;
    }

    if (trackPion.charge() > 0) { //invariant mass cut
      if (TMath::Abs(InvMassD0(hfCandProng2) - RecoDecay::getMassPDG(421)) > cuts[pTBin][0]) {
        return false;
      }
    } else {
      if (TMath::Abs(InvMassD0bar(hfCandProng2) - RecoDecay::getMassPDG(421)) > cuts[pTBin][0]) {
        return false;
      }
    }

    if (trackPion.pt() < cuts[pTBin][4] || trackKaon.pt() < cuts[pTBin][3]) {
      return false; //cut on daughter pT
    }
    if (TMath::Abs(trackPion.dcaPrim0()) > cuts[pTBin][6] || TMath::Abs(trackKaon.dcaPrim0()) > cuts[pTBin][5]) {
      return false; //cut on daughter dca - need to add secondary vertex constraint here
    }

    if (trackPion.charge() > 0) { //cut on cos(theta *)
      if (TMath::Abs(CosThetaStarD0(hfCandProng2)) > cuts[pTBin][2]) {
        return false;
      }
    } else {
      if (TMath::Abs(CosThetaStarD0bar(hfCandProng2)) > cuts[pTBin][2]) {
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
  /// \note nPDG=211 pion  nPDG=321 kaon
  /// \return true if track satisfies TPC PID hypothesis for given Nsigma cut
  template <typename T>
  bool selectionPIDTPC(const T& track, int nPDG, int nSigmaCut)
  {
    double nSigma = 100.0; //arbitarily large value
    nPDG = TMath::Abs(nPDG);
    if (nPDG == 111) {
      nSigma = track.tpcNSigmaPi();
    } else if (nPDG == 321) {
      nSigma = track.tpcNSigmaKa();
    } else {
      return false;
    }
    return nSigma < nSigmaCut;
  }

  /// Check if track is compatible with given TOF NSigma cut for a given flavour hypothesis
  /// \param track is the track
  /// \param nPDG is the flavour hypothesis PDG number
  /// \param nSigmaCut is the nSigma threshold to test against
  /// \note nPDG=211 pion  nPDG=321 kaon
  /// \return true if track satisfies TOF PID hypothesis for given NSigma cut
  template <typename T>
  bool selectionPIDTOF(const T& track, int nPDG, int nSigmaCut)
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

  /// PID selection on daughter track
  /// \param track is the daughter track
  /// \param nPDG is the PDG code of the flavour hypothesis
  /// \note nPDG=211 pion  nPDG=321 kaon
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

  void process(aod::HfCandProng2 const& hfCandProng2s, aod::BigTracksPID const& tracks)
  {
    int statusD0, statusD0bar; // final selection flag : 0-rejected  1-accepted
    bool topolD0, topolD0bar;
    int pidD0, pidD0bar, pionPlus, pionMinus, kaonPlus, kaonMinus;

    for (auto& hfCandProng2 : hfCandProng2s) { //looping over 2 prong candidates

      statusD0 = 0;
      statusD0bar = 0;

      if (!(hfCandProng2.hfflag() & 1 << D0ToPiK)) {
        hfSelD0Candidate(statusD0, statusD0bar);
        continue;
      }

      auto trackPos = hfCandProng2.index0_as<aod::BigTracksPID>(); //positive daughter
      auto trackNeg = hfCandProng2.index1_as<aod::BigTracksPID>(); //negative daughter

      topolD0 = true;
      topolD0bar = true;
      pidD0 = -1;
      pidD0bar = -1;
      pionPlus = -1;
      pionMinus = -1;
      kaonPlus = -1;
      kaonMinus = -1;

      // daughter track validity selection
      if (!daughterSelection(trackPos) || !daughterSelection(trackNeg)) {
        hfSelD0Candidate(statusD0, statusD0bar);
        continue;
      }

      //implement filter bit 4 cut - should be done before this task at the track selection level
      //need to add special cuts (additional cuts on decay length and d0 norm)

      //conjugate independent topological selection
      if (!selectionTopol(hfCandProng2)) {
        hfSelD0Candidate(statusD0, statusD0bar);
        continue;
      }

      //conjugate dependent toplogical selection for D0
      topolD0 = selectionTopolConjugate(hfCandProng2, trackPos, trackNeg);
      //conjugate dependent toplogical selection for D0bar
      topolD0bar = selectionTopolConjugate(hfCandProng2, trackNeg, trackPos);

      if (!topolD0 && !topolD0bar) {
        hfSelD0Candidate(statusD0, statusD0bar);
        continue;
      }

      pionPlus = selectionPID(trackPos, 211);
      kaonMinus = selectionPID(trackNeg, 321);
      pionMinus = selectionPID(trackNeg, 211);
      kaonPlus = selectionPID(trackPos, 321);

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

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFD0CandidateSelector>("hf-d0-candidate-selector")};
}
