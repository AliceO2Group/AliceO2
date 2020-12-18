// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFJpsiCandidateSelector.cxx
/// \brief Jpsi selection task.
///
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;

static const int npTBins = 9;
static const int nCutVars = 4;
//temporary until 2D array in configurable is solved - then move to json
//    mass  dcaxy dcaz  pt_e
constexpr double cuts[npTBins][nCutVars] =
    {{0.5, 0.2, 0.4, 1},  /* pt<0.5   */
    {0.5, 0.2, 0.4,  1},  /* 0.5<pt<1 */
    {0.5, 0.2, 0.4,  1},  /* 1<pt<2   */
    {0.5, 0.2, 0.4,  1},  /* 2<pt<3   */
    {0.5, 0.2, 0.4,  1},  /* 3<pt<4   */
    {0.5, 0.2, 0.4,  1},  /* 4<pt<5   */
    {0.5, 0.2, 0.4,  1},  /* 5<pt<7   */
    {0.5, 0.2, 0.4,  1},  /* 7<pt<10  */
    {0.5, 0.2, 0.4,  1}}; /* 10<pt<15 */

/// Struct for applying Jpsi selection cuts

struct HFJpsiCandidateSelector {

  Produces<aod::HFSelJpsiCandidate> hfSelJpsiCandidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 50., "Upper bound of candidate pT"};

  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 10., "Upper bound of track pT for TPC PID"};
 

  Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 70., "Lower bound of TPC findable clusters for good PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC only"};

  /// Gets corresponding pT bin from cut file array
  /// \param candpT is the pT of the candidate
  /// \return corresponding bin number of array
  template <typename T>
  int getpTBin(T candpT)
  {
    double pTBins[npTBins + 1] = {0, 0.5, 1., 2., 3., 4., 5., 7., 10., 15.};
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
  
    return true;
  }

  /// Conjugate dependent toplogical cuts
  /// \param hfCandProng2 is candidate
  /// \param trackPositron is the track with the positron hypothesis
  /// \return true if candidate passes all cuts for the given Conjugate
  template <typename T1, typename T2>
  bool selectionTopolConjugate(const T1& hfCandProng2, const T2& trackPositron,const T2& trackElectron)
  {

    auto candpT = hfCandProng2.pt();
    int pTBin = getpTBin(candpT);
    if (pTBin == -1) {
      return false;
    }

      if (TMath::Abs(InvMassJpsi(hfCandProng2) - RecoDecay::getMassPDG(443)) > cuts[pTBin][0]) {
        return false;
      }
      
    if ((TMath::Abs(trackElectron.pt()) < TMath::Abs(cuts[pTBin][3]))|| (TMath::Abs(trackPositron.pt()) < TMath::Abs(cuts[pTBin][3]))) {
      return false; //cut on daughter pT
    }
      if (TMath::Abs(trackElectron.dcaPrim0()) > cuts[pTBin][1] || TMath::Abs(trackPositron.dcaPrim0()) > cuts[pTBin][1]) {
       return false; //cut on daughter dca - need to add secondary vertex constraint here
      }
      if (TMath::Abs(trackElectron.dcaPrim1()) > cuts[pTBin][2]||TMath::Abs(trackPositron.dcaPrim1()) > cuts[pTBin][2]) {
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

  /// Check if track is compatible with given TPC Nsigma cut for a given flavour hypothesis
  /// \param track is the track
  /// \param nPDG is the flavour hypothesis PDG number
  /// \param nSigmaCut is the nsigma threshold to test against
  /// \note nPDG=11 electron  
  /// \return true if track satisfies TPC PID hypothesis for given Nsigma cut
  template <typename T>
  bool selectionPIDTPC(const T& track, int nPDG, int nSigmaCut)
  {
    double nSigma = 100.0; //arbitarily large value
    nPDG = TMath::Abs(nPDG);
    if (nPDG == 11) {
      nSigma = track.tpcNSigmaEl();
    } else {
      return false;
    }
    return nSigma < nSigmaCut;
  }


  /// PID selection on daughter track
  /// \param track is the daughter track
  /// \param nPDG is the PDG code of the flavour hypothesis
  /// \note nPDG=11 electron
  /// \return 1 if successful PID match, 0 if successful PID rejection, -1 if no PID info
  template <typename T>
  int selectionPID(const T& track, int nPDG)
  {
    int statusTPC = -1;
   // int statusTOF = -1;

    if (validTPCPID(track)) {
      if (!selectionPIDTPC(track, nPDG, d_nSigmaTPC)) {
       
          statusTPC = 0; //rejected by PID
         }else {
          statusTPC = 1; //positive PID
        }
      }  else {
      statusTPC = -1; //no PID info
    }

    if (statusTPC == 1) {
      return 1; //what if we have 2 && 0 ?
    }
     else if (statusTPC == 0) {
      return 0;
    } else {
      return -1;
    }
  }

  void process(aod::HfCandProng2 const& hfCandProng2s, aod::BigTracksPID const& tracks)
  {
    int statusJpsi; // final selection flag : 0-rejected  1-accepted
    bool topolJpsi;
    int pidJpsi, electronPlus, electronMinus;

    for (auto& hfCandProng2 : hfCandProng2s) { //looping over 2 prong candidates

      auto trackPos = hfCandProng2.index0_as<aod::BigTracksPID>(); //positive daughter
      auto trackNeg = hfCandProng2.index1_as<aod::BigTracksPID>(); //negative daughter

      statusJpsi = 0;
      topolJpsi = true;
      pidJpsi = -1;
      electronPlus = -1;
      electronMinus = -1;
     

      // daughter track validity selection
      if (!daughterSelection(trackPos) || !daughterSelection(trackNeg)) {
        hfSelJpsiCandidate(statusJpsi);
        continue;
      }

      //implement filter bit 4 cut - should be done before this task at the track selection level
      //need to add special cuts (additional cuts on decay length and d0 norm)

      //conjugate independent topological selection
      if (!selectionTopol(hfCandProng2)) {
        hfSelJpsiCandidate(statusJpsi);
        continue;
      }

      //conjugate dependent toplogical selection for Jpsi
      topolJpsi = selectionTopolConjugate(hfCandProng2, trackPos, trackNeg);

      if (!topolJpsi) {
        hfSelJpsiCandidate(statusJpsi);
        continue;
      }

      electronPlus = selectionPID(trackPos, 11);
      electronMinus = selectionPID(trackNeg, 11);
      

      if (electronPlus == 0 || electronMinus == 0 ) {
        pidJpsi = 0; //exclude Jpsi
      }
      if (electronPlus == 1 && electronMinus == 1) {
        pidJpsi = 1; //accept Jpsi
      }
      

      if (pidJpsi == 0) {
        hfSelJpsiCandidate(statusJpsi);
        continue;
      }

      if ((pidJpsi == -1 || pidJpsi==1) && topolJpsi)
 {
        statusJpsi = 1; //identified as Jpsi
      }

      hfSelJpsiCandidate(statusJpsi);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFJpsiCandidateSelector>("hf-jpsi-candidate-selector")};
}
