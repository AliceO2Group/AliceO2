// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFXicToPKPiCandidateSelector.cxx
/// \brief Ξc± → p± K∓ π± selection task
/// \note Inspired from HFLcCandidateSelector.cxx
///
/// \author Mattia Faggin <mattia.faggin@cern.ch>, University and INFN PADOVA

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "AnalysisCore/HFSelectorCuts.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;
using namespace o2::analysis::hf_cuts_xic_topkpi;

/// Struct for applying Xic selection cuts
struct HFXicToPKPiCandidateSelector {

  Produces<aod::HFSelXicToPKPiCandidate> hfSelXicToPKPiCandidate;

  Configurable<bool> d_FilterPID{"d_FilterPID", true, "Bool to use or not the PID at filtering level"};

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

  Configurable<std::vector<double>> pTBins{"pTBins", std::vector<double>{hf_cuts_xic_topkpi::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"Xic_to_p_K_pi_cuts", {hf_cuts_xic_topkpi::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "Xic candidate selection per pT bin"};

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

  /// Conjugate-independent topological cuts
  /// \param hfCandProng3 is candidate
  /// \return true if candidate passes all cuts
  template <typename T>
  bool selectionTopol(const T& hfCandProng3)
  {
    auto candpT = hfCandProng3.pt();
    int pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      return false;
    }

    if (candpT < d_pTCandMin || candpT >= d_pTCandMax) {
      return false; //check that the candidate pT is within the analysis range
    }

    if (hfCandProng3.cpa() <= cuts->get(pTBin, "cos pointing angle")) {
      return false; //cosine of pointing angle
    }

    /*  if (hfCandProng3.chi2PCA() > cuts->get(pTBin, "DCA")) { //candidate DCA
      return false;
      }*/

    if (hfCandProng3.decayLength() <= cuts->get(pTBin, "decay length")) {
      return false;
    }
    return true;
  }

  /// Conjugate-dependent topological cuts
  /// \param hfCandProng3 is candidate
  /// \param trackProton is the track with the proton hypothesis
  /// \param trackPion is the track with the pion hypothesis
  /// \param trackKaon is the track with the kaon hypothesis
  /// \return true if candidate passes all cuts for the given Conjugate
  template <typename T1, typename T2>
  bool selectionTopolConjugate(const T1& hfCandProng3, const T2& trackProton, const T2& trackKaon, const T2& trackPion)
  {

    auto candpT = hfCandProng3.pt();
    int pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      return false;
    }

    if (trackProton.pt() < cuts->get(pTBin, "pT p") || trackKaon.pt() < cuts->get(pTBin, "pT K") || trackPion.pt() < cuts->get(pTBin, "pT Pi")) {
      return false; //cut on daughter pT
    }

    if (trackProton.globalIndex() == hfCandProng3.index0Id()) {
      if (std::abs(InvMassXicToPKPi(hfCandProng3) - RecoDecay::getMassPDG(pdg::Code::kXiCPlus)) > cuts->get(pTBin, "m")) {
        return false;
      }
    } else {
      if (std::abs(InvMassXicToPiKP(hfCandProng3) - RecoDecay::getMassPDG(pdg::Code::kXiCPlus)) > cuts->get(pTBin, "m")) {
        return false;
      }
    }

    return true;
  }

  void process(aod::HfCandProng3 const& hfCandProng3s, aod::BigTracksPID const&)
  {
    for (auto& hfCandProng3 : hfCandProng3s) { //looping over 3 prong candidates

      // final selection flag : 0-rejected  1-accepted
      auto statusXicToPKPi = 0;
      auto statusXicToPiKP = 0;

      if (!(hfCandProng3.hfflag() & 1 << DecayType::XicToPKPi)) {
        hfSelXicToPKPiCandidate(statusXicToPKPi, statusXicToPiKP);
        continue;
      }

      auto trackPos1 = hfCandProng3.index0_as<aod::BigTracksPID>(); //positive daughter (negative for the antiparticles)
      auto trackNeg1 = hfCandProng3.index1_as<aod::BigTracksPID>(); //negative daughter (positive for the antiparticles)
      auto trackPos2 = hfCandProng3.index2_as<aod::BigTracksPID>(); //positive daughter (negative for the antiparticles)

      bool topolXicToPKPi = true;
      bool topolXicToPiKP = true;
      auto pidXicToPKPi = -1;
      auto pidXicToPiKP = -1;
      auto kaonNeg = -1;
      auto prot1 = -1;
      auto prot2 = -1;
      auto pion1 = -1;
      auto pion2 = -1;
      //proton = -1;
      //kaonMinus = -1;
      //pionPlus = -1;

      // daughter track validity selection
      if (!daughterSelection(trackPos1) || !daughterSelection(trackNeg1) || !daughterSelection(trackPos2)) {
        hfSelXicToPKPiCandidate(statusXicToPKPi, statusXicToPiKP);
        continue;
      }

      //implement filter bit 4 cut - should be done before this task at the track selection level

      //conjugate-independent topological selection
      if (!selectionTopol(hfCandProng3)) {
        hfSelXicToPKPiCandidate(statusXicToPKPi, statusXicToPiKP);
        continue;
      }

      //conjugate-dependent topplogical selection for Xic

      topolXicToPKPi = selectionTopolConjugate(hfCandProng3, trackPos1, trackNeg1, trackPos2);
      topolXicToPiKP = selectionTopolConjugate(hfCandProng3, trackPos2, trackNeg1, trackPos1);

      if (!topolXicToPKPi && !topolXicToPiKP) {
        hfSelXicToPKPiCandidate(statusXicToPKPi, statusXicToPiKP);
        continue;
      }

      /*
      proton = getStatusTrackPIDAll(trackPos1, 2212);
      kaonMinus = getStatusTrackPIDAll(trackNeg1, 321);
      pionPlus = getStatusTrackPIDAll(trackPos2, 211);

      if (proton == 0 || kaonMinus == 0 || pionPlus == 0) {
        pidLc = 0; //exclude Lc
      }
      if (proton == 1 && kaonMinus == 1 && pionPlus == 1) {
        pidLc = 1; //accept Lc
      }

      if (pidLc == 0) {
        hfSelLcCandidate(statusLcpKpi, statusLcpiKp);
        continue;
      }

      if ((pidLc == -1 || pidLc == 1) && topolLcpKpi) {
        statusLcpKpi = 1; //identified as Lc
      }
      if ((pidLc == -1 || pidLc == 1) && topolLcpiKp) {
        statusLcpiKp = 1; //identified as Lc
      }
      */

      kaonNeg = pid::getStatusTrackPIDAll(trackNeg1, kKPlus, d_pidTPCMinpT, d_pidTPCMaxpT, d_nSigmaTPC, d_nSigmaTPCCombined, d_pidTOFMinpT, d_pidTOFMaxpT, d_nSigmaTOF, d_nSigmaTOFCombined);
      prot1 = pid::getStatusTrackPIDAll(trackPos1, kProton, d_pidTPCMinpT, d_pidTPCMaxpT, d_nSigmaTPC, d_nSigmaTPCCombined, d_pidTOFMinpT, d_pidTOFMaxpT, d_nSigmaTOF, d_nSigmaTOFCombined);
      prot2 = pid::getStatusTrackPIDAll(trackPos2, kProton, d_pidTPCMinpT, d_pidTPCMaxpT, d_nSigmaTPC, d_nSigmaTPCCombined, d_pidTOFMinpT, d_pidTOFMaxpT, d_nSigmaTOF, d_nSigmaTOFCombined);
      pion1 = pid::getStatusTrackPIDAll(trackPos1, kPiPlus, d_pidTPCMinpT, d_pidTPCMaxpT, d_nSigmaTPC, d_nSigmaTPCCombined, d_pidTOFMinpT, d_pidTOFMaxpT, d_nSigmaTOF, d_nSigmaTOFCombined);
      pion2 = pid::getStatusTrackPIDAll(trackPos2, kPiPlus, d_pidTPCMinpT, d_pidTPCMaxpT, d_nSigmaTPC, d_nSigmaTPCCombined, d_pidTOFMinpT, d_pidTOFMaxpT, d_nSigmaTOF, d_nSigmaTOFCombined);

      if (!d_FilterPID) {
        /// PID not applied at filtering level
        /// set the PID flags to 1
        pidXicToPKPi = 1;
        pidXicToPiKP = 1;
      } else {
        /// PID applied at filtering level
        /// Accept only candidates recognised either at least as pKpi or as piKp
        if (std::abs(kaonNeg) == 1) {
          if (std::abs(prot1) == 1 && std::abs(pion2) == 1) {
            pidXicToPKPi = 1;
          }
          if (std::abs(pion1) == 1 && std::abs(prot2) == 1) {
            pidXicToPiKP = 1;
          }
        }
      }

      if ((pidXicToPKPi == 1) && topolXicToPKPi) {
        statusXicToPKPi = 1; //identified as Xic->pKpi
      }
      if ((pidXicToPiKP == 1) && topolXicToPiKP) {
        statusXicToPiKP = 1; //identified as Xic->piKp
      }

      hfSelXicToPKPiCandidate(statusXicToPKPi, statusXicToPiKP);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFXicToPKPiCandidateSelector>(cfgc, TaskName{"hf-xic-topkpi-candidate-selector"})};
}
