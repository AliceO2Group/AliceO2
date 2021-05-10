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
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "AnalysisCore/TrackSelectorPID.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;
using namespace o2::analysis::hf_cuts_xic_topkpi;

/// Struct for applying Xic selection cuts
struct HFXicToPKPiCandidateSelector {
  Produces<aod::HFSelXicToPKPiCandidate> hfSelXicToPKPiCandidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 36., "Upper bound of candidate pT"};
  Configurable<bool> d_FilterPID{"d_FilterPID", true, "Bool to use or not the PID at filtering level"};
  // TPC
  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 1., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC only"};
  Configurable<double> d_nSigmaTPCCombined{"d_nSigmaTPCCombined", 5., "Nsigma cut on TPC combined with TOF"};
  //Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 70., "Lower bound of TPC findable clusters for good PID"};
  // TOF
  Configurable<double> d_pidTOFMinpT{"d_pidTOFMinpT", 0.5, "Lower bound of track pT for TOF PID"};
  Configurable<double> d_pidTOFMaxpT{"d_pidTOFMaxpT", 4., "Upper bound of track pT for TOF PID"};
  Configurable<double> d_nSigmaTOF{"d_nSigmaTOF", 3., "Nsigma cut on TOF only"};
  Configurable<double> d_nSigmaTOFCombined{"d_nSigmaTOFCombined", 5., "Nsigma cut on TOF combined with TPC"};
  // topological cuts
  Configurable<std::vector<double>> pTBins{"pTBins", std::vector<double>{hf_cuts_xic_topkpi::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"Xic_to_p_K_pi_cuts", {hf_cuts_xic_topkpi::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "Xic candidate selection per pT bin"};

  /*
  /// Selection on goodness of daughter tracks
  /// \note should be applied at candidate selection
  /// \param track is daughter track
  /// \return true if track is good
  template <typename T>
  bool daughterSelection(const T& track)
  {
    if (track.tpcNClsFound() == 0) {
      return false; //is it clusters findable or found - need to check
    }
    return true;
  }
  */

  /// Conjugate-independent topological cuts
  /// \param candidate is candidate
  /// \return true if candidate passes all cuts
  template <typename T>
  bool selectionTopol(const T& candidate)
  {
    auto candpT = candidate.pt();
    int pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      return false;
    }

    // check that the candidate pT is within the analysis range
    if (candpT < d_pTCandMin || candpT >= d_pTCandMax) {
      return false;
    }

    // cosine of pointing angle
    if (candidate.cpa() <= cuts->get(pTBin, "cos pointing angle")) {
      return false;
    }

    // candidate DCA
    /*  if (candidate.chi2PCA() > cuts->get(pTBin, "DCA")) {
      return false;
      }*/

    if (candidate.decayLength() <= cuts->get(pTBin, "decay length")) {
      return false;
    }
    return true;
  }

  /// Conjugate-dependent topological cuts
  /// \param candidate is candidate
  /// \param trackProton is the track with the proton hypothesis
  /// \param trackPion is the track with the pion hypothesis
  /// \param trackKaon is the track with the kaon hypothesis
  /// \return true if candidate passes all cuts for the given Conjugate
  template <typename T1, typename T2>
  bool selectionTopolConjugate(const T1& candidate, const T2& trackProton, const T2& trackKaon, const T2& trackPion)
  {

    auto candpT = candidate.pt();
    int pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      return false;
    }

    // cut on daughter pT
    if (trackProton.pt() < cuts->get(pTBin, "pT p") || trackKaon.pt() < cuts->get(pTBin, "pT K") || trackPion.pt() < cuts->get(pTBin, "pT Pi")) {
      return false;
    }

    if (trackProton.globalIndex() == candidate.index0Id()) {
      if (std::abs(InvMassXicToPKPi(candidate) - RecoDecay::getMassPDG(pdg::Code::kXiCPlus)) > cuts->get(pTBin, "m")) {
        return false;
      }
    } else {
      if (std::abs(InvMassXicToPiKP(candidate) - RecoDecay::getMassPDG(pdg::Code::kXiCPlus)) > cuts->get(pTBin, "m")) {
        return false;
      }
    }

    return true;
  }

  void process(aod::HfCandProng3 const& candidates, aod::BigTracksPID const&)
  {
    TrackSelectorPID selectorPion(kPiPlus);
    selectorPion.setRangePtTPC(d_pidTPCMinpT, d_pidTPCMaxpT);
    selectorPion.setRangeNSigmaTPC(-d_nSigmaTPC, d_nSigmaTPC);
    selectorPion.setRangeNSigmaTPCCondTOF(-d_nSigmaTPCCombined, d_nSigmaTPCCombined);
    selectorPion.setRangePtTOF(d_pidTOFMinpT, d_pidTOFMaxpT);
    selectorPion.setRangeNSigmaTOF(-d_nSigmaTOF, d_nSigmaTOF);
    selectorPion.setRangeNSigmaTOFCondTPC(-d_nSigmaTOFCombined, d_nSigmaTOFCombined);

    TrackSelectorPID selectorKaon(selectorPion);
    selectorKaon.setPDG(kKPlus);

    TrackSelectorPID selectorProton(selectorPion);
    selectorProton.setPDG(kProton);

    // looping over 3-prong candidates
    for (auto& candidate : candidates) {

      // final selection flag: 0 - rejected, 1 - accepted
      auto statusXicToPKPi = 0;
      auto statusXicToPiKP = 0;

      if (!(candidate.hfflag() & 1 << DecayType::XicToPKPi)) {
        hfSelXicToPKPiCandidate(statusXicToPKPi, statusXicToPiKP);
        continue;
      }

      auto trackPos1 = candidate.index0_as<aod::BigTracksPID>(); // positive daughter (negative for the antiparticles)
      auto trackNeg = candidate.index1_as<aod::BigTracksPID>();  // negative daughter (positive for the antiparticles)
      auto trackPos2 = candidate.index2_as<aod::BigTracksPID>(); // positive daughter (negative for the antiparticles)

      /*
      // daughter track validity selection
      if (!daughterSelection(trackPos1) || !daughterSelection(trackNeg) || !daughterSelection(trackPos2)) {
        hfSelXicToPKPiCandidate(statusXicToPKPi, statusXicToPiKP);
        continue;
      }
      */

      // implement filter bit 4 cut - should be done before this task at the track selection level

      // conjugate-independent topological selection
      if (!selectionTopol(candidate)) {
        hfSelXicToPKPiCandidate(statusXicToPKPi, statusXicToPiKP);
        continue;
      }

      // conjugate-dependent topplogical selection for Xic

      bool topolXicToPKPi = selectionTopolConjugate(candidate, trackPos1, trackNeg, trackPos2);
      bool topolXicToPiKP = selectionTopolConjugate(candidate, trackPos2, trackNeg, trackPos1);

      if (!topolXicToPKPi && !topolXicToPiKP) {
        hfSelXicToPKPiCandidate(statusXicToPKPi, statusXicToPiKP);
        continue;
      }

      auto pidXicToPKPi = -1;
      auto pidXicToPiKP = -1;

      if (!d_FilterPID) {
        // PID non applied
        pidXicToPKPi = 1;
        pidXicToPiKP = 1;
      } else {
        // track-level PID selection
        int pidTrackPos1Proton = selectorProton.getStatusTrackPIDAll(trackPos1);
        int pidTrackPos2Proton = selectorProton.getStatusTrackPIDAll(trackPos2);
        int pidTrackPos1Pion = selectorPion.getStatusTrackPIDAll(trackPos1);
        int pidTrackPos2Pion = selectorPion.getStatusTrackPIDAll(trackPos2);
        int pidTrackNegKaon = selectorKaon.getStatusTrackPIDAll(trackNeg);

        if (pidTrackPos1Proton == TrackSelectorPID::Status::PIDAccepted &&
            pidTrackNegKaon == TrackSelectorPID::Status::PIDAccepted &&
            pidTrackPos2Pion == TrackSelectorPID::Status::PIDAccepted) {
          pidXicToPKPi = 1; // accept LcpKpi
        } else if (pidTrackPos1Proton == TrackSelectorPID::Status::PIDRejected ||
                   pidTrackNegKaon == TrackSelectorPID::Status::PIDRejected ||
                   pidTrackPos2Pion == TrackSelectorPID::Status::PIDRejected) {
          pidXicToPKPi = 0; // exclude LcpKpi
        }
        if (pidTrackPos2Proton == TrackSelectorPID::Status::PIDAccepted &&
            pidTrackNegKaon == TrackSelectorPID::Status::PIDAccepted &&
            pidTrackPos1Pion == TrackSelectorPID::Status::PIDAccepted) {
          pidXicToPiKP = 1; // accept LcpiKp
        } else if (pidTrackPos1Pion == TrackSelectorPID::Status::PIDRejected ||
                   pidTrackNegKaon == TrackSelectorPID::Status::PIDRejected ||
                   pidTrackPos2Proton == TrackSelectorPID::Status::PIDRejected) {
          pidXicToPiKP = 0; // exclude LcpiKp
        }
      }

      if (pidXicToPKPi == 0 && pidXicToPiKP == 0) {
        hfSelXicToPKPiCandidate(statusXicToPKPi, statusXicToPiKP);
        continue;
      }

      if ((pidXicToPKPi == -1 || pidXicToPKPi == 1) && topolXicToPKPi) {
        statusXicToPKPi = 1; // identified as Xic->pKpi
      }
      if ((pidXicToPiKP == -1 || pidXicToPiKP == 1) && topolXicToPiKP) {
        statusXicToPiKP = 1; // identified as Xic->piKp
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
