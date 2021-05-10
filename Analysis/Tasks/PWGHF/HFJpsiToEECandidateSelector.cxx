// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFJpsiToEECandidateSelector.cxx
/// \brief J/ψ → e+ e− selection task
///
/// \author Biao Zhang <biao.zhang@cern.ch>, CCNU
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "AnalysisCore/TrackSelectorPID.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::analysis::hf_cuts_jpsi_toee;

/// Struct for applying J/ψ → e+ e− selection cuts
struct HFJpsiToEECandidateSelector {
  Produces<aod::HFSelJpsiToEECandidate> hfSelJpsiToEECandidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 50., "Upper bound of candidate pT"};
  // TPC
  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 10., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC only"};
  //Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 70., "Lower bound of TPC findable clusters for good PID"};
  // topological cuts
  Configurable<std::vector<double>> pTBins{"pTBins", std::vector<double>{hf_cuts_jpsi_toee::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"Jpsi_to_ee_cuts", {hf_cuts_jpsi_toee::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "Jpsi candidate selection per pT bin"};

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
  /// \param trackPositron is the track with the positron hypothesis
  /// \param trackElectron is the track with the electron hypothesis
  /// \return true if candidate passes all cuts
  template <typename T1, typename T2>
  bool selectionTopol(const T1& candidate, const T2& trackPositron, const T2& trackElectron)
  {
    auto candpT = candidate.pt();
    auto pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      return false;
    }

    // check that the candidate pT is within the analysis range
    if (candpT < d_pTCandMin || candpT >= d_pTCandMax) {
      return false;
    }

    // cut on invariant mass
    if (std::abs(InvMassJpsiToEE(candidate) - RecoDecay::getMassPDG(pdg::Code::kJpsi)) > cuts->get(pTBin, "m")) {
      return false;
    }

    // cut on daughter pT
    if (trackElectron.pt() < cuts->get(pTBin, "pT El") || trackPositron.pt() < cuts->get(pTBin, "pT El")) {
      return false;
    }

    // cut on daughter DCA - need to add secondary vertex constraint here
    if (std::abs(trackElectron.dcaPrim0()) > cuts->get(pTBin, "DCA_xy") || std::abs(trackPositron.dcaPrim0()) > cuts->get(pTBin, "DCA_xy")) {
      return false;
    }

    // cut on daughter DCA - need to add secondary vertex constraint here
    if (std::abs(trackElectron.dcaPrim1()) > cuts->get(pTBin, "DCA_z") || std::abs(trackPositron.dcaPrim1()) > cuts->get(pTBin, "DCA_z")) {
      return false;
    }

    return true;
  }

  void process(aod::HfCandProng2 const& candidates, aod::BigTracksPID const&)
  {
    TrackSelectorPID selectorElectron(kElectron);
    selectorElectron.setRangePtTPC(d_pidTPCMinpT, d_pidTPCMaxpT);
    selectorElectron.setRangeNSigmaTPC(-d_nSigmaTPC, d_nSigmaTPC);

    // looping over 2-prong candidates
    for (auto& candidate : candidates) {

      auto trackPos = candidate.index0_as<aod::BigTracksPID>(); // positive daughter
      auto trackNeg = candidate.index1_as<aod::BigTracksPID>(); // negative daughter

      if (!(candidate.hfflag() & 1 << DecayType::JpsiToEE)) {
        hfSelJpsiToEECandidate(0);
        continue;
      }

      /*
      // daughter track validity selection
      if (!daughterSelection(trackPos) || !daughterSelection(trackNeg)) {
        hfSelJpsiToEECandidate(0);
        continue;
      }
      */

      // implement filter bit 4 cut - should be done before this task at the track selection level
      // need to add special cuts (additional cuts on decay length and d0 norm)

      if (!selectionTopol(candidate, trackPos, trackNeg)) {
        hfSelJpsiToEECandidate(0);
        continue;
      }

      // track-level PID selection
      if (selectorElectron.getStatusTrackPIDTPC(trackPos) == TrackSelectorPID::Status::PIDRejected ||
          selectorElectron.getStatusTrackPIDTPC(trackNeg) == TrackSelectorPID::Status::PIDRejected) {
        hfSelJpsiToEECandidate(0);
        continue;
      }

      hfSelJpsiToEECandidate(1);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFJpsiToEECandidateSelector>(cfgc, TaskName{"hf-jpsi-toee-candidate-selector"})};
}
