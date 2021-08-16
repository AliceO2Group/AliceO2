// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFJpsiCandidateSelector.cxx
/// \brief J/ψ → e+ e−, μ+ μ− selection task
///
/// \author Biao Zhang <biao.zhang@cern.ch>, CCNU
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "AnalysisCore/TrackSelectorPID.h"
#include "ALICE3Analysis/RICH.h"
#include "ALICE3Analysis/MID.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::analysis::hf_cuts_jpsi_toee;

namespace o2::aod
{
namespace hf_track_index_alice3_pid
{
DECLARE_SOA_INDEX_COLUMN(Track, track); //!
DECLARE_SOA_INDEX_COLUMN(RICH, rich);   //!
DECLARE_SOA_INDEX_COLUMN(MID, mid);     //!
} // namespace hf_track_index_alice3_pid

DECLARE_SOA_INDEX_TABLE_USER(HfTrackIndexALICE3PID, Tracks, "HFTRKIDXA3PID", //!
                             hf_track_index_alice3_pid::TrackId,
                             hf_track_index_alice3_pid::RICHId,
                             hf_track_index_alice3_pid::MIDId);
} // namespace o2::aod

struct Alice3PidIndexBuilder {
  Builds<o2::aod::HfTrackIndexALICE3PID> index;
  void init(o2::framework::InitContext&) {}
};

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec isAlice3{"isAlice3", VariantType::Bool, false, {"Switch between ALICE 2 and ALICE 3 detector setup"}};
  workflowOptions.push_back(isAlice3);
}

#include "Framework/runDataProcessing.h"

/// Struct for applying J/ψ → e+ e−, μ+ μ− selection cuts
struct HfJpsiCandidateSelector {
  Produces<aod::HFSelJpsiCandidate> hfSelJpsiCandidate;

  Configurable<bool> selectENotPi{"selectENotPi", true, "Apply combined TOF + RICH e/π selection"};
  Configurable<double> d_pTCandMin{"d_pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 50., "Upper bound of candidate pT"};
  // TPC
  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 10., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC only"};
  // TOF
  Configurable<double> d_pidTOFMinpT{"d_pidTOFMinpT", 0.15, "Lower bound of track pT for TOF PID"};
  Configurable<double> d_pidTOFMaxpT{"d_pidTOFMaxpT", 5., "Upper bound of track pT for TOF PID"};
  Configurable<double> d_nSigmaTOF{"d_nSigmaTOF", 3., "Nsigma cut on TOF only"};
  Configurable<double> d_nSigmaTOFCombined{"d_nSigmaTOFCombined", 5., "Nsigma cut on TOF combined with TPC"};
  // RICH
  Configurable<double> d_pidRICHMinpT{"d_pidRICHMinpT", 0.15, "Lower bound of track pT for RICH PID"};
  Configurable<double> d_pidRICHMaxpT{"d_pidRICHMaxpT", 10., "Upper bound of track pT for RICH PID"};
  Configurable<double> d_nSigmaRICH{"d_nSigmaRICH", 3., "Nsigma cut on RICH only"};
  Configurable<double> d_nSigmaRICHCombinedTOF{"d_nSigmaRICHCombinedTOF", 5., "Nsigma cut on RICH combined with TOF"};
  // topological cuts
  Configurable<std::vector<double>> pTBins{"pTBins", std::vector<double>{hf_cuts_jpsi_toee::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"Jpsi_to_ee_cuts", {hf_cuts_jpsi_toee::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "Jpsi candidate selection per pT bin"};

  /// Conjugate-independent topological cuts
  /// \param candidate is candidate
  /// \param trackPos is the positive track
  /// \param trackNeg is the negative track
  /// \return true if candidate passes all cuts
  template <typename T1, typename T2>
  bool selectionTopol(const T1& candidate, const T2& trackPos, const T2& trackNeg, int& selEE, int& selMuMu)
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

    // cut on e+ e− invariant mass
    if (std::abs(InvMassJpsiToEE(candidate) - RecoDecay::getMassPDG(pdg::Code::kJpsi)) > cuts->get(pTBin, "m")) {
      selEE = 0;
    }

    // cut on μ+ μ− invariant mass
    if (std::abs(InvMassJpsiToMuMu(candidate) - RecoDecay::getMassPDG(pdg::Code::kJpsi)) > cuts->get(pTBin, "m")) {
      selMuMu = 0;
    }

    if (selEE == 0 && selMuMu == 0) {
      return false;
    }

    // cut on daughter pT (same cut used for both channels)
    if (trackNeg.pt() < cuts->get(pTBin, "pT El") || trackPos.pt() < cuts->get(pTBin, "pT El")) {
      return false;
    }

    // cut on daughter DCA - need to add secondary vertex constraint here
    if (std::abs(trackNeg.dcaPrim0()) > cuts->get(pTBin, "DCA_xy") || std::abs(trackPos.dcaPrim0()) > cuts->get(pTBin, "DCA_xy")) {
      return false;
    }

    // cut on daughter DCA - need to add secondary vertex constraint here
    if (std::abs(trackNeg.dcaPrim1()) > cuts->get(pTBin, "DCA_z") || std::abs(trackPos.dcaPrim1()) > cuts->get(pTBin, "DCA_z")) {
      return false;
    }

    // cut on chi2 point of closest approach
    if (std::abs(candidate.chi2PCA()) > cuts->get(pTBin, "chi2PCA")) {
      return false;
    }
    return true;
  }

  using TracksPID = soa::Join<aod::BigTracksPID, aod::HfTrackIndexALICE3PID>;

  void processAlice2(aod::HfCandProng2 const& candidates, aod::BigTracksPID const&)
  {
    TrackSelectorPID selectorElectron(kElectron);
    selectorElectron.setRangePtTPC(d_pidTPCMinpT, d_pidTPCMaxpT);
    selectorElectron.setRangeNSigmaTPC(-d_nSigmaTPC, d_nSigmaTPC);
    selectorElectron.setRangePtTOF(d_pidTOFMinpT, d_pidTOFMaxpT);
    selectorElectron.setRangeNSigmaTOF(-d_nSigmaTOF, d_nSigmaTOF);
    selectorElectron.setRangeNSigmaTOFCondTPC(-d_nSigmaTOFCombined, d_nSigmaTOFCombined);
    selectorElectron.setRangePtRICH(d_pidRICHMinpT, d_pidRICHMaxpT);
    selectorElectron.setRangeNSigmaRICH(-d_nSigmaRICH, d_nSigmaRICH);
    selectorElectron.setRangeNSigmaRICHCondTOF(-d_nSigmaRICHCombinedTOF, d_nSigmaRICHCombinedTOF);

    // looping over 2-prong candidates
    for (auto& candidate : candidates) {

      if (!(candidate.hfflag() & 1 << DecayType::JpsiToEE) && !(candidate.hfflag() & 1 << DecayType::JpsiToMuMu)) {
        hfSelJpsiCandidate(0, 0, 0, 0, 0, 0, 0, 0, 0);
        //hfSelJpsiCandidate(0, 0);
        continue;
      }

      auto trackPos = candidate.index0_as<aod::BigTracksPID>(); // positive daughter
      auto trackNeg = candidate.index1_as<aod::BigTracksPID>(); // negative daughter

      int selectedEETopol = 1;
      int selectedEETpc = 1;
      int selectedEETof = 1;
      int selectedEERich = 1;
      int selectedEETofRich = 1;
      int selectedMuMuTopol = 1;
      int selectedMuMuMid = 1;
      int selectedEE = 1;
      int selectedMuMu = 1;

      // track selection level need to add special cuts (additional cuts on decay length and d0 norm)

      if (!selectionTopol(candidate, trackPos, trackNeg, selectedEETopol, selectedMuMuTopol)) {
        selectedEETopol = 0;
        selectedMuMuTopol = 0;
        selectedEE = 0;
        selectedMuMu = 0;
        //if (!selectionTopol(candidate, trackPos, trackNeg, selectedEE, selectedMuMu)) {
        //hfSelJpsiCandidate(0, 0);
        //continue;
      }

      // track-level electron PID TOF selection
      if (selectorElectron.getStatusTrackPIDTOF(trackPos) == TrackSelectorPID::Status::PIDRejected ||
          selectorElectron.getStatusTrackPIDTOF(trackNeg) == TrackSelectorPID::Status::PIDRejected) {
        selectedEETof = 0;
        selectedEE = 0;
        //if (selectedMuMu == 0) {
        //  hfSelJpsiCandidate(0, 0);
        //  continue;
        //}
      }

      // track-level electron PID TPC selection
      if (selectorElectron.getStatusTrackPIDTPC(trackPos) == TrackSelectorPID::Status::PIDRejected ||
          selectorElectron.getStatusTrackPIDTPC(trackNeg) == TrackSelectorPID::Status::PIDRejected) {
        selectedEETpc = 0;
        selectedEE = 0;
      }

      hfSelJpsiCandidate(selectedEE,
                         selectedMuMu,
                         selectedEETopol,
                         selectedEETpc,
                         selectedEETof,
                         selectedEERich,
                         selectedEETofRich,
                         selectedMuMuTopol,
                         selectedMuMuMid);
      //hfSelJpsiCandidate(selectedEE, selectedMuMu);
    }
  }

  PROCESS_SWITCH(HfJpsiCandidateSelector, processAlice2, "Use ALICE 2 detector setup", true);

  void processAlice3(aod::HfCandProng2 const& candidates, TracksPID const&, aod::RICHs const&, aod::MIDs const&)
  {
    TrackSelectorPID selectorElectron(kElectron);
    selectorElectron.setRangePtTPC(d_pidTPCMinpT, d_pidTPCMaxpT);
    selectorElectron.setRangeNSigmaTPC(-d_nSigmaTPC, d_nSigmaTPC);
    selectorElectron.setRangePtTOF(d_pidTOFMinpT, d_pidTOFMaxpT);
    selectorElectron.setRangeNSigmaTOF(-d_nSigmaTOF, d_nSigmaTOF);
    selectorElectron.setRangeNSigmaTOFCondTPC(-d_nSigmaTOFCombined, d_nSigmaTOFCombined);
    selectorElectron.setRangePtRICH(d_pidRICHMinpT, d_pidRICHMaxpT);
    selectorElectron.setRangeNSigmaRICH(-d_nSigmaRICH, d_nSigmaRICH);
    selectorElectron.setRangeNSigmaRICHCondTOF(-d_nSigmaRICHCombinedTOF, d_nSigmaRICHCombinedTOF);

    TrackSelectorPID selectorMuon(kMuonMinus);

    // looping over 2-prong candidates
    for (auto& candidate : candidates) {

      if (!(candidate.hfflag() & 1 << DecayType::JpsiToEE) && !(candidate.hfflag() & 1 << DecayType::JpsiToMuMu)) {
        hfSelJpsiCandidate(0, 0, 0, 0, 0, 0, 0, 0, 0);
        //hfSelJpsiCandidate(0, 0);
        continue;
      }

      auto trackPos = candidate.index0_as<TracksPID>(); // positive daughter
      auto trackNeg = candidate.index1_as<TracksPID>(); // negative daughter

      int selectedEETopol = 1;
      int selectedEETpc = 1;
      int selectedEETof = 1;
      int selectedEERich = 1;
      int selectedEETofRich = 1;
      int selectedMuMuTopol = 1;
      int selectedMuMuMid = 1;
      int selectedEE = 1;
      int selectedMuMu = 1;

      // track selection level need to add special cuts (additional cuts on decay length and d0 norm)

      if (!selectionTopol(candidate, trackPos, trackNeg, selectedEETopol, selectedMuMuTopol)) {
        selectedEETopol = 0;
        selectedMuMuTopol = 0;
        selectedEE = 0;
        selectedMuMu = 0;
        //if (!selectionTopol(candidate, trackPos, trackNeg, selectedEE, selectedMuMu)) {
        //hfSelJpsiCandidate(0, 0);
        //continue;
      }

      //if (selectENotPi) {
      // combined TOF + RICH e selection with π rejection
      if (!selectorElectron.isElectronAndNotPion(trackPos) ||
          !selectorElectron.isElectronAndNotPion(trackNeg)) {
        selectedEETofRich = 0;
        selectedEE = 0;
      }
      //} else {
      // track-level electron PID TOF selection
      if (selectorElectron.getStatusTrackPIDTOF(trackPos) == TrackSelectorPID::Status::PIDRejected ||
          selectorElectron.getStatusTrackPIDTOF(trackNeg) == TrackSelectorPID::Status::PIDRejected) {
        selectedEETof = 0;
        selectedEE = 0;
        //if (selectedMuMu == 0) {
        //  hfSelJpsiCandidate(0, 0);
        //  continue;
        //}
      }

      // track-level electron PID RICH selection
      if (selectorElectron.getStatusTrackPIDRICH(trackPos) == TrackSelectorPID::Status::PIDRejected ||
          selectorElectron.getStatusTrackPIDRICH(trackNeg) == TrackSelectorPID::Status::PIDRejected) {
        selectedEERich = 0;
        selectedEE = 0;
      }
      //}

      //if (selectedEE == 0 && selectedMuMu == 0) {
      //  hfSelJpsiCandidate(0, 0);
      //  continue;
      //}

      // track-level muon PID MID selection
      if (selectorMuon.getStatusTrackPIDMID(trackPos) != TrackSelectorPID::Status::PIDAccepted ||
          selectorMuon.getStatusTrackPIDMID(trackNeg) != TrackSelectorPID::Status::PIDAccepted) {
        selectedMuMuMid = 0;
        selectedMuMu = 0;
      }

      hfSelJpsiCandidate(selectedEE,
                         selectedMuMu,
                         selectedEETopol,
                         selectedEETpc,
                         selectedEETof,
                         selectedEERich,
                         selectedEETofRich,
                         selectedMuMuTopol,
                         selectedMuMuMid);
      //hfSelJpsiCandidate(selectedEE, selectedMuMu);
    }
  }

  PROCESS_SWITCH(HfJpsiCandidateSelector, processAlice3, "Use ALICE 3 detector setup", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{};
  const bool isAlice3 = cfgc.options().get<bool>("isAlice3");
  if (isAlice3) {
    workflow.push_back(adaptAnalysisTask<Alice3PidIndexBuilder>(cfgc));
    workflow.push_back(adaptAnalysisTask<HfJpsiCandidateSelector>(cfgc, SetDefaultProcesses{{{"processAlice2", false}, {"processAlice3", true}}}));
  } else {
    workflow.push_back(adaptAnalysisTask<HfJpsiCandidateSelector>(cfgc));
  }
  return workflow;
}
