// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFDplusToPiKPiCandidateSelector.cxx
/// \brief Dplus->piKpi selection task
///
/// \author Fabio Catalano <fabio.catalano@cern.ch>, Politecnico and INFN Torino

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;
using namespace o2::analysis::hf_cuts_dplus_topikpi;

/// Struct for applying Dplus to piKpi selection cuts
struct HFDplusToPiKPiCandidateSelector {
  Produces<aod::HFSelDplusToPiKPiCandidate> hfSelDplusToPiKPiCandidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 1., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 36., "Upper bound of candidate pT"};

  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 20., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_pidTOFMinpT{"d_pidTOFMinpT", 0.15, "Lower bound of track pT for TOF PID"};
  Configurable<double> d_pidTOFMaxpT{"d_pidTOFMaxpT", 20., "Upper bound of track pT for TOF PID"};

  Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 50., "Lower bound of TPC findable clusters for good PID"};
  Configurable<bool> b_requireTPC{"b_requireTPC", true, "Flag to require a positive Number of found clusters in TPC"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC"};
  Configurable<double> d_nSigmaTOF{"d_nSigmaTOF", 3., "Nsigma cut on TOF"};

  Configurable<std::vector<double>> pTBins{"pTBins", std::vector<double>{hf_cuts_dplus_topikpi::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"DPlus_to_Pi_K_Pi_cuts", {hf_cuts_dplus_topikpi::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "Dplus candidate selection per pT bin"};

  /// Selection on goodness of daughter tracks
  /// \note should be applied at candidate selection
  /// \param track is daughter track
  /// \return true if track is good
  template <typename T>
  bool daughterSelection(const T& track)
  {
    if (b_requireTPC.value && track.tpcNClsFound() == 0) {
      return false; //is it clusters findable or found - need to check
    }
    return true;
  }

  /// Candidate selections
  /// \param hfCandProng3 is candidate
  /// \param trackPion1 is the first track with the pion hypothesis
  /// \param trackKaon is the track with the kaon hypothesis
  /// \param trackPion2 is the second track with the pion hypothesis
  /// \return true if candidate passes all cuts
  template <typename T1, typename T2>
  bool selection(const T1& hfCandProng3, const T2& trackPion1, const T2& trackKaon, const T2& trackPion2)
  {
    auto candpT = hfCandProng3.pt();
    int pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      return false;
    }
    if (candpT < d_pTCandMin || candpT > d_pTCandMax) {
      return false; //check that the candidate pT is within the analysis range
    }
    if (trackPion1.pt() < cuts->get(pTBin, "pT Pi") || trackKaon.pt() < cuts->get(pTBin, "pT K") || trackPion2.pt() < cuts->get(pTBin, "pT Pi")) {
      return false; // cut on daughter pT
    }
    if (std::abs(InvMassDPlus(hfCandProng3) - RecoDecay::getMassPDG(pdg::Code::kDPlus)) > cuts->get(pTBin, "deltaM")) {
      return false; // invariant mass cut
    }
    if (hfCandProng3.decayLength() < cuts->get(pTBin, "decay length")) {
      return false;
    }
    if (hfCandProng3.decayLengthXYNormalised() < cuts->get(pTBin, "normalized decay length XY")) {
      return false;
    }
    if (hfCandProng3.cpa() < cuts->get(pTBin, "cos pointing angle")) {
      return false;
    }
    if (hfCandProng3.cpaXY() < cuts->get(pTBin, "cos pointing angle XY")) {
      return false;
    }
    if (std::abs(hfCandProng3.maxNormalisedDeltaIP()) > cuts->get(pTBin, "max normalized deltaIP")) {
      return false;
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
    if (track.pt() < d_pidTPCMinpT || track.pt() >= d_pidTPCMaxpT) {
      return false;
    }
    //if (track.TPCNClsFindable() < d_TPCNClsFindablePIDCut) {
    //  return false;
    //}
    return true;
  }

  /// Check if track is ok for TOF PID
  /// \param track is the track
  /// \note function to be expanded
  /// \return true if track is ok for TOF PID
  template <typename T>
  bool validTOFPID(const T& track)
  {
    if (track.pt() < d_pidTOFMinpT || track.pt() >= d_pidTOFMaxpT) {
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
    nPDG = std::abs(nPDG);
    if (nPDG == kPiPlus) {
      nSigma = track.tpcNSigmaPi();
    } else if (nPDG == kKPlus) {
      nSigma = track.tpcNSigmaKa();
    } else {
      return false;
    }
    return std::abs(nSigma) < nSigmaCut;
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
    nPDG = std::abs(nPDG);
    if (nPDG == kPiPlus) {
      nSigma = track.tofNSigmaPi();
    } else if (nPDG == kKPlus) {
      nSigma = track.tofNSigmaKa();
    } else {
      return false;
    }
    return std::abs(nSigma) < nSigmaCut;
  }

  /// PID selection on daughter track
  /// \param track is the daughter track
  /// \param nPDG is the PDG code of the flavour hypothesis
  /// \note nPDG=211 pion  nPDG=321 kaon
  /// \return 1 if successful PID match, 0 if successful PID rejection, -1 if no PID info
  template <typename T>
  int selectionPID(const T& track, int nPDG)
  {
    int statusTPC = -1; //no PID info
    int statusTOF = -1; //no PID info

    if (validTPCPID(track)) {
      if (!selectionPIDTPC(track, nPDG, d_nSigmaTPC)) {
        statusTPC = 0; //rejected by PID
      } else {
        statusTPC = 1; //positive PID
      }
    }

    if (validTOFPID(track)) {
      if (!selectionPIDTOF(track, nPDG, d_nSigmaTOF)) {
        statusTOF = 0; //rejected by PID
      } else {
        statusTOF = 1; //positive PID
      }
    }

    //conservative PID strategy
    if (statusTPC == 1 && statusTOF != 0) {
      return 1;
    } else if (statusTPC == 0 || statusTOF == 0) {
      return 0;
    } else {
      return -1;
    }
  }

  void process(aod::HfCandProng3 const& hfCandProng3s, aod::BigTracksPID const&)
  {
    for (auto& hfCandProng3 : hfCandProng3s) { //looping over 3 prong candidates

      auto statusDplusToPiKPi = 0; // final selection flag : 0-rejected  1-accepted

      if (!(hfCandProng3.hfflag() & 1 << DecayType::DPlusToPiKPi)) {
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        continue;
      }

      auto trackPos1 = hfCandProng3.index0_as<aod::BigTracksPID>(); //positive daughter (negative for the antiparticles)
      auto trackNeg1 = hfCandProng3.index1_as<aod::BigTracksPID>(); //negative daughter (positive for the antiparticles)
      auto trackPos2 = hfCandProng3.index2_as<aod::BigTracksPID>(); //positive daughter (negative for the antiparticles)

      // daughter track validity selection
      if (!daughterSelection(trackPos1) || !daughterSelection(trackNeg1) || !daughterSelection(trackPos2)) {
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        continue;
      }

      // topological selection
      if (!selection(hfCandProng3, trackPos1, trackNeg1, trackPos2)) {
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        continue;
      }

      // pid selection
      auto pionPlus1 = selectionPID(trackPos1, kPiPlus);
      auto kaonMinus = selectionPID(trackNeg1, kKPlus);
      auto pionPlus2 = selectionPID(trackPos2, kPiPlus);

      if (pionPlus1 == 0 || kaonMinus == 0 || pionPlus2 == 0) { //exclude Dplus for PID
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        continue;
      }

      statusDplusToPiKPi = 1;
      hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFDplusToPiKPiCandidateSelector>(cfgc, TaskName{"hf-dplus-topikpi-candidate-selector"})};
}
