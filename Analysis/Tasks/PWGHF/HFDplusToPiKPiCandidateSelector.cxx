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
/// \brief D± → π± K∓ π± selection task
///
/// \author Fabio Catalano <fabio.catalano@cern.ch>, Politecnico and INFN Torino
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "AnalysisCore/TrackSelectorPID.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;
using namespace o2::analysis::hf_cuts_dplus_topikpi;

/// Struct for applying Dplus to piKpi selection cuts
struct HFDplusToPiKPiCandidateSelector {
  Produces<aod::HFSelDplusToPiKPiCandidate> hfSelDplusToPiKPiCandidate;

  Configurable<double> d_pTCandMin{"d_pTCandMin", 1., "Lower bound of candidate pT"};
  Configurable<double> d_pTCandMax{"d_pTCandMax", 36., "Upper bound of candidate pT"};
  // TPC
  Configurable<bool> b_requireTPC{"b_requireTPC", true, "Flag to require a positive Number of found clusters in TPC"};
  Configurable<double> d_pidTPCMinpT{"d_pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> d_pidTPCMaxpT{"d_pidTPCMaxpT", 20., "Upper bound of track pT for TPC PID"};
  Configurable<double> d_nSigmaTPC{"d_nSigmaTPC", 3., "Nsigma cut on TPC"};
  //Configurable<double> d_TPCNClsFindablePIDCut{"d_TPCNClsFindablePIDCut", 50., "Lower bound of TPC findable clusters for good PID"};
  // TOF
  Configurable<double> d_pidTOFMinpT{"d_pidTOFMinpT", 0.15, "Lower bound of track pT for TOF PID"};
  Configurable<double> d_pidTOFMaxpT{"d_pidTOFMaxpT", 20., "Upper bound of track pT for TOF PID"};
  Configurable<double> d_nSigmaTOF{"d_nSigmaTOF", 3., "Nsigma cut on TOF"};
  // topological cuts
  Configurable<std::vector<double>> pTBins{"pTBins", std::vector<double>{hf_cuts_dplus_topikpi::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"DPlus_to_Pi_K_Pi_cuts", {hf_cuts_dplus_topikpi::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "Dplus candidate selection per pT bin"};

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

  /// Candidate selections
  /// \param candidate is candidate
  /// \param trackPion1 is the first track with the pion hypothesis
  /// \param trackKaon is the track with the kaon hypothesis
  /// \param trackPion2 is the second track with the pion hypothesis
  /// \return true if candidate passes all cuts
  template <typename T1, typename T2>
  bool selection(const T1& candidate, const T2& trackPion1, const T2& trackKaon, const T2& trackPion2)
  {
    auto candpT = candidate.pt();
    int pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      return false;
    }
    // check that the candidate pT is within the analysis range
    if (candpT < d_pTCandMin || candpT > d_pTCandMax) {
      return false;
    }
    // cut on daughter pT
    if (trackPion1.pt() < cuts->get(pTBin, "pT Pi") || trackKaon.pt() < cuts->get(pTBin, "pT K") || trackPion2.pt() < cuts->get(pTBin, "pT Pi")) {
      return false;
    }
    // invariant-mass cut
    if (std::abs(InvMassDPlus(candidate) - RecoDecay::getMassPDG(pdg::Code::kDPlus)) > cuts->get(pTBin, "deltaM")) {
      return false;
    }
    if (candidate.decayLength() < cuts->get(pTBin, "decay length")) {
      return false;
    }
    if (candidate.decayLengthXYNormalised() < cuts->get(pTBin, "normalized decay length XY")) {
      return false;
    }
    if (candidate.cpa() < cuts->get(pTBin, "cos pointing angle")) {
      return false;
    }
    if (candidate.cpaXY() < cuts->get(pTBin, "cos pointing angle XY")) {
      return false;
    }
    if (std::abs(candidate.maxNormalisedDeltaIP()) > cuts->get(pTBin, "max normalized deltaIP")) {
      return false;
    }
    return true;
  }

  void process(aod::HfCandProng3 const& candidates, aod::BigTracksPID const&)
  {
    TrackSelectorPID selectorPion(kPiPlus);
    selectorPion.setRangePtTPC(d_pidTPCMinpT, d_pidTPCMaxpT);
    selectorPion.setRangeNSigmaTPC(-d_nSigmaTPC, d_nSigmaTPC);
    selectorPion.setRangePtTOF(d_pidTOFMinpT, d_pidTOFMaxpT);
    selectorPion.setRangeNSigmaTOF(-d_nSigmaTOF, d_nSigmaTOF);

    TrackSelectorPID selectorKaon(selectorPion);
    selectorKaon.setPDG(kKPlus);

    // looping over 3-prong candidates
    for (auto& candidate : candidates) {

      // final selection flag: 0 - rejected, 1 - accepted
      auto statusDplusToPiKPi = 0;

      if (!(candidate.hfflag() & 1 << DecayType::DPlusToPiKPi)) {
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        continue;
      }

      auto trackPos1 = candidate.index0_as<aod::BigTracksPID>(); // positive daughter (negative for the antiparticles)
      auto trackNeg = candidate.index1_as<aod::BigTracksPID>();  // negative daughter (positive for the antiparticles)
      auto trackPos2 = candidate.index2_as<aod::BigTracksPID>(); // positive daughter (negative for the antiparticles)

      /*
      // daughter track validity selection
      if (!daughterSelection(trackPos1) ||
          !daughterSelection(trackNeg) ||
          !daughterSelection(trackPos2)) {
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        continue;
      }
      */

      // topological selection
      if (!selection(candidate, trackPos1, trackNeg, trackPos2)) {
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        continue;
      }

      // track-level PID selection
      int pidTrackPos1Pion = selectorPion.getStatusTrackPIDAll(trackPos1);
      int pidTrackNegKaon = selectorKaon.getStatusTrackPIDAll(trackNeg);
      int pidTrackPos2Pion = selectorPion.getStatusTrackPIDAll(trackPos2);

      if (pidTrackPos1Pion == TrackSelectorPID::Status::PIDRejected ||
          pidTrackNegKaon == TrackSelectorPID::Status::PIDRejected ||
          pidTrackPos2Pion == TrackSelectorPID::Status::PIDRejected) { // exclude D±
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
