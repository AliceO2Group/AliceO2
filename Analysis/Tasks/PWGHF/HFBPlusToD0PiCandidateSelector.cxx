// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFBPlusToD0PiCandidateSelector.cxx
/// \brief B± → D0bar(D0) π± candidate selector
///
/// \author Antonio Palasciano <antonio.palasciano@cern.ch>, Università degli Studi di Bari & INFN, Sezione di Bari
/// \author Deepa Thomas <deepa.thomas@cern.ch>, UT Austin

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisCore/HFSelectorCuts.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "AnalysisCore/HFSelectorCuts.h"

using namespace o2;
using namespace o2::aod;
using namespace o2::framework;
using namespace o2::aod::hf_cand_bplus;
using namespace o2::analysis;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::analysis::hf_cuts_bplus_tod0pi;

struct HfBplusTod0piCandidateSelector {

  Produces<aod::HFSelBPlusToD0PiCandidate> hfSelBPlusToD0PiCandidate;

  Configurable<double> pTCandMin{"pTCandMin", 0., "Lower bound of candidate pT"};
  Configurable<double> pTCandMax{"pTCandMax", 50., "Upper bound of candidate pT"};

  Configurable<double> pidTPCMinpT{"pidTPCMinpT", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> pidTPCMaxpT{"pidTPCMaxpT", 10., "Upper bound of track pT for TPC PID"};
  Configurable<double> pidTOFMinpT{"pidTOFMinpT", 0.15, "Lower bound of track pT for TOF PID"};
  Configurable<double> pidTOFMaxpT{"pidTOFMaxpT", 10., "Upper bound of track pT for TOF PID"};

  Configurable<double> TPCNClsFindablePIDCut{"TPCNClsFindablePIDCut", 70., "Lower bound of TPC findable clusters for good PID"};
  Configurable<double> nSigmaTPC{"nSigmaTPC", 5., "Nsigma cut on TPC only"};
  Configurable<double> nSigmaTPCCombined{"nSigmaTPCCombined", 5., "Nsigma cut on TPC combined with TOF"};
  Configurable<double> nSigmaTOF{"nSigmaTOF", 5., "Nsigma cut on TOF only"};
  Configurable<double> nSigmaTOFCombined{"nSigmaTOFCombined", 5., "Nsigma cut on TOF combined with TPC"};

  Configurable<std::vector<double>> pTBins{"pTBins", std::vector<double>{hf_cuts_bplus_tod0pi::pTBins_v}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"BPlus_to_d0pi_cuts", {hf_cuts_bplus_tod0pi::cuts[0], npTBins, nCutVars, pTBinLabels, cutVarLabels}, "B+ candidate selection per pT bin"};
  Configurable<int> selectionFlagD0{"selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> selectionFlagD0bar{"selectionFlagD0bar", 1, "Selection Flag for D0bar"};

  // Apply topological cuts as defined in HFSelectorCuts.h; return true if candidate passes all cuts
  template <typename T1, typename T2, typename T3>
  bool selectionTopol(const T1& hfCandBPlus, const T2& hfCandD0, const T3& trackPi)
  {
    auto candpT = hfCandBPlus.pt();
    int pTBin = findBin(pTBins, candpT);
    if (pTBin == -1) {
      // Printf("B+ topol selection failed at getpTBin");
      return false;
    }

    //pi pt
    if (trackPi.pt() < cuts->get(pTBin, "pT Pi")) {
      return false;
    }

    //d0(D0)xd0(pi)
    if (hfCandBPlus.impactParameterProduct() > cuts->get(pTBin, "Imp. Par. Product")) {
      return false;
    }

    //D0 mass
    if (trackPi.sign() > 0) {
      if (std::abs(InvMassD0bar(hfCandD0) - RecoDecay::getMassPDG(pdg::Code::kD0)) > cuts->get(pTBin, "DeltaMD0")) {
        return false;
      }
    }
    if (trackPi.sign() < 0) {
      if (std::abs(InvMassD0(hfCandD0) - RecoDecay::getMassPDG(pdg::Code::kD0)) > cuts->get(pTBin, "DeltaMD0")) {
        return false;
      }
    }

    //B Decay length
    if (hfCandBPlus.decayLength() < cuts->get(pTBin, "B decLen")) {
      return false;
    }

    //B Decay length XY
    if (hfCandBPlus.decayLengthXY() < cuts->get(pTBin, "B decLenXY")) {
      return false;
    }

    //B+ CPA cut
    if (hfCandBPlus.cpa() < cuts->get(pTBin, "CPA")) {
      return false;
    }

    //if (candpT < pTCandMin || candpT >= pTCandMax) {
    // Printf("B+ topol selection failed at cand pT check");
    // return false;
    // }

    //B+ mass cut
    //if (std::abs(InvMassBPlus(hfCandBPlus) - RecoDecay::getMassPDG(521)) > cuts->get(pTBin, "m")) {
    // Printf("B+ topol selection failed at mass diff check");
    //  return false;
    // }

    //d0 of D0 and pi
    //if ((std::abs(hfCandBPlus.impactParameter0()) > cuts->get(pTBin, "d0 D0")) ||
    //    (std::abs(hfCandBPlus.impactParameter1()) > cuts->get(pTBin, "d0 Pi"))){
    //  return false;
    //}
    //D0 CPA
    // if (std::abs(hfCandD0.cpa()) < cuts->get(pTBin, "CPA D0")){
    //  return false;
    //}
    return true;
  }

  void process(aod::HfCandBPlus const& hfCandBs, soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>, aod::BigTracksPID const& tracks)
  {
    for (auto& hfCandB : hfCandBs) { //looping over Bplus candidates

      int statusBplus = 0;

      // check if flagged as B+ --> D0bar Pi
      if (!(hfCandB.hfflag() & 1 << hf_cand_bplus::DecayType::BPlusToD0Pi)) {
        hfSelBPlusToD0PiCandidate(statusBplus);
        // Printf("B+ candidate selection failed at hfflag check");
        continue;
      }

      // D0 is always index0 and pi is index1 by default
      auto candD0 = hfCandB.index0_as<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>>();
      auto trackPi = hfCandB.index1_as<aod::BigTracksPID>();

      //topological cuts
      if (!selectionTopol(hfCandB, candD0, trackPi)) {
        hfSelBPlusToD0PiCandidate(statusBplus);
        // Printf("B+ candidate selection failed at selection topology");
        continue;
      }

      hfSelBPlusToD0PiCandidate(1);
      // Printf("B+ candidate selection successful, candidate should be selected");
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{};
  workflow.push_back(adaptAnalysisTask<HfBplusTod0piCandidateSelector>(cfgc));
  return workflow;
}
