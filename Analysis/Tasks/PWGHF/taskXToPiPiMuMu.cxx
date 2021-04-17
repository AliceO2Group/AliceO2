// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskX.cxx
/// \brief X(3872) analysis task
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Rik Spijkers <r.spijkers@students.uu.nl>, Utrecht University
/// \author Antonio Uras <antonio.uras@cern.ch>, IP2I-Lyon

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Fill MC histograms."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// X(3872) analysis task
struct TaskX {
  HistogramRegistry registry{
    "registry",
    {{"hMassJpsi", "2-prong candidates;inv. mass (mu+ mu-) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hPtCand", "X candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}}}};

  Configurable<int> selectionFlagJpsiToMuMu{"selectionFlagJpsiToMuMu", 1, "Selection Flag for JpsiToMuMu"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_jpsitomumu::isSelJpsiToMuMu >= selectionFlagJpsiToMuMu);

  /// aod::BigTracks is not soa::Filtered, should be added when filters are added
  void process(aod::Collision const&, aod::BigTracks const& tracks, soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelJpsiToMuMuCandidate>> const& candidates)
  {
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << JpsiToMuMu)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YJpsi(candidate)) > cutYCandMax) {
        continue;
      }
      registry.fill(HIST("hMassJpsi"), InvMassJpsiToMuMu(candidate));

      int index0jpsi = candidate.index0Id();
      int index1jpsi = candidate.index1Id();
      for (auto& track1 : tracks) {
        int signTrack1 = track1.sign();
        int indexTrack1 = track1.globalIndex();
        if (signTrack1 > 0) {
          if (indexTrack1 == index0jpsi) {
            continue;
          }
        } else if (indexTrack1 == index1jpsi) {
          continue;
        }
        for (auto track2 = track1 + 1; track2 != tracks.end(); ++track2) {
          if (signTrack1 == track2.sign()) {
            continue;
          }
          int indexTrack2 = track2.globalIndex();
          if (signTrack1 > 0) {
            if (indexTrack2 == index1jpsi) {
              continue;
            }
          } else if (indexTrack2 == index0jpsi) {
            continue;
          }
          registry.fill(HIST("hPtCand"), candidate.pt() + track1.pt() + track2.pt());
        } // track2 loop (pion)
      }   // track1 loop (pion)
    }     // Jpsi loop
  }       // process
};        // struct

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskX>(cfgc, TaskName{"hf-task-x-topipiee"})};
  return workflow;
}
