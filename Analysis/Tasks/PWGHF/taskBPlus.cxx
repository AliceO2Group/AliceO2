// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskBplus.cxx
/// \brief B+ analysis task
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

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

/// B+ analysis task
struct TaskBplus {
  HistogramRegistry registry{
    "registry",
    {{"hmassD0", "2-prong candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hptcand", "B+ candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}}}};

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= d_selectionFlagD0bar);

  void process(aod::Collisions const& collisions, aod::BigTracks const& tracks, soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>> const& candidates)
  {
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      if (cutEtaCandMax >= 0. && std::abs(candidate.eta()) > cutEtaCandMax) {
        //Printf("Candidate: eta rejection: %g", candidate.eta());
        continue;
      }
      if (candidate.isSelD0bar() >= d_selectionFlagD0bar) {
        //Printf("Track size: %d", tracks.size());
        registry.fill(HIST("hmassD0"), InvMassD0bar(candidate));
        for (auto trackPos1 = tracks.begin(); trackPos1 != tracks.end(); ++trackPos1) {
          if (trackPos1.signed1Pt() < 0) {
            continue;
          }
          if (candidate.index0().collisionId() != trackPos1.collisionId()) {
            continue;
          }
          //Printf("Positive track pt: %f", trackPos1.pt());
          registry.fill(HIST("hptcand"), candidate.pt() + trackPos1.pt());
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskBplus>("hf-task-bplus")};
  return workflow;
}
