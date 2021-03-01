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
/// \author Rik Spijkers <r.spijkers@students.uu.nl>

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

namespace o2::aod
{
namespace extra
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
}
DECLARE_SOA_TABLE(Colls, "AOD", "COLLSID", o2::aod::extra::CollisionId);
} // namespace o2::aod
struct AddCollisionId {
  Produces<o2::aod::Colls> colls;
  void process(aod::HfCandProng2 const& candidates, aod::Tracks const&)
  {
    for (auto& candidate : candidates) {
      colls(candidate.index0_as<aod::Tracks>().collisionId());
    }
  }
};

/// X(3872) analysis task
/// FIXME: Still need to remove track duplication!!!
struct TaskX {
  HistogramRegistry registry{
    "registry",
    {{"hMassJpsi", "2-prong candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hPtCand", "X candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}}}};

  Configurable<int> selectionFlagJpsi{"selectionFlagJpsi", 1, "Selection Flag for Jpsi"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_jpsi::isSelJpsiToEE >= d_selectionFlagJpsi);

  /// aod::BigTracks is not soa::Filtered, should be added when filters are added
  void process(aod::Collision const&, aod::BigTracks const& tracks, soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelJpsiToEECandidate, aod::Colls>> const& candidates)
  {
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << JpsiToEE)) {
        continue;
      }
      if (cutEtaCandMax >= 0. && std::abs(candidate.eta()) > cutEtaCandMax) {
        continue;
      }
      registry.fill(HIST("hmassJpsi"), InvMassJpsiToEE(candidate));

      int index0jpsi = candidate.index0Id();
      int index1jpsi = candidate.index1Id();
      auto JpsiTrackPos = candidate.index0();
      auto JpsiTrackNeg = candidate.index1();
      for (auto trackPos = tracks.begin(); trackPos != tracks.end(); ++trackPos) {
        if (trackPos.signed1Pt() < 0) {
          continue;
        }
        if (trackPos.globalIndex() == index0jpsi){
          printf("pos track id check: %ld, %u\n", trackPos.globalIndex(), index0jpsi);
          printf("pos track pt check: %f, %f\n", JpsiTrackPos.pt(), trackPos.pt());
          continue;
        }
        for (auto trackNeg = tracks.begin(); trackNeg != tracks.end(); ++trackNeg) {
          if (trackNeg.signed1Pt() > 0) {
            continue;
          }
          if (trackNeg.globalIndex() == index1jpsi){
            printf("neg track id check: %ld, %u\n", trackNeg.globalIndex(), index1jpsi);
            printf("neg track pt check: %f, %f\n", JpsiTrackNeg.pt(), trackNeg.pt());
            continue;
          }
          registry.fill(HIST("hptcand"), candidate.pt() + trackPos.pt() + trackNeg.pt());
        } // pi- loop
      } // pi+ loop
    } // Jpsi loop
  } // process
}; // struct

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<AddCollisionId>("hf-task-add-collisionId"),
    adaptAnalysisTask<TaskX>("hf-task-x")};
  return workflow;
}
