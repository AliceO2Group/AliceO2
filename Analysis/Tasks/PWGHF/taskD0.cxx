// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskD0.cxx
/// \brief D0 analysis task
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Analysis/SecondaryVertexHF.h"
#include "Analysis/CandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::framework::expressions;

/// D0 analysis task
struct TaskD0 {
  OutputObj<TH1F> hmass{TH1F("hmass", "2-track inv mass", 500, 0, 5.0)};
  OutputObj<TH1F> hptcand{TH1F("hptcand", "pt candidate", 100, 0, 10.0)};
  OutputObj<TH1F> hptprong0{TH1F("hptprong0", "pt prong0", 100, 0, 10.0)};
  OutputObj<TH1F> hptprong1{TH1F("hptprong1", "pt prong1", 100, 0, 10.0)};
  OutputObj<TH1F> hdeclength{TH1F("declength", "decay length", 200, 0., 2.0)};
  OutputObj<TH1F> hdeclengthxy{TH1F("declengthxy", "decay length xy", 200, 0., 2.0)};
  OutputObj<TH1F> hd0{TH1F("hd0", "dca xy to prim. vertex (cm)", 100, -1.0, 1.0)};
  OutputObj<TH1F> hd0d0{TH1F("hd0d0", "product of dca xy to prim. vertex (cm^{2})", 500, -1.0, 1.0)};
  OutputObj<TH1F> hCTS{TH1F("hCTS", "cos #it{#theta}*", 120, -1.1, 1.1)};
  OutputObj<TH1F> hCt{TH1F("hCt", "proper lifetime * #it{c} (cm)", 120, -20, 100)};
  OutputObj<TH1F> hEta{TH1F("hEta", "#it{#eta}", 100, -2, 2)};
  OutputObj<TH1F> hselectionstatus{TH1F("selectionstatus", "selection status", 5, -0.5, 4.5)};
  OutputObj<TH1F> hImpParErr{TH1F("hImpParErr", "impact parameter error", 100, -1.0, 1.0)};
  OutputObj<TH1F> hDecLenErr{TH1F("hDecLenErr", "decay length error", 100, 0., 1.0)};
  OutputObj<TH1F> hDecLenXYErr{TH1F("hDecLenXYErr", "decay length XY error", 100, 0., 1.0)};

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};

  Filter seltrack = (aod::hfselcandidate::isSelD0 >= d_selectionFlagD0 || aod::hfselcandidate::isSelD0bar >= d_selectionFlagD0bar);

  void process(soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>> const& candidates)
  {
    for (auto& candidate : candidates) {
      if (candidate.isSelD0() >= d_selectionFlagD0) {
        hmass->Fill(InvMassD0(candidate));
      }
      if (candidate.isSelD0bar() >= d_selectionFlagD0bar) {
        hmass->Fill(InvMassD0bar(candidate));
      }
      hptcand->Fill(candidate.pt());
      hptprong0->Fill(candidate.ptProng0());
      hptprong1->Fill(candidate.ptProng1());
      hdeclength->Fill(candidate.decayLength());
      hdeclengthxy->Fill(candidate.decayLengthXY());
      hd0->Fill(candidate.impactParameter0());
      hd0->Fill(candidate.impactParameter1());
      hd0d0->Fill(candidate.impactParameterProduct());
      hCTS->Fill(CosThetaStarD0(candidate));
      hCt->Fill(CtD0(candidate));
      hEta->Fill(candidate.eta());
      hselectionstatus->Fill(candidate.isSelD0() + (candidate.isSelD0bar() * 2));
      hImpParErr->Fill(candidate.errorImpactParameter0());
      hImpParErr->Fill(candidate.errorImpactParameter1());
      hDecLenErr->Fill(candidate.errorDecayLength());
      hDecLenXYErr->Fill(candidate.errorDecayLengthXY());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TaskD0>("hf-task-d0")};
}
