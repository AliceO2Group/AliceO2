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
#include "Analysis/HFSecondaryVertex.h"
#include "Analysis/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::framework::expressions;

/// D0 analysis task
struct TaskD0 {
  OutputObj<TH1F> hmass{TH1F("hmass", "2-prong candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", 500, 0., 5.)};
  OutputObj<TH1F> hptcand{TH1F("hptcand", "2-prong candidates;candidate #it{p}_{T} (GeV/#it{c});entries", 100, 0., 10.)};
  OutputObj<TH1F> hptprong0{TH1F("hptprong0", "2-prong candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", 100, 0., 10.)};
  OutputObj<TH1F> hptprong1{TH1F("hptprong1", "2-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", 100, 0., 10.)};
  OutputObj<TH1F> hdeclength{TH1F("declength", "2-prong candidates;decay length (cm);entries", 200, 0., 2.)};
  OutputObj<TH1F> hdeclengthxy{TH1F("declengthxy", "2-prong candidates;decay length xy (cm);entries", 200, 0., 2.)};
  OutputObj<TH1F> hd0Prong0{TH1F("hd0Prong0", "2-prong candidates;prong 0 DCAxy to prim. vertex (cm);entries", 100, -1., 1.)};
  OutputObj<TH1F> hd0Prong1{TH1F("hd0Prong1", "2-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", 100, -1., 1.)};
  OutputObj<TH1F> hd0d0{TH1F("hd0d0", "2-prong candidates;product of DCAxy to prim. vertex (cm^{2});entries", 500, -1., 1.)};
  OutputObj<TH1F> hCTS{TH1F("hCTS", "2-prong candidates;cos #it{#theta}* (D^{0});entries", 110, -1.1, 1.1)};
  OutputObj<TH1F> hCt{TH1F("hCt", "2-prong candidates;proper lifetime (D^{0}) * #it{c} (cm);entries", 120, -20., 100.)};
  OutputObj<TH1F> hCPA{TH1F("hCPA", "2-prong candidates;cosine of pointing angle;entries", 110, -1.1, 1.1)};
  OutputObj<TH1F> hEta{TH1F("hEta", "2-prong candidates;candidate #it{#eta};entries", 100, -2., 2.)};
  //OutputObj<TH1F> hselectionstatus{TH1F("selectionstatus", "2-prong candidates;selection status;entries", 5, -0.5, 4.5)};
  OutputObj<TH1F> hImpParErr{TH1F("hImpParErr", "2-prong candidates;impact parameter error (cm);entries", 100, -1., 1.)};
  OutputObj<TH1F> hDecLenErr{TH1F("hDecLenErr", "2-prong candidates;decay length error (cm);entries", 100, 0., 1.)};
  OutputObj<TH1F> hDecLenXYErr{TH1F("hDecLenXYErr", "2-prong candidates;decay length xy error (cm);entries", 100, 0., 1.)};

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};

  Filter filterSelectCandidates = (aod::hf_selcandidate::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate::isSelD0bar >= d_selectionFlagD0bar);

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
      hd0Prong0->Fill(candidate.impactParameter0());
      hd0Prong1->Fill(candidate.impactParameter1());
      hd0d0->Fill(candidate.impactParameterProduct());
      hCTS->Fill(CosThetaStarD0(candidate));
      hCt->Fill(CtD0(candidate));
      hCPA->Fill(candidate.cpa());
      hEta->Fill(candidate.eta());
      //hselectionstatus->Fill(candidate.isSelD0() + (candidate.isSelD0bar() * 2));
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
