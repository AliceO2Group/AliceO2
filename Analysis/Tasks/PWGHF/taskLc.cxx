// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskDPlus.cxx
/// \brief D± analysis task
/// \note Extended from taskD0
///
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;
using namespace o2::framework::expressions;

/// Lc+ analysis task
struct TaskLc {
  OutputObj<TH1F> hmass{TH1F("hmass", "3-prong candidates;inv. mass (#pi K #pi) (GeV/#it{c}^{2});entries", 500, 1.6, 3.1)};
  OutputObj<TH1F> hptcand{TH1F("hptcand", "3-prong candidates;candidate #it{p}_{T} (GeV/#it{c});entries", 100, 0., 10.)};
  OutputObj<TH1F> hptprong0{TH1F("hptprong0", "3-prong candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", 100, 0., 10.)};
  OutputObj<TH1F> hptprong1{TH1F("hptprong1", "3-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", 100, 0., 10.)};
  OutputObj<TH1F> hptprong2{TH1F("hptprong2", "3-prong candidates;prong 2 #it{p}_{T} (GeV/#it{c});entries", 100, 0., 10.)};
  OutputObj<TH1F> hdeclength{TH1F("declength", "3-prong candidates;decay length (cm);entries", 200, 0., 2.)};
  OutputObj<TH1F> hd0Prong0{TH1F("hd0Prong0", "3-prong candidates;prong 0 DCAxy to prim. vertex (cm);entries", 100, -1., 1.)};
  OutputObj<TH1F> hd0Prong1{TH1F("hd0Prong1", "3-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", 100, -1., 1.)};
  OutputObj<TH1F> hd0Prong2{TH1F("hd0Prong2", "3-prong candidates;prong 2 DCAxy to prim. vertex (cm);entries", 100, -1., 1.)};
  OutputObj<TH1F> hCt{TH1F("hCt", "3-prong candidates;proper lifetime (D^{#pm}) * #it{c} (cm);entries", 120, -20., 100.)};
  OutputObj<TH1F> hCPA{TH1F("hCPA", "3-prong candidates;cosine of pointing angle;entries", 110, -1.1, 1.1)};
  OutputObj<TH1F> hEta{TH1F("hEta", "3-prong candidates;candidate #it{#eta};entries", 100, -2., 2.)};
  //OutputObj<TH1F> hselectionstatus{TH1F("selectionstatus", "3-prong candidates;selection status;entries", 5, -0.5, 4.5)};
  OutputObj<TH1F> hImpParErr{TH1F("hImpParErr", "3-prong candidates;impact parameter error (cm);entries", 100, -1., 1.)};
  OutputObj<TH1F> hDecLenErr{TH1F("hDecLenErr", "3-prong candidates;decay length error (cm);entries", 100, 0., 1.)};
  //OutputObj<TH1F> hdca{TH1F("hdca", "3-prong candidates;prongs DCA to prim. vertex (cm);entries", 100, -1., 1.)};
  OutputObj<TH1F> hdca2{TH1F("hdca2", "3-prong candidates;prongs DCA to sec. vertex (cm);entries", 100, 0., 1.)};

  Configurable<int> d_selectionFlagLc{"d_selectionFlagLc", 1, "Selection Flag for Lc"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_lc::isSelLc >= d_selectionFlagLc);

  //void process(aod::HfCandProng3 const& candidates)
  void process(soa::Filtered<soa::Join<aod::HfCandProng3, aod::HFSelLcCandidate>> const& candidates)
  {
    for (auto& candidate : candidates) {
      if (candidate.isSelLc() >= d_selectionFlagLc) {
        hmass->Fill(InvMassLc(candidate));
      }
      hptcand->Fill(candidate.pt());
      hptprong0->Fill(candidate.ptProng0());
      hptprong1->Fill(candidate.ptProng1());
      hptprong2->Fill(candidate.ptProng2());
      hdeclength->Fill(candidate.decayLength());
      hd0Prong0->Fill(candidate.impactParameter0());
      hd0Prong1->Fill(candidate.impactParameter1());
      hd0Prong2->Fill(candidate.impactParameter2());
      hCt->Fill(CtLc(candidate));
      hCPA->Fill(candidate.cpa());
      hEta->Fill(candidate.eta());
      //hselectionstatus->Fill(candidate.isSelLc());
      hImpParErr->Fill(candidate.errorImpactParameter0());
      hImpParErr->Fill(candidate.errorImpactParameter1());
      hImpParErr->Fill(candidate.errorImpactParameter2());
      hDecLenErr->Fill(candidate.errorDecayLength());
      // hdca->Fill(candidate.dca());
      hdca2->Fill(candidate.chi2PCA());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TaskLc>("hf-task-lc")};
}
