// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskdzero.h
/// \brief D0 analysis task
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Analysis/SecondaryVertexHF.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using std::array;

/// D0 analysis task
struct TaskDzero {
  OutputObj<TH1F> hmass{TH1F("hmass", "2-track inv mass", 500, 0, 5.0)};
  OutputObj<TH1F> hptcand{TH1F("hptcand", "pt candidate", 100, 0, 10.0)};
  OutputObj<TH1F> hptprong0{TH1F("hptprong0", "pt prong0", 100, 0, 10.0)};
  OutputObj<TH1F> hptprong1{TH1F("hptprong1", "pt prong1", 100, 0, 10.0)};
  OutputObj<TH1F> hdeclength{TH1F("declength", "decay length", 100, 0., 1.0)};
  OutputObj<TH1F> hdeclengthxy{TH1F("declengthxy", "decay length xy", 100, 0., 1.0)};
  OutputObj<TH1F> hd0{TH1F("hd0", "dca xy to prim. vertex (cm)", 100, -1.0, 1.0)};
  OutputObj<TH1F> hd0d0{TH1F("hd0d0", "product of dca xy to prim. vertex (cm^{2})", 100, -1.0, 1.0)};
  OutputObj<TH1F> hCTS{TH1F("hCTS", "cos #it{#theta}*", 120, -1.1, 1.1)};
  OutputObj<TH1F> hCt{TH1F("hCt", "proper lifetime * #it{c} (cm)", 120, -20, 100)};
  OutputObj<TH1F> hEta{TH1F("hEta", "#it{#eta}", 100, -2, 2)};

  void process(aod::HfCandProng2 const& candidates)
  {
    for (auto& candidate : candidates) {
      hmass->Fill(InvMassD0(candidate));
      hmass->Fill(InvMassD0bar(candidate));
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
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TaskDzero>("hf-taskdzero")};
}
