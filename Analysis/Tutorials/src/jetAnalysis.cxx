// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// jet analysis tasks (subscribing to jet finder task)
//
// Author: Jochen Klein
//
// o2-analysis-jetfinder --aod-file <aod> -b | o2-analysistutorial-jet-analysis -b

#include "TH1F.h"

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"

#include "AnalysisDataModel/Jet.h"

using namespace o2;
using namespace o2::framework;

struct JetAnalysis {
  OutputObj<TH1F> hJetPt{"jetPt"};
  OutputObj<TH1F> hConstPt{"constPt"};

  void init(InitContext const&)
  {
    hJetPt.setObject(new TH1F("jetPt", "jet p_{T};p_{T} (GeV/#it{c})",
                              100, 0., 100.));
    hConstPt.setObject(new TH1F("constPt", "constituent p_{T};p_{T} (GeV/#it{c})",
                                100, 0., 100.));
  }

  void process(aod::Jet const& jet,
               aod::JetConstituents const& constituents, aod::Tracks const& tracks)
  {
    hJetPt->Fill(jet.pt());
    for (const auto c : constituents) {
      LOGF(DEBUG, "jet %d: track id %d, track pt %g", jet.index(), c.trackId(), c.track().pt());
      hConstPt->Fill(c.track().pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetAnalysis>(cfgc, TaskName{"jet-analysis"})};
}
