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

#include "Analysis/Jet.h"

using namespace o2;
using namespace o2::framework;

struct JetAnalysis {
  OutputObj<TH1F> hJetPt{"pt"};

  void init(InitContext const&)
  {
    hJetPt.setObject(new TH1F("pt", "jet p_{T};p_{T} (GeV/#it{c})",
                              100, 0., 100.));
  }

  // TODO: add aod::Tracks (when available)
  void process(aod::Jet const& jet,
               aod::JetConstituents const& constituents)
  {
    hJetPt->Fill(jet.pt());
    for (const auto c : constituents) {
      LOGF(INFO, "jet %d: track id %d", jet.index(), c.trackId());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetAnalysis>("jet-analysis")};
}
