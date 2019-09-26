// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/HistogramRegistry.h"
#include <TH1F.h>

using namespace o2;
using namespace o2::framework;

// This is a very simple example showing how to create an histogram
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  HistogramRegistry registry{
    "MyAnalysisHistos",
    true,
    {{"Test1", "SomethingElse1", {"TH1F", 100, 0, 2 * M_PI}},
     {"Test2", "SomethingElse2", {"TH1F", 100, 0, 2 * M_PI}},
     {"Test3", "SomethingElse3", {"TH1F", 100, 0, 2 * M_PI}},
     {"Test4", "SomethingElse4", {"TH1F", 100, 0, 2 * M_PI}},
     {"Test5", "SomethingElse5", {"TH1F", 100, 0, 2 * M_PI}},
     {"Test6", "SomethingElse6", {"TH1F", 100, 0, 2 * M_PI}},
     {"Test7", "SomethingElse7", {"TH1F", 100, 0, 2 * M_PI}}}};

  OutputObj<TH1F> myHisto{TH1F("something", "somethingElse", 100, 0., 1.)};

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      auto phi = asin(track.snp()) + track.alpha() + M_PI;
      myHisto->Fill(phi);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("fill-etaphi-histogram")};
}
