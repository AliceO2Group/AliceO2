// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief Example of histogram handling with OutputObj.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include <TH1F.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct DummyTask {
  OutputObj<TH1F> pt1{TH1F("pt1________0123456789", "pt task1", 100, 0., 50.)};
  OutputObj<TH1F> pt2{TH1F("pt2________0123456789", "pt task1", 100, 0., 50.)};
  OutputObj<TH1F> pt3{TH1F("pt3________0123456789", "pt task1", 100, 0., 50.)};

  void process(aod::Track const& track)
  {
    auto pt = track.pt();
    pt1->Fill(pt);
    pt2->Fill(pt);
    pt3->Fill(pt);
  }
};

struct DummyTask2 {
  OutputObj<TH1F> pt1{TH1F("pt1________0123456789", "pt task2", 100, 0., 50.)};
  OutputObj<TH1F> pt2{TH1F("pt2________0123456789", "pt task2", 100, 0., 50.)};
  OutputObj<TH1F> pt3{TH1F("pt3________0123456789", "pt task2", 100, 0., 50.)};

  void process(aod::Track const& track)
  {
    auto pt = track.pt();
    pt1->Fill(pt);
    pt2->Fill(pt);
    pt3->Fill(pt);
  }
};

struct DummyTask3 {
  OutputObj<TH1F> pt1{TH1F("pt1________0123456789", "pt task3", 100, 0., 50.)};
  OutputObj<TH1F> pt2{TH1F("pt2________0123456789", "pt task3", 100, 0., 50.)};
  OutputObj<TH1F> pt3{TH1F("pt3________0123456789", "pt task3", 100, 0., 50.)};
  void process(aod::Track const& track)
  {
    auto pt = track.pt();
    pt1->Fill(pt);
    pt2->Fill(pt);
    pt3->Fill(pt);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<DummyTask>(cfgc),
    adaptAnalysisTask<DummyTask2>(cfgc),
    adaptAnalysisTask<DummyTask3>(cfgc),
  };
}
