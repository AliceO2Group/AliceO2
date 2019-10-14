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

#include <cmath>

using namespace o2;
using namespace o2::framework;

// This is a very simple example showing how to create an histogram
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  OutputObj<TH1F> phiH{TH1F("phi", "phi", 100, 0., 2. * M_PI)};
  OutputObj<TH1F> etaH{TH1F("eta", "eta", 102, -2.01, 2.01)};

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      phiH->Fill(track.phi());
      etaH->Fill(track.eta());
    }
  }
};

struct BTask {
  OutputObj<TH2F> etaphiH{TH2F("etaphi", "etaphi", 100, 0., 2. * M_PI, 102, -2.01, 2.01)};

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      etaphiH->Fill(track.phi(), track.eta());
    }
  }
};

struct CTask {
  OutputObj<TH1F> ptH{TH1F("pt", "pt", 100, -0.01, 10.01)};

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      ptH->Fill(abs(track.signed1Pt()));
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("eta-and-phi-histograms"),
    adaptAnalysisTask<BTask>("etaphi-histogram"),
    adaptAnalysisTask<CTask>("pt-histogram"),
  };
}
