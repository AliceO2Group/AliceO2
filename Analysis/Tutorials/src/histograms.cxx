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
#include <TH1F.h>

#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

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
  // Create a configurable which can be used inside the process method.
  Configurable<float> phiCut{"phiCut", 6.29f, "A cut on phi"};

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      // FIXME: this is until we have configurables which
      //        can be used in expressions.
      if (track.phi() < phiCut) {
        etaphiH->Fill(track.phi(), track.eta());
      }
    }
  }
};

struct CTask {
  // needs to be initialized with a label or an obj
  // when adding an object to OutputObj later, the object name will be
  // *reset* to OutputObj label - needed for correct placement in the output file
  OutputObj<TH1F> ptH{TH1F("pt", "pt", 100, -0.01, 10.01)};
  OutputObj<TH1F> trZ{"trZ", OutputObjHandlingPolicy::QAObject};
  Configurable<float> pTCut{"pTCut", 0.5f, "Lower pT limit"};

  Filter ptfilter = aod::track::pt > pTCut;

  void init(InitContext const&)
  {
    trZ.setObject(new TH1F("Z", "Z", 100, -10., 10.));
    // other options:
    // TH1F* t = new TH1F(); trZ.setObject(t); <- resets content!
    // TH1F t(); trZ.setObject(t) <- makes a copy
    // trZ.setObject({"Z","Z",100,-10.,10.}); <- creates new
  }

  void process(soa::Filtered<aod::Tracks> const& tracks)
  {
    for (auto& track : tracks) {
      ptH->Fill(track.pt());
      trZ->Fill(track.z());
    }
  }
};

struct DTask {
  OutputObj<TList> list{"list"};

  void init(InitContext const&)
  {
    list.setObject(new TList);
    list->Add(new TH1F("pHist", "", 100, 0, 10));
    list->Add(new TH1F("etaHist", "", 102, -2.01, 2.01));
  }

  void process(aod::Track const& track)
  {
    auto pHist = dynamic_cast<TH1F*>(list->At(0));
    auto etaHist = dynamic_cast<TH1F*>(list->At(1));

    pHist->Fill(track.p());
    etaHist->Fill(track.eta());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>(cfgc, "eta-and-phi-histograms"),
    adaptAnalysisTask<BTask>(cfgc, "etaphi-histogram"),
    adaptAnalysisTask<CTask>(cfgc, "pt-histogram"),
    adaptAnalysisTask<DTask>(cfgc, "output-wrapper"),
  };
}
