// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
///
/// \brief Both tasks, ATask and BTask create two histograms. But whereas in
///        the first case (ATask) the histograms are not saved to file, this
///        happens automatically if OutputObj<TH1F> is used to create a
///        histogram. By default the histogram is saved to file
///        AnalysisResults.root. HistogramRegistry is yet an other possibility
///        to deal with histograms. See tutorial example histogramRegistery.cxx
///        for details.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
using namespace o2::framework::expressions;
using namespace o2;
using namespace o2::framework;

struct ATask {

  // normal creation of a histogram
  TH1F* phiHA = new TH1F("phiA", "phiA", 100, 0., 2. * M_PI);
  TH1F* etaHA = new TH1F("etaA", "etaA", 102, -2.01, 2.01);

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      phiHA->Fill(track.phi());
      etaHA->Fill(track.eta());
    }
  }
};

struct BTask {

  // histogram created with OutputObj<TH1F>
  OutputObj<TH1F> phiB{TH1F("phiB", "phiB", 100, 0., 2. * M_PI), OutputObjHandlingPolicy::QAObject};
  OutputObj<TH2F> etaptB{TH2F("etaptB", "etaptB", 102, -2.01, 2.01, 100, 0.0, 5.0), OutputObjHandlingPolicy::AnalysisObject};

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      phiB->Fill(track.phi());
      etaptB->Fill(track.eta(), track.pt());
    }
  }
};

struct CTask {
  // incomplete definition of an OutputObj
  OutputObj<TH1F> trZ{"trZ", OutputObjHandlingPolicy::QAObject};

  Filter ptfilter = aod::track::pt > 0.5;

  void init(InitContext const&)
  {
    // complete the definition of the OutputObj
    trZ.setObject(new TH1F("Z", "Z", 100, -10., 10.));
    // other options:
    // TH1F* t = new TH1F(); trZ.setObject(t); <- resets content!
    // TH1F t(); trZ.setObject(t) <- makes a copy
    // trZ.setObject({"Z","Z",100,-10.,10.}); <- creates new
  }

  void process(soa::Filtered<aod::Tracks> const& tracks)
  {
    for (auto& track : tracks) {
      trZ->Fill(track.z());
    }
  }
};

struct DTask {

  // histogram defined with HistogramRegistry
  HistogramRegistry registry{
    "registry",
    {{"phiC", "phiC", {HistType::kTH1F, {{100, 0., 2. * M_PI}}}},
     {"etaptC", "etaptC", {HistType::kTH2F, {{102, -2.01, 2.01}, {100, 0.0, 5.0}}}}}};

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      registry.get<TH1>(HIST("phiC"))->Fill(track.phi());
      registry.get<TH2>(HIST("etaptC"))->Fill(track.eta(), track.pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>(cfgc, TaskName{TaskName{"histograms-tutorial_A"}}),
    adaptAnalysisTask<BTask>(cfgc, TaskName{TaskName{"histograms-tutorial_B"}}),
    adaptAnalysisTask<CTask>(cfgc, TaskName{TaskName{"histograms-tutorial_C"}}),
    adaptAnalysisTask<DTask>(cfgc, TaskName{TaskName{"histograms-tutorial_D"}}),
  };
}
