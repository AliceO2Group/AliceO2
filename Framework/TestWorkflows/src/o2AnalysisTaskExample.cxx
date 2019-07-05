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

#include <TFile.h>
#include <TH1F.h>

using namespace o2;
using namespace o2::framework;

// This is a stateful task, where we send the state downstream.
class ATask : public AnalysisTask
{
 public:
  explicit ATask(int state)
    : mSomeState{ state } {}

  void init(InitContext& ic)
  {
    mSomeState += 1;
  }
  void run(ProcessingContext& pc)
  {
  }

  void process(aod::Timeframe const&, aod::Tracks const& tracks)
  {
    auto hPhi = new TH1F("phi", "Phi", 100, 0, 2 * M_PI);
    auto hEta = new TH1F("eta", "Eta", 100, 0, 2 * M_PI);
    for (auto& track : tracks) {
      auto phi = asin(track.snp()) + track.alpha() + M_PI;
      auto eta = log(tan(0.25 * M_PI - 0.5 * atan(track.tgl())));
      hPhi->Fill(phi);
      hEta->Fill(eta);
    }
    TFile f("result1.root", "RECREATE");
    hPhi->SetName("Phi");
    hPhi->Write();
    hEta->SetName("Eta");
    hEta->Write();
  }

 private:
  int mSomeState;
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("mySimpleTrackAnalysis", 0)
  };
}
