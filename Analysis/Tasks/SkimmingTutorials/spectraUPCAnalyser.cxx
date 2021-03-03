// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// O2 includes
#include "ReconstructionDataFormats/Track.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "DataModel/UDDerived.h"
#include "TLorentzVector.h"

#include "Framework/HistogramRegistry.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

#include "Framework/runDataProcessing.h"

struct UPCSpectraAnalyserTask {
  float mPion = 0.13957;
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(o2::framework::InitContext&)
  {
    histos.add("fhMass", "Mass of mother", kTH1F, {{500, 0.55, 1.4}});
    histos.add("fhPt", "Pt of mother", kTH1F, {{500, 0., 1.}});
  }

  void process(aod::UDTrack const& track)
  {
    TLorentzVector p1, p2, p;
    p1.SetXYZM(track.pt1() * TMath::Cos(track.phi1()), track.pt1() * TMath::Sin(track.phi1()), track.pt1() * TMath::SinH(track.eta1()), mPion);
    p2.SetXYZM(track.pt2() * TMath::Cos(track.phi2()), track.pt2() * TMath::Sin(track.phi2()), track.pt2() * TMath::SinH(track.eta2()), mPion);
    p = p1 + p2;
    histos.fill(HIST("fhPt"), p.Pt());
    float signalTPC1 = track.tpcSignal1();
    float signalTPC2 = track.tpcSignal2();
    if ((p.Pt() < 0.1) && (signalTPC1 + signalTPC2 < 140.)) {
      histos.fill(HIST("fhMass"), p.M());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<UPCSpectraAnalyserTask>("upcspectra-task-skim-analyser")};
  return workflow;
}
