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
#include "Analysis/TrackSelection.h"
#include "Analysis/TrackSelectionTables.h"
#include "Analysis/EventSelection.h"
#include <TH1D.h>
#include <TH2D.h>
#include <cmath>
#include <vector>
#include "TLorentzVector.h"
#include "TVector3.h"

using namespace o2;
using namespace o2::framework;

#define mpion 0.13957

struct UPCAnalysis {
  OutputObj<TH1D> hMass{TH1D("hMass", ";#it{m_{#pi#pi}}, GeV/c^{2};", 500, 0.55, 1.4)};
  OutputObj<TH1D> hPt{TH1D("hPt", ";#it{p_{T}}, GeV/c;", 500, 0., 1.)};
  OutputObj<TH2D> hSignalTPC1vsSignalTPC2{TH2D("hSignalTPC1vsSignalTPC2", ";TPC Signal 1;TPC Signal 2", 1000, 0., 200., 1000, 0., 200.)};

  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator const& col, soa::Join<aod::Tracks, aod::TracksExtra, aod::TrackSelection> const& tracks)
  {
    bool checkV0 = col.bbV0A() || col.bbV0C() || col.bgV0A() || col.bgV0C();
    if (checkV0) {
      return;
    }
    bool checkFDD = col.bbFDA() || col.bbFDC() || col.bgFDA() || col.bgFDC();
    if (checkFDD) {
      return;
    }
    if (!col.alias()[kCUP9]) {
      return;
    }
    std::vector<soa::Join<aod::Tracks, aod::TracksExtra, aod::TrackSelection>::iterator> selTracks;
    for (auto track : tracks) {
      if (!track.isGlobalTrack()) {
        continue;
      }
      selTracks.push_back(track);
      if (selTracks.size() > 2) {
        break;
      }
    }
    if (selTracks.size() != 2) {
      return;
    }
    if (selTracks[0].charge() * selTracks[1].charge() >= 0) {
      return;
    }
    UChar_t clustermap1 = selTracks[0].itsClusterMap();
    UChar_t clustermap2 = selTracks[1].itsClusterMap();
    bool checkClusMap = TESTBIT(clustermap1, 0) && TESTBIT(clustermap1, 1) && TESTBIT(clustermap2, 0) && TESTBIT(clustermap2, 1);
    if (!checkClusMap) {
      return;
    }
    TLorentzVector p1, p2, p;
    p1.SetXYZM(selTracks[0].px(), selTracks[0].py(), selTracks[0].pz(), mpion);
    p2.SetXYZM(selTracks[1].px(), selTracks[1].py(), selTracks[1].pz(), mpion);
    p = p1 + p2;
    hPt->Fill(p.Pt());
    float signalTPC1 = selTracks[0].tpcSignal();
    float signalTPC2 = selTracks[1].tpcSignal();
    hSignalTPC1vsSignalTPC2->Fill(signalTPC1, signalTPC2);
    if ((p.Pt() < 0.1) && (signalTPC1 + signalTPC2 < 140.)) {
      hMass->Fill(p.M());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<UPCAnalysis>("upc-an")};
}
