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
  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    std::vector<soa::Join<aod::Tracks, aod::TracksExtra>::iterator> selTracks;
    for (auto track : tracks) {
      UChar_t clustermap = track.itsClusterMap();
      bool isselected = track.tpcNClsFound() > 70 && track.flags() & 0x4;
      isselected = isselected && (TESTBIT(clustermap, 0) && TESTBIT(clustermap, 1));
      if (!isselected)
        continue;
      selTracks.push_back(track);
      if (selTracks.size() == 2)
        break;
    }
    if (selTracks.size() == 2 && selTracks[0].charge() * selTracks[1].charge() < 0) {
      TLorentzVector p1, p2, p;
      p1.SetXYZM(selTracks[0].px(), selTracks[0].py(), selTracks[0].pz(), mpion);
      p2.SetXYZM(selTracks[1].px(), selTracks[1].py(), selTracks[1].pz(), mpion);
      p = p1 + p2;
      hPt->Fill(p.Pt());
      double signalTPC1 = selTracks[0].tpcSignal();
      double signalTPC2 = selTracks[1].tpcSignal();
      hSignalTPC1vsSignalTPC2->Fill(signalTPC1, signalTPC2);
      if ((p.Pt() < 0.1) && (signalTPC1 + signalTPC2 < 140)) {
        hMass->Fill(p.M());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<UPCAnalysis>("upc-an")};
}
