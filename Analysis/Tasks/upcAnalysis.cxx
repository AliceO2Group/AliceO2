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
#include "TLorentzVector.h"
#include "TVector3.h"

using namespace o2;
using namespace o2::framework;

#define mpion 0.13957

struct UPCAnalysis {
  OutputObj<TH1D> hMass{TH1D("hMass", ";#it{m_{#pi#pi}}, GeV/c^{2};", 500, 0.55, 1.4)};
  OutputObj<TH1D> hPt{TH1D("hPt", ";#it{p_{T}}, GeV/c;", 500, 0., 1.)};
  OutputObj<TH2D> hSignalTPC1vsSignalTPC2{TH2D("hSignalTPC1vsSignalTPC2", ";#pi^{+} TPC Signal; #pi^{-} TPC Signal", 1000, 0., 200., 1000, 0., 200.)};
  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    TLorentzVector p1, p2, p;
    TVector3 ptemp;
    double signalTPC1, signalTPC2;
    double invM;
    bool sel1 = false;
    bool sel2 = false;
    int selnum = 0;
    for (auto& track : tracks) {
      UChar_t clustermap = track.itsClusterMap();
      bool isselected = track.tpcNClsFound() > 70 && track.flags() & 0x4;
      isselected = isselected && (TESTBIT(clustermap, 0) && TESTBIT(clustermap, 1));
      if (!isselected) {
        continue;
      }
      double pt = track.pt();
      double snp = track.snp();
      double alp = track.alpha();
      double tgl = track.tgl();
      double r = sqrt((1. - snp) * (1. + snp));
      ptemp[0] = pt * (r * cos(alp) - snp * sin(alp));
      ptemp[1] = pt * (snp * cos(alp) + r * sin(alp));
      ptemp[2] = pt * tgl;
      if (track.charge() > 0) {
        selnum++;
        sel1 = true;
        p1.SetVectM(ptemp, mpion);
        signalTPC1 = track.tpcSignal();
      }
      if (track.charge() < 0) {
        selnum++;
        sel2 = true;
        p2.SetVectM(ptemp, mpion);
        signalTPC2 = track.tpcSignal();
      }
      if (track.charge() == 0) {
        selnum++;
      }
      if (selnum > 2) {
        break;
      }
    }
    if (selnum == 2 && sel1 && sel2) {
      p = p1 + p2;
      hPt->Fill(p.Pt());
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
