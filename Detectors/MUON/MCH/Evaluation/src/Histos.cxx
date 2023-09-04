// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//_________________________________________________________________________________________________

#include "Histos.h"
#include "MCHEvaluation/ExtendedTrack.h"
#include "MCHTracking/TrackParam.h"
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TMath.h>

namespace o2::mch::eval
{
void createHistosForClusterResiduals(std::vector<TH1*>& histos, const char* extension, double range)
{
  if (histos.size() == 0) {
    for (int iSt = 1; iSt <= 5; ++iSt) {
      histos.emplace_back(new TH1F(Form("resX%sSt%d", extension, iSt),
                                   Form("#DeltaX Station %d;#DeltaX (cm)", iSt), 2000, -range, range));
      histos.emplace_back(new TH1F(Form("resY%sSt%d", extension, iSt),
                                   Form("#DeltaY Station %d;#DeltaY (cm)", iSt), 2000, -range, range));
    }
    histos.emplace_back(new TH1F(Form("resX%s", extension), "#DeltaX;#DeltaX (cm)", 2000, -range, range));
    histos.emplace_back(new TH1F(Form("resY%s", extension), "#DeltaY;#DeltaY (cm)", 2000, -range, range));
  }
}

void createHistosForTrackResiduals(std::vector<TH1*>& histos)
{
  if (histos.size() == 0) {
    histos.emplace_back(new TH1F("dx", "dx;dx (cm)", 20001, -1.00005, 1.00005));
    histos.emplace_back(new TH1F("dy", "dy;dy (cm)", 20001, -1.00005, 1.00005));
    histos.emplace_back(new TH1F("dz", "dz;dz (cm)", 20001, -1.00005, 1.00005));
    histos.emplace_back(new TH1F("dpx", "dpx;dpx (GeV/c)", 20001, -1.00005, 1.00005));
    histos.emplace_back(new TH1F("dpy", "dpy;dpy (GeV/c)", 20001, -1.00005, 1.00005));
    histos.emplace_back(new TH1F("dpz", "dpz;dpz (GeV/c)", 20001, -1.00005, 1.00005));
    histos.emplace_back(new TH2F("dpxvspx", "dpxvspx;px (GeV/c);dpx/px (\%)", 2000, 0., 20., 2001, -10.005, 10.005));
    histos.emplace_back(new TH2F("dpyvspy", "dpyvspy;py (GeV/c);dpy/py (\%)", 2000, 0., 20., 2001, -10.005, 10.005));
    histos.emplace_back(new TH2F("dpzvspz", "dpzvspz;pz (GeV/c);dpz/pz (\%)", 2000, 0., 200., 2001, -10.005, 10.005));
    histos.emplace_back(new TH2F("dslopexvsp", "dslopexvsp;p (GeV/c);dslopex", 2000, 0., 200., 2001, -0.0010005, 0.0010005));
    histos.emplace_back(new TH2F("dslopeyvsp", "dslopeyvsp;p (GeV/c);dslopey", 2000, 0., 200., 2001, -0.0010005, 0.0010005));
    histos.emplace_back(new TH2F("dpvsp", "dpvsp;p (GeV/c);dp/p (\%)", 2000, 0., 200., 2001, -10.005, 10.005));
  }
}

void createHistosAtVertex(std::vector<TH1*>& histos, const char* extension)
{
  if (histos.size() == 0) {
    histos.emplace_back(new TH1F(Form("pT%s", extension), "pT;p_{T} (GeV/c)", 300, 0., 30.));
    histos.emplace_back(new TH1F(Form("eta%s", extension), "eta;eta", 200, -4.5, -2.));
    histos.emplace_back(new TH1F(Form("phi%s", extension), "phi;phi", 360, 0., 360.));
    histos.emplace_back(new TH1F(Form("rAbs%s", extension), "rAbs;R_{abs} (cm)", 1000, 0., 100.));
    histos.emplace_back(new TH1F(Form("p%s", extension), "p;p (GeV/c)", 300, 0., 300.));
    histos.emplace_back(new TH1F(Form("dca%s", extension), "DCA;DCA (cm)", 500, 0., 500.));
    histos.emplace_back(new TH1F(Form("pDCA23%s", extension), "pDCA for #theta_{abs} < 3#circ;pDCA (GeV.cm/c)", 2500, 0., 5000.));
    histos.emplace_back(new TH1F(Form("pDCA310%s", extension), "pDCA for #theta_{abs} > 3#circ;pDCA (GeV.cm/c)", 2500, 0., 5000.));
    histos.emplace_back(new TH1F(Form("nClusters%s", extension), "number of clusters per track;n_{clusters}", 20, 0., 20.));
    histos.emplace_back(new TH1F(Form("chi2%s", extension), "normalized #chi^{2};#chi^{2} / ndf", 500, 0., 50.));
    histos.emplace_back(new TH1F(Form("matchChi2%s", extension), "normalized matched #chi^{2};#chi^{2} / ndf", 160, 0., 16.));
    histos.emplace_back(new TH1F(Form("mass%s", extension), "#mu^{+}#mu^{-} invariant mass;mass (GeV/c^{2})", 1600, 0., 20.));
  }
}

void fillHistosAtVertex(const std::list<ExtendedTrack>& tracks, const std::vector<TH1*>& histos)
{
  for (auto itTrack1 = tracks.begin(); itTrack1 != tracks.end(); ++itTrack1) {
    fillHistosMuAtVertex(*itTrack1, histos);
    for (auto itTrack2 = std::next(itTrack1); itTrack2 != tracks.end(); ++itTrack2) {
      fillHistosDimuAtVertex(*itTrack1, *itTrack2, histos);
    }
  }
}

void fillHistosMuAtVertex(const ExtendedTrack& track, const std::vector<TH1*>& histos)
{
  double thetaAbs = TMath::ATan(track.getRabs() / 505.) * TMath::RadToDeg();
  double pUncorr = std::sqrt(track.param().px() * track.param().px() + track.param().py() * track.param().py() + track.param().pz() * track.param().pz());
  double pDCA = pUncorr * track.getDCA();

  histos[0]->Fill(track.P().Pt());
  histos[1]->Fill(track.P().Eta());
  histos[2]->Fill(180. + std::atan2(-track.P().Px(), -track.P().Py()) / TMath::Pi() * 180.);
  histos[3]->Fill(track.getRabs());
  histos[4]->Fill(track.P().P());
  histos[5]->Fill(track.getDCA());
  if (thetaAbs < 3) {
    histos[6]->Fill(pDCA);
  } else {
    histos[7]->Fill(pDCA);
  }
  histos[8]->Fill(track.getClusters().size());
  histos[9]->Fill(track.getNormalizedChi2());
}

void fillHistosDimuAtVertex(const ExtendedTrack& track1, const ExtendedTrack& track2, const std::vector<TH1*>& histos)
{
  if (track1.getCharge() * track2.getCharge() < 0.) {
    ROOT::Math::PxPyPzMVector dimu = track1.P() + track2.P();
    histos[11]->Fill(dimu.M());
  }
}

void fillComparisonsAtVertex(std::list<ExtendedTrack>& tracks1,
                             std::list<ExtendedTrack>& tracks2,
                             const std::array<std::vector<TH1*>, 5>& histos)
{
  for (auto itTrack21 = tracks2.begin(); itTrack21 != tracks2.end(); ++itTrack21) {

    // fill histograms for identical, similar (from file 2) and additional muons
    if (itTrack21->hasMatchIdentical()) {
      fillHistosMuAtVertex(*itTrack21, histos[0]);
    } else if (itTrack21->hasMatchFound()) {
      fillHistosMuAtVertex(*itTrack21, histos[2]);
    } else {
      fillHistosMuAtVertex(*itTrack21, histos[3]);
    }

    for (auto itTrack22 = std::next(itTrack21); itTrack22 != tracks2.end(); ++itTrack22) {

      // fill histograms for identical, similar (from file 2) and additional dimuons
      if (itTrack21->hasMatchIdentical() && itTrack22->hasMatchIdentical()) {
        fillHistosDimuAtVertex(*itTrack21, *itTrack22, histos[0]);
      } else if (itTrack21->hasMatchFound() && itTrack22->hasMatchFound()) {
        fillHistosDimuAtVertex(*itTrack21, *itTrack22, histos[2]);
      } else {
        fillHistosDimuAtVertex(*itTrack21, *itTrack22, histos[3]);
      }
    }
  }

  for (auto itTrack11 = tracks1.begin(); itTrack11 != tracks1.end(); ++itTrack11) {

    // fill histograms for missing and similar (from file 1) muons
    if (!itTrack11->hasMatchFound()) {
      fillHistosMuAtVertex(*itTrack11, histos[4]);
    } else if (!itTrack11->hasMatchIdentical()) {
      fillHistosMuAtVertex(*itTrack11, histos[1]);
    }

    for (auto itTrack12 = std::next(itTrack11); itTrack12 != tracks1.end(); ++itTrack12) {

      // fill histograms for missing and similar (from file 1) dimuons
      if (!itTrack11->hasMatchFound() || !itTrack12->hasMatchFound()) {
        fillHistosDimuAtVertex(*itTrack11, *itTrack12, histos[4]);
      } else if (!itTrack11->hasMatchIdentical() || !itTrack12->hasMatchIdentical()) {
        fillHistosDimuAtVertex(*itTrack11, *itTrack12, histos[1]);
      }
    }
  }
}

void fillTrackResiduals(const TrackParam& param1, const TrackParam& param2, std::vector<TH1*>& histos)
{
  double p1 = TMath::Sqrt(param1.px() * param1.px() + param1.py() * param1.py() + param1.pz() * param1.pz());
  double p2 = TMath::Sqrt(param2.px() * param2.px() + param2.py() * param2.py() + param2.pz() * param2.pz());
  histos[0]->Fill(param2.getNonBendingCoor() - param1.getNonBendingCoor());
  histos[1]->Fill(param2.getBendingCoor() - param1.getBendingCoor());
  histos[2]->Fill(param2.getZ() - param1.getZ());
  histos[3]->Fill(param2.px() - param1.px());
  histos[4]->Fill(param2.py() - param1.py());
  histos[5]->Fill(param2.pz() - param1.pz());
  histos[6]->Fill(TMath::Abs(param1.px()), 100. * (param2.px() - param1.px()) / param1.px());
  histos[7]->Fill(TMath::Abs(param1.py()), 100. * (param2.py() - param1.py()) / param1.py());
  histos[8]->Fill(TMath::Abs(param1.pz()), 100. * (param2.pz() - param1.pz()) / param1.pz());
  histos[9]->Fill(p1, param2.px() / param2.pz() - param1.px() / param1.pz());
  histos[10]->Fill(p1, param2.py() / param2.pz() - param1.py() / param1.pz());
  histos[11]->Fill(p1, 100. * (p2 - p1) / p1);
}

void fillClusterClusterResiduals(const ExtendedTrack& track1, const ExtendedTrack& track2, std::vector<TH1*>& histos)
{
  for (const auto& cl1 : track1.getClusters()) {
    for (const auto& cl2 : track2.getClusters()) {
      if (cl1.getDEId() == cl2.getDEId()) {
        double dx = cl2.getX() - cl1.getX();
        double dy = cl2.getY() - cl1.getY();
        histos[cl1.getChamberId() / 2 * 2]->Fill(dx);
        histos[cl1.getChamberId() / 2 * 2 + 1]->Fill(dy);
        histos[10]->Fill(dx);
        histos[11]->Fill(dy);
      }
    }
  }
}

//_________________________________________________________________________________________________
void fillClusterTrackResiduals(const std::list<ExtendedTrack>& tracks, std::vector<TH1*>& histos, bool matched)
{
  for (const auto& track : tracks) {
    if (!matched || track.hasMatchFound()) {
      for (const auto& param : track.track()) {
        double dx = param.getClusterPtr()->getX() - param.getNonBendingCoor();
        double dy = param.getClusterPtr()->getY() - param.getBendingCoor();
        histos[param.getClusterPtr()->getChamberId() / 2 * 2]->Fill(dx);
        histos[param.getClusterPtr()->getChamberId() / 2 * 2 + 1]->Fill(dy);
        histos[10]->Fill(dx);
        histos[11]->Fill(dy);
      }
    }
  }
}

} // namespace o2::mch::eval
