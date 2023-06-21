// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


#include "TRDBase/RawDisplay.h"
#include "TRDBase/DataManager.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"


// #include <TSystem.h>
// #include <TFile.h>
// #include <TTree.h>
// #include <TTreeReader.h>
// #include <TTreeReaderArray.h>
#include <TVirtualPad.h>
#include <TPad.h>
#include <TCanvas.h>
#include <TH2.h>
#include <TLine.h>
#include <TMarker.h>

// #include <ostream>
// #include <sstream>

using namespace o2::trd;
using namespace o2::trd::rawdisp;

/// Modified version of o2::trd::Tracklet64::getPadCol returning a float

namespace o2::trd::rawdisp 
{
float PadColF(o2::trd::Tracklet64 &tracklet)
{
  // obtain pad number relative to MCM center
  float padLocal = tracklet.getPositionBinSigned() * constants::GRANULARITYTRKLPOS;
  // MCM number in column direction (0..7)
  int mcmCol = (tracklet.getMCM() % constants::NMCMROBINCOL) + constants::NMCMROBINCOL * (tracklet.getROB() % 2);

  // original calculation
  // FIXME: understand why the offset seems to be 6 pads and not nChannels / 2 = 10.5
  //return CAMath::Nint(6.f + mcmCol * ((float)constants::NCOLMCM) + padLocal);

  // my calculation
  return float((mcmCol + 1) * constants::NCOLMCM) + padLocal - 10.0;
}

float SlopeF(o2::trd::Tracklet64 &trkl)
{
  return - trkl.getSlopeBinSigned() * constants::GRANULARITYTRKLSLOPE / constants::ADDBITSHIFTSLOPE;
}

// float UncalibratedPad(o2::trd::Tracklet64 &tracklet)
// {
//   float y = tracklet.getUncalibratedY();
//   int mcmCol = (tracklet.getMCM() % NMCMROBINCOL) + NMCMROBINCOL * (tracklet.getROB() % 2);
//   // one pad column has 144 pads, the offset of -63 is the center of the first MCM in that column
//   // which is connected to the pads -63 - 9 = -72 to -63 + 9 = -54
//   // float offset = -63.f + ((float)NCOLMCM) * mcmCol;
//   float padWidth = 0.635f + 0.03f * (tracklet.getDetector() % NLAYER);
//   return y / padWidth + 71.0;
// }

};


// TPad* rawdisp::DrawMCM(RawDataSpan &mcm, TPad *pad)
// {
//   auto x = *mcm.digits.begin();

//   int det = x.getDetector();
//   std::string name = Form("det%03d_rob%d_mcm%02d", det, x.getROB(), x.getMCM());
//   std::string desc = Form("Detector %02d_%d_%d (%03d) - MCM %d:%02d", det/30, (det%30)/6, det%6, det, x.getROB(), x.getMCM());;

//    // MCM column number on ROC [0..7]
//   int mcmcol = x.getMCM() % constants::NMCMROBINCOL + HelperMethods::getROBSide(x.getROB()) * constants::NMCMROBINCOL;

//   float firstpad = mcmcol * constants::NCOLMCM - 1;
//   float lastpad = (mcmcol+1) * constants::NCOLMCM + 2;

//   if (pad == NULL) {
//     pad = new TCanvas(name.c_str(), desc.c_str(), 800, 600);
//     std::cout << "create canvas " << desc << std::endl;
//   } else {
//     pad->SetName(name.c_str());
//     pad->SetTitle(desc.c_str());
//   }
//   pad->cd();

//   std::cout << firstpad << " - " << lastpad << std::endl;
//   TH2F *digit_disp = new TH2F(name.c_str(), (desc + ";pad;time bin").c_str(), 21, firstpad, lastpad, 30, 0., 30.);

//   for (auto digit : mcm.digits) {
//     auto adc = digit.getADC();
//     for (int tb = 0; tb < 30; ++tb) {
//       digit_disp->Fill(digit.getPadCol(), tb, adc[tb]);
//     }
//   }
//   digit_disp->SetStats(0);
//   digit_disp->Draw("colz");

//   TLine trkl;
//   trkl.SetLineColor(kRed);
//   trkl.SetLineWidth(3);

//   for (auto tracklet : mcm.tracklets) {
//     auto pos = PadColF(tracklet);
//     auto slope = SlopeF(tracklet);
//     trkl.DrawLine(pos, 0, pos + 30 * slope, 30);
//   }

//   TMarker clustermarker;
//   clustermarker.SetMarkerColor(kRed);
//   clustermarker.SetMarkerStyle(2);
//   clustermarker.SetMarkerSize(1.5);

//   TMarker cogmarker;
//   cogmarker.SetMarkerColor(kGreen);
//   cogmarker.SetMarkerStyle(3);
//   cogmarker.SetMarkerSize(1.5);
//   for(int t=1; t<=digit_disp->GetNbinsY(); ++t) {
//     for(int p=2; p<=digit_disp->GetNbinsX()-1; ++p) {
//       // cout << p << "/" << t << " -> " << digit_disp->GetBinContent(i,j) << endl;
//         double baseline = 9.5;
//         double left = digit_disp->GetBinContent(p-1,t) - baseline;
//         double centre = digit_disp->GetBinContent(p,t) - baseline;
//         double right = digit_disp->GetBinContent(p+1,t) - baseline;
//         if (centre > left && centre > right) {
//           double pos = 0.5 * log(right/left) / log(centre*centre / left / right);
//           double clpos = digit_disp->GetXaxis()->GetBinCenter(p) + pos;
//           double cog = (right - left) / (right+centre+left) + digit_disp->GetXaxis()->GetBinCenter(p);
//           // cout << "t=" << t << " p=" << p 
//           //      << ":   ADCs = " << left << " / " << centre << " / " << right
//           //      << "   pos = " << pos << " ~ " << clpos
//           //      << endl;
//           clustermarker.DrawMarker(clpos, t-0.5);
//           cogmarker.DrawMarker(cog, t-0.5);
//         }
//     }
//   }


//   // TODO: At the moment, hits are not propagated during splitting of a RawDataSpan, therefore this code does not work yet.
//   // TMarker hitmarker;
//   // hitmarker.SetMarkerColor(kBlack);
//   // hitmarker.SetMarkerStyle(38);

//   // auto ct = CoordinateTransformer::instance();
//   // for (auto hit : mcm.hits) {
//   //   std::cout << hit.GetCharge() << std::endl;
//   //   auto rct = ct->Local2RCT(hit);
//   //   hitmarker.SetMarkerSize(hit.GetCharge() / 50.);
//   //   hitmarker.DrawMarker(rct[1], rct[2]);
//   // }

//   return pad;
// }

rawdisp::RawDisplay::RawDisplay(RawDataSpan& dataspan, TVirtualPad* pad)
: mDataSpan(dataspan), mPad(pad)
{}

rawdisp::MCMDisplay::MCMDisplay(RawDataSpan& mcmdata, TVirtualPad* pad)
: rawdisp::RawDisplay(mcmdata, pad) // initializes mDataSpan, mPad
// : mDataSpan(mcmdata)
{
  int det=-1,rob=-1,mcm=-1;

  if(std::distance(mDataSpan.digits.begin(), mDataSpan.digits.end())) {
    auto x = *mDataSpan.digits.begin();
    det = x.getDetector();
    rob = x.getROB();
    mcm = x.getMCM();
  } else if(std::distance(mDataSpan.tracklets.begin(), mDataSpan.tracklets.end())) {
    auto x = *mDataSpan.tracklets.begin();
    det = x.getDetector();
    rob = x.getROB();
    mcm = x.getMCM();
  } else {
    std::cerr << "ERROR: found neither digits nor tracklets in MCM" << std::endl;
    assert(false);
  }

  mName = Form("det%03d_rob%d_mcm%02d", det, rob, mcm);
  mDesc = Form("Detector %02d_%d_%d (%03d) - MCM %d:%02d", det/30, (det%30)/6, det%6, det, rob, mcm);;

   // MCM column number on ROC [0..7]
  int mcmcol = mcm % constants::NMCMROBINCOL + HelperMethods::getROBSide(rob) * constants::NMCMROBINCOL;

  mFirstPad = mcmcol * constants::NCOLMCM - 1;
  mLastPad = (mcmcol+1) * constants::NCOLMCM + 2;

  if (pad == NULL) {
    mPad = new TCanvas(mName.c_str(), mDesc.c_str(), 800, 600);
    std::cout << "create canvas " << mDesc << std::endl;
  } else {
    mPad = pad;
    mPad->SetName(mName.c_str());
    mPad->SetTitle(mDesc.c_str());
  }
}

void rawdisp::RawDisplay::DrawDigits()
{
  mPad->cd();

  std::cout << mFirstPad << " - " << mLastPad << std::endl;
  if (mDigitsHisto) {
    delete mDigitsHisto;
  }
  mDigitsHisto = new TH2F(mName.c_str(), (mDesc + ";pad;time bin").c_str(), (mLastPad-mFirstPad), mFirstPad, mLastPad, 30, 0., 30.);

  for (auto digit : mDataSpan.digits) {
    auto adc = digit.getADC();
    for (int tb = 0; tb < 30; ++tb) {
      mDigitsHisto->Fill(digit.getPadCol(), tb, adc[tb]);
    }
  }
  mDigitsHisto->SetStats(0);
  mDigitsHisto->Draw("colz");
}

void rawdisp::RawDisplay::DrawTracklets()
{
  mPad->cd();

  TLine trkl;
  trkl.SetLineColor(kRed);
  trkl.SetLineWidth(3);

  for (auto tracklet : mDataSpan.tracklets) {
    auto pos = PadColF(tracklet);
    auto slope = SlopeF(tracklet);
    trkl.DrawLine(pos, 0, pos + 30 * slope, 30);
  }

}

void rawdisp::RawDisplay::DrawClusters()
{
  mPad->cd();

  if(!mDigitsHisto) {
    DrawDigits();
  }

  TMarker clustermarker;
  clustermarker.SetMarkerColor(kRed);
  clustermarker.SetMarkerStyle(2);
  clustermarker.SetMarkerSize(1.5);

  TMarker cogmarker;
  cogmarker.SetMarkerColor(kGreen);
  cogmarker.SetMarkerStyle(3);
  cogmarker.SetMarkerSize(1.5);
  for(int t=1; t<=mDigitsHisto->GetNbinsY(); ++t) {
    for(int p=2; p<=mDigitsHisto->GetNbinsX()-1; ++p) {
      // cout << p << "/" << t << " -> " << mDigitsHisto->GetBinContent(i,j) << endl;
        double baseline = 9.5;
        double left = mDigitsHisto->GetBinContent(p-1,t) - baseline;
        double centre = mDigitsHisto->GetBinContent(p,t) - baseline;
        double right = mDigitsHisto->GetBinContent(p+1,t) - baseline;
        if (centre > left && centre > right) {
          double pos = 0.5 * log(right/left) / log(centre*centre / left / right);
          double clpos = mDigitsHisto->GetXaxis()->GetBinCenter(p) + pos;
          double cog = (right - left) / (right+centre+left) + mDigitsHisto->GetXaxis()->GetBinCenter(p);
          // cout << "t=" << t << " p=" << p 
          //      << ":   ADCs = " << left << " / " << centre << " / " << right
          //      << "   pos = " << pos << " ~ " << clpos
          //      << endl;
          clustermarker.DrawMarker(clpos, t-0.5);
          cogmarker.DrawMarker(cog, t-0.5);
        }
    }
  }


  // TODO: At the moment, hits are not propagated during splitting of a RawDataSpan, therefore this code does not work yet.
  // TMarker hitmarker;
  // hitmarker.SetMarkerColor(kBlack);
  // hitmarker.SetMarkerStyle(38);

  // auto ct = CoordinateTransformer::instance();
  // for (auto hit : mcm.hits) {
  //   std::cout << hit.GetCharge() << std::endl;
  //   auto rct = ct->Local2RCT(hit);
  //   hitmarker.SetMarkerSize(hit.GetCharge() / 50.);
  //   hitmarker.DrawMarker(rct[1], rct[2]);
  // }

}

